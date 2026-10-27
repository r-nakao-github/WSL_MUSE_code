#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

mpl.rcParams["axes.unicode_minus"] = False

REQUIRED_COLUMNS = {"target", "pred_0", "pred_1"}


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load binary-classification predictions from a CSV file."""
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    y = df["target"].astype(int).to_numpy()
    p0 = df["pred_0"].astype(float).to_numpy()
    p1 = df["pred_1"].astype(float).to_numpy()

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"target must contain only 0 and 1 in {csv_path}.")

    if np.any((p0 < 0) | (p0 > 1) | (p1 < 0) | (p1 > 1)):
        raise ValueError(
            f"pred_0 and pred_1 must contain probabilities in [0, 1] "
            f"in {csv_path}."
        )

    if not np.allclose(p0 + p1, 1.0, atol=1e-3):
        raise ValueError(
            f"pred_0 + pred_1 must be approximately 1 in {csv_path}."
        )

    return y, p1


def select_single_csv(output_dir: Path, pattern: str, label: str) -> Path:
    """Require exactly one prediction CSV matching a pattern."""
    candidates = sorted(output_dir.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {label} CSV matching '{pattern}' "
            f"in {output_dir}, found {len(candidates)}."
        )
    return candidates[0]


def choose_threshold(
    y: np.ndarray,
    p: np.ndarray,
    mode: str = "youden",
    fixed_thr: float = 0.5,
    target: float = 0.9,
) -> float:
    """Determine a classification threshold from OOF predictions."""
    if mode == "fixed":
        return float(fixed_thr)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Threshold selection requires both classes in the OOF predictions."
        )

    fpr, tpr, thresholds = roc_curve(y, p)

    # The first ROC threshold is inf and is not a usable classifier threshold.
    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]

    if len(thresholds) == 0:
        raise ValueError("No finite ROC threshold could be determined.")

    if mode == "youden":
        return float(thresholds[np.argmax(tpr - fpr)])

    if mode == "sens_at":
        return float(thresholds[np.argmin(np.abs(tpr - target))])

    if mode == "spec_at":
        specificity = 1.0 - fpr
        return float(thresholds[np.argmin(np.abs(specificity - target))])

    raise ValueError(f"Unknown threshold mode: {mode}")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Compute binary-classification metrics from confusion-matrix counts."""
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "N": int(tp + tn + fp + fn),
    }


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Compute fold-level metrics using a supplied threshold."""
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    metrics = metrics_from_counts(
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )
    metrics["AUC"] = float(roc_auc_score(y, p))
    metrics["threshold"] = float(threshold)
    return metrics


def infer_label(output_dir: Path, index: int) -> str:
    """Infer a readable outer-fold label from the output directory name."""
    match = re.search(r"test(\d+)", output_dir.name, flags=re.IGNORECASE)
    return f"Test {match.group(1)}" if match else f"Outer fold {index + 1}"


def summary_stats(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def save_roc_plot(
    fold_rows: list[dict],
    out_path: Path,
    *,
    title: str,
    pad: float,
    show_legend: bool,
) -> None:
    """Save overlaid ROC curves for all outer test folds."""
    colors = ["#E69F00", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]

    plt.figure(figsize=(2.7, 2.6), dpi=600)
    plt.plot([0, 1], [0, 1], linestyle=":", lw=1, color="#888888")

    for index, row in enumerate(fold_rows):
        fpr, tpr, _ = roc_curve(row["y"], row["p"])
        plt.plot(
            fpr,
            tpr,
            lw=1.2,
            color=colors[index % len(colors)],
            label=f"{row['label']} (AUC={row['AUC']:.4f})",
        )

    plt.xlim(-pad, 1 + pad)
    plt.ylim(-pad, 1 + pad)
    plt.xticks(np.arange(0, 1.02, 0.2), fontsize=7)
    plt.yticks(np.arange(0, 1.02, 0.2), fontsize=7)
    plt.xlabel("False positive rate", fontsize=9)
    plt.ylabel("True positive rate", fontsize=9)
    plt.title(title, fontsize=12)

    if show_legend:
        plt.legend(loc="lower right", fontsize=7, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    *,
    positive_class_name: str,
    negative_class_name: str,
) -> None:
    """Save the summed outer-fold confusion matrix.

    The matrix is expected in labels=[1, 0] order:
        [[TP, FN],
         [FP, TN]]
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label("Row-normalized value", fontsize=8)

    labels = [positive_class_name, negative_class_name]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11, labelpad=6)
    ax.set_ylabel("True class", fontsize=11, labelpad=8)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10,
                fontweight="bold" if i == j else "normal",
            )

    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate nested cross-validation outputs using an OOF-derived "
            "threshold for each outer test fold."
        )
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        required=True,
        help=(
            "Output directories for the outer folds. Each directory must "
            "contain exactly one oof_*.csv and one test_*.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for aggregated evaluation outputs. "
            "Default: <first-output-dir>/nested_eval"
        ),
    )
    parser.add_argument(
        "--thr-mode",
        default="youden",
        choices=["fixed", "youden", "sens_at", "spec_at"],
        help="Threshold-selection method applied independently to each OOF set.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Fixed threshold used when --thr-mode=fixed.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.9,
        help=(
            "Target sensitivity or specificity used for "
            "--thr-mode=sens_at/spec_at."
        ),
    )
    parser.add_argument(
        "--title",
        default="ROC curves across outer test folds",
        help="Title of the overlaid ROC figure.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.02,
        help="Axis padding for the ROC figure.",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show fold labels and AUROCs in the ROC figure.",
    )
    parser.add_argument(
        "--font-family",
        default="Arial",
        help="Matplotlib font family.",
    )
    parser.add_argument(
        "--positive-class-name",
        default="Cancerous",
        help="Display name for class 1 in the confusion matrix.",
    )
    parser.add_argument(
        "--negative-class-name",
        default="Non-cancerous",
        help="Display name for class 0 in the confusion matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams["font.family"] = args.font_family

    output_dirs = [Path(path).expanduser().resolve() for path in args.output_dirs]

    if len(output_dirs) < 2:
        raise ValueError(
            "--output-dirs must contain at least two outer-fold directories."
        )

    for output_dir in output_dirs:
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output directory not found: {output_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else output_dirs[0] / "nested_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    per_fold_metrics = []
    combined_test_frames = []
    combined_oof_frames = []
    cm_sum = np.zeros((2, 2), dtype=int)

    for index, output_dir in enumerate(output_dirs):
        label = infer_label(output_dir, index)

        oof_path = select_single_csv(output_dir, "oof_*.csv", "OOF")
        test_path = select_single_csv(output_dir, "test_*.csv", "test")

        print(f"[INFO] {label}")
        print(f"       OOF : {oof_path}")
        print(f"       Test: {test_path}")

        y_oof, p_oof = load_predictions(oof_path)
        y_test, p_test = load_predictions(test_path)

        if len(np.unique(y_test)) < 2:
            raise ValueError(
                f"{test_path} must contain both classes to calculate AUROC."
            )

        threshold = choose_threshold(
            y_oof,
            p_oof,
            mode=args.thr_mode,
            fixed_thr=args.thr,
            target=args.target,
        )

        metrics = binary_metrics(y_test, p_test, threshold)
        metrics.update(
            {
                "outer_fold": label,
                "output_dir": str(output_dir),
                "oof_csv": str(oof_path),
                "test_csv": str(test_path),
                "threshold_source": "this_outer_fold_oof",
            }
        )
        per_fold_metrics.append(metrics)

        yhat = (p_test >= threshold).astype(int)

        # labels=[1, 0]:
        # [[TP, FN],
        #  [FP, TN]]
        cm_fold = confusion_matrix(y_test, yhat, labels=[1, 0])
        cm_sum += cm_fold

        fold_rows.append(
            {
                "label": label,
                "y": y_test,
                "p": p_test,
                "AUC": metrics["AUC"],
            }
        )

        test_df = pd.read_csv(test_path).copy()
        test_df["outer_fold"] = label
        test_df["threshold"] = threshold
        test_df["pred_label"] = yhat
        combined_test_frames.append(test_df)

        oof_df = pd.read_csv(oof_path).copy()
        oof_df["outer_fold"] = label
        combined_oof_frames.append(oof_df)

        print(
            f"       AUC={metrics['AUC']:.6f}, "
            f"threshold={threshold:.6f}, "
            f"sensitivity={metrics['sensitivity']:.3f}, "
            f"specificity={metrics['specificity']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}"
        )

    per_fold_df = pd.DataFrame(per_fold_metrics)

    metric_names = [
        "AUC",
        "sensitivity",
        "specificity",
        "accuracy",
        "precision",
        "npv",
        "f1",
        "balanced_accuracy",
        "threshold",
    ]

    summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                **summary_stats(per_fold_df[metric].astype(float).tolist()),
            }
            for metric in metric_names
        ]
    )

    # cm_sum uses labels=[1, 0]:
    # [[TP, FN],
    #  [FP, TN]]
    combined_metrics = metrics_from_counts(
        tp=int(cm_sum[0, 0]),
        fn=int(cm_sum[0, 1]),
        fp=int(cm_sum[1, 0]),
        tn=int(cm_sum[1, 1]),
    )

    auc_summary = summary_df.loc[summary_df["metric"] == "AUC"].iloc[0]

    combined_metrics.update(
        {
            "threshold_source": "outer_fold_specific_oof_thresholds",
            "n_outer_folds": len(output_dirs),
            "AUC_mean_outer_folds": float(auc_summary["mean"]),
            "AUC_sd_outer_folds": float(auc_summary["sd"]),
        }
    )

    roc_path = out_dir / "roc_curves_outer_folds.png"
    cm_path = out_dir / "confusion_matrix_sum_outer_folds.png"
    per_fold_path = out_dir / "metrics_per_outer_fold.csv"
    summary_path = out_dir / "metrics_summary_outer_folds.csv"
    combined_metrics_path = out_dir / "metrics_from_summed_confusion_matrix.csv"
    combined_test_path = out_dir / "combined_test_predictions_with_fold_thresholds.csv"
    combined_oof_path = out_dir / "combined_oof_predictions.csv"

    save_roc_plot(
        fold_rows,
        roc_path,
        title=args.title,
        pad=args.pad,
        show_legend=args.show_legend,
    )

    save_confusion_matrix(
        cm_sum,
        cm_path,
        positive_class_name=args.positive_class_name,
        negative_class_name=args.negative_class_name,
    )

    per_fold_df.to_csv(per_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame([combined_metrics]).to_csv(combined_metrics_path, index=False)
    pd.concat(combined_test_frames, ignore_index=True).to_csv(
        combined_test_path, index=False
    )
    pd.concat(combined_oof_frames, ignore_index=True).to_csv(
        combined_oof_path, index=False
    )

    print()
    print(
        "[RESULT] AUC mean ± SD = "
        f"{combined_metrics['AUC_mean_outer_folds']:.6f} ± "
        f"{combined_metrics['AUC_sd_outer_folds']:.6f}"
    )
    print(
        "[RESULT] Summed confusion matrix: "
        f"sensitivity={combined_metrics['sensitivity']:.3f}, "
        f"specificity={combined_metrics['specificity']:.3f}, "
        f"accuracy={combined_metrics['accuracy']:.3f}, "
        f"N={combined_metrics['N']}"
    )
    print()
    print(f"[SAVE] {roc_path}")
    print(f"[SAVE] {cm_path}")
    print(f"[SAVE] {per_fold_path}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {combined_metrics_path}")
    print(f"[SAVE] {combined_test_path}")
    print(f"[SAVE] {combined_oof_path}")


if __name__ == "__main__":
    main()
