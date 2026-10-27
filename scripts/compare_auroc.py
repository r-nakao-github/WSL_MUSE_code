#!/usr/bin/env python3
"""Compare AUROC across multiple models and experimental conditions.

Models, conditions, and outer folds are defined in a manifest CSV rather than
being hard-coded. The analysis uses a shared patient-cluster bootstrap so that
all model-condition series within an outer fold are evaluated on the same
resampled patients.

Manifest columns
----------------
model, condition, fold, csv

Prediction CSV columns
----------------------
ID, case, target, pred_1
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED_MANIFEST_COLUMNS = ("model", "condition", "fold", "csv")
REQUIRED_PREDICTION_COLUMNS = ("ID", "case", "target", "pred_1")


@dataclass
class FoldData:
    fold: str
    frame: pd.DataFrame
    y: np.ndarray
    pred: Dict[str, np.ndarray]
    class_to_cases: Dict[int, np.ndarray]
    case_to_indices: Dict[str, np.ndarray]


class InputError(RuntimeError):
    pass


def series_key(model: str, condition: str) -> str:
    return f"{model}__{condition}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AUROC across multiple models and experimental conditions "
            "using a shared patient-cluster bootstrap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns: model, condition, fold, csv.",
    )
    parser.add_argument("--output-dir", default="auroc_comparison")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(v) for v in values))


def load_manifest(
    manifest_path: Path,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    if not manifest_path.is_file():
        raise InputError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise InputError(f"{manifest_path}: missing required columns: {missing}")

    manifest = manifest[list(REQUIRED_MANIFEST_COLUMNS)].copy()
    if manifest.isna().any().any():
        raise InputError(f"{manifest_path}: manifest contains missing values.")

    for col in REQUIRED_MANIFEST_COLUMNS:
        manifest[col] = manifest[col].astype(str).str.strip()
    if (manifest[list(REQUIRED_MANIFEST_COLUMNS)] == "").any().any():
        raise InputError(f"{manifest_path}: manifest contains empty values.")

    duplicated = manifest.duplicated(["model", "condition", "fold"], keep=False)
    if duplicated.any():
        examples = manifest.loc[duplicated, ["model", "condition", "fold"]].head(10)
        raise InputError(
            "Each model-condition-fold combination must appear exactly once.\n"
            + examples.to_string(index=False)
        )

    models = ordered_unique(manifest["model"].tolist())
    conditions = ordered_unique(manifest["condition"].tolist())
    folds = ordered_unique(manifest["fold"].tolist())

    if len(models) < 1:
        raise InputError("At least one model is required.")
    if len(conditions) < 2:
        raise InputError("At least two conditions are required.")
    if len(folds) < 2:
        raise InputError("At least two outer folds are required.")

    expected = {
        (m, c, f) for m in models for c in conditions for f in folds
    }
    observed = set(
        manifest[["model", "condition", "fold"]].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise InputError(
            "The manifest must contain a complete model × condition × fold design.\n"
            f"Missing combinations (first 20): {sorted(expected - observed)[:20]}\n"
            f"Unexpected combinations (first 20): {sorted(observed - expected)[:20]}"
        )

    base_dir = manifest_path.parent
    resolved = []
    for raw in manifest["csv"]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved.append(str(path.resolve()))
    manifest["csv"] = resolved

    return manifest, models, conditions, folds


def load_prediction_csv(path: Path, model: str, condition: str) -> pd.DataFrame:
    if not path.is_file():
        raise InputError(f"Prediction CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise InputError(f"{path}: missing required columns: {missing}")

    x = df[list(REQUIRED_PREDICTION_COLUMNS)].copy()
    x.columns = ["ID", "case", "target", "score"]
    x["ID"] = x["ID"].astype(str)
    x["case"] = x["case"].astype(str)

    if x[["target", "score"]].isna().any().any():
        raise InputError(f"{path}: target or pred_1 contains missing values.")

    try:
        x["target"] = x["target"].astype(int)
        x["score"] = x["score"].astype(float)
    except Exception as exc:
        raise InputError(f"{path}: target/pred_1 conversion failed.") from exc

    targets = set(x["target"].unique().tolist())
    if not targets.issubset({0, 1}):
        raise InputError(f"{path}: target must be binary 0/1; found {sorted(targets)}")
    if not np.isfinite(x["score"].to_numpy()).all():
        raise InputError(f"{path}: pred_1 contains non-finite values.")
    if x["ID"].duplicated().any():
        dups = x.loc[x["ID"].duplicated(keep=False), "ID"].head(10).tolist()
        raise InputError(
            f"{model} / {condition} / {path}: duplicated image IDs, e.g. {dups}"
        )

    x["cluster_id"] = x["target"].astype(str) + ":" + x["case"]
    return x[["ID", "case", "target", "cluster_id", "score"]]


def build_path_lookup(manifest: pd.DataFrame) -> Dict[Tuple[str, str, str], Path]:
    return {
        (row.model, row.condition, row.fold): Path(row.csv)
        for row in manifest.itertuples(index=False)
    }


def align_fold(
    fold: str,
    lookup: Mapping[Tuple[str, str, str], Path],
    models: Sequence[str],
    conditions: Sequence[str],
) -> FoldData:
    raw: Dict[str, pd.DataFrame] = {}
    for model in models:
        for condition in conditions:
            key = series_key(model, condition)
            raw[key] = load_prediction_csv(
                lookup[(model, condition, fold)], model, condition
            )

    ref_key = series_key(models[0], conditions[0])
    ref_ids = set(raw[ref_key]["ID"])
    for model in models:
        for condition in conditions:
            key = series_key(model, condition)
            ids = set(raw[key]["ID"])
            if ids != ref_ids:
                raise InputError(
                    f"Outer fold {fold}: image-ID sets differ between {ref_key} and {key}."
                )

    ref = raw[ref_key][["ID", "case", "target", "cluster_id"]].copy()
    ref = ref.rename(
        columns={"case": "case_ref", "target": "target_ref", "cluster_id": "cluster_ref"}
    )
    merged = ref.copy()

    for model in models:
        for condition in conditions:
            key = series_key(model, condition)
            add = raw[key][["ID", "case", "target", "score"]].copy().rename(
                columns={
                    "case": "case_check",
                    "target": "target_check",
                    "score": key,
                }
            )
            merged = merged.merge(add, on="ID", how="inner", validate="one_to_one")
            bad_case = merged["case_check"].astype(str) != merged["case_ref"].astype(str)
            bad_target = merged["target_check"].astype(int) != merged["target_ref"].astype(int)
            if bad_case.any() or bad_target.any():
                bad = merged.loc[
                    bad_case | bad_target,
                    ["ID", "case_ref", "case_check", "target_ref", "target_check"],
                ].head(10)
                raise InputError(
                    f"Outer fold {fold}: case/target mismatch between {ref_key} and {key}.\n"
                    f"Examples:\n{bad.to_string(index=False)}"
                )
            merged = merged.drop(columns=["case_check", "target_check"])

    score_cols = [series_key(m, c) for m in models for c in conditions]
    frame = merged[["ID", "case_ref", "target_ref", "cluster_ref", *score_cols]].copy()
    frame = frame.rename(
        columns={"case_ref": "case", "target_ref": "target", "cluster_ref": "cluster_id"}
    )
    frame = frame.sort_values("ID").reset_index(drop=True)

    if frame["target"].nunique() != 2:
        raise InputError(f"Outer fold {fold}: both classes are required for AUROC.")

    y = frame["target"].to_numpy(dtype=int)
    pred = {c: frame[c].to_numpy(dtype=float) for c in score_cols}
    case_to_indices = {
        cid: idx.to_numpy(dtype=int)
        for cid, idx in frame.groupby("cluster_id", sort=False).groups.items()
    }
    class_to_cases: Dict[int, np.ndarray] = {}
    for target in (0, 1):
        cases = (
            frame.loc[frame["target"] == target, "cluster_id"]
            .drop_duplicates()
            .to_numpy(dtype=str)
        )
        if len(cases) == 0:
            raise InputError(f"Outer fold {fold}: no patient clusters for target={target}.")
        class_to_cases[target] = cases

    return FoldData(fold, frame, y, pred, class_to_cases, case_to_indices)


def check_cross_fold_patient_overlap(folds: Sequence[FoldData]) -> None:
    seen: Dict[str, str] = {}
    overlaps = []
    for fd in folds:
        for cid in fd.frame["cluster_id"].drop_duplicates().astype(str):
            if cid in seen:
                overlaps.append((cid, seen[cid], fd.fold))
            else:
                seen[cid] = fd.fold
    if overlaps:
        raise InputError(
            "The same patient cluster appears in more than one outer test fold. "
            f"Examples: {overlaps[:10]}"
        )


def auc_for_indices(fd: FoldData, indices: np.ndarray, key: str) -> float:
    y = fd.y[indices]
    if np.unique(y).size != 2:
        raise InputError(f"Outer fold {fd.fold}: bootstrap sample lacks one class.")
    return float(roc_auc_score(y, fd.pred[key][indices]))


def observed_fold_aucs(
    folds: Sequence[FoldData], models: Sequence[str], conditions: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for fd in folds:
        row = {
            "outer_fold": fd.fold,
            "n_images": len(fd.frame),
            "n_patients": fd.frame["cluster_id"].nunique(),
            "n_negative_patients": len(fd.class_to_cases[0]),
            "n_positive_patients": len(fd.class_to_cases[1]),
        }
        idx = np.arange(len(fd.frame), dtype=int)
        for model in models:
            for condition in conditions:
                key = series_key(model, condition)
                row[f"AUROC_{key}"] = auc_for_indices(fd, idx, key)
        rows.append(row)
    return pd.DataFrame(rows)


def resample_fold_indices(fd: FoldData, rng: np.random.Generator) -> np.ndarray:
    parts: List[np.ndarray] = []
    for target in (0, 1):
        cases = fd.class_to_cases[target]
        sampled = rng.choice(cases, size=len(cases), replace=True)
        for cid in sampled:
            parts.append(fd.case_to_indices[str(cid)])
    return np.concatenate(parts)


def run_bootstrap(
    folds: Sequence[FoldData],
    models: Sequence[str],
    conditions: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if n_bootstrap < 100:
        raise InputError("--n-bootstrap must be at least 100.")

    keys = [series_key(m, c) for m in models for c in conditions]
    key_idx = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, len(keys)), dtype=float)
    progress_every = max(1, n_bootstrap // 10)

    for b in range(n_bootstrap):
        fold_aucs = np.empty((len(folds), len(keys)), dtype=float)
        for f_idx, fd in enumerate(folds):
            idx = resample_fold_indices(fd, rng)
            for key in keys:
                fold_aucs[f_idx, key_idx[key]] = auc_for_indices(fd, idx, key)
        boot[b, :] = fold_aucs.mean(axis=0)
        if (b + 1) % progress_every == 0 or (b + 1) == n_bootstrap:
            print(f"Bootstrap: {b + 1}/{n_bootstrap}", file=sys.stderr)
    return boot, key_idx


def percentile_ci(values: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = 1.0 - ci
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def normal_approximation_p(observed: float, bootstrap_values: np.ndarray) -> Tuple[float, float, float]:
    se = float(np.std(bootstrap_values, ddof=1))
    if se == 0.0:
        if observed == 0.0:
            return 0.0, 0.0, 1.0
        return 0.0, math.copysign(math.inf, observed), 0.0
    z = observed / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return se, float(z), float(min(max(p, 0.0), 1.0))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, pv in enumerate(sorted_p):
        adj = (m - i) * pv
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def bootstrap_condition_means(
    boot_model_condition: np.ndarray,
    key_idx: Mapping[str, int],
    models: Sequence[str],
    conditions: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.empty((boot_model_condition.shape[0], len(conditions)), dtype=float)
    cond_idx = {c: i for i, c in enumerate(conditions)}
    for condition in conditions:
        cols = [key_idx[series_key(model, condition)] for model in models]
        out[:, cond_idx[condition]] = boot_model_condition[:, cols].mean(axis=1)
    return out, cond_idx


def observed_model_condition_means(
    fold_auc_df: pd.DataFrame, models: Sequence[str], conditions: Sequence[str]
) -> Dict[str, float]:
    out = {}
    for model in models:
        for condition in conditions:
            key = series_key(model, condition)
            out[key] = float(fold_auc_df[f"AUROC_{key}"].mean())
    return out


def observed_condition_means(
    model_condition_means: Mapping[str, float],
    models: Sequence[str],
    conditions: Sequence[str],
) -> Dict[str, float]:
    return {
        condition: float(
            np.mean([model_condition_means[series_key(model, condition)] for model in models])
        )
        for condition in conditions
    }


def build_model_summary(
    fold_auc_df: pd.DataFrame,
    boot_model_condition: np.ndarray,
    key_idx: Mapping[str, int],
    models: Sequence[str],
    conditions: Sequence[str],
    ci: float,
) -> pd.DataFrame:
    rows = []
    label = int(round(ci * 100))
    for model in models:
        for condition in conditions:
            key = series_key(model, condition)
            vals = fold_auc_df[f"AUROC_{key}"].to_numpy(dtype=float)
            boot = boot_model_condition[:, key_idx[key]]
            lo, hi = percentile_ci(boot, ci)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "mean_AUROC": float(vals.mean()),
                    "SD_across_outer_folds": float(vals.std(ddof=1)),
                    f"bootstrap_{label}%_CI_lower": lo,
                    f"bootstrap_{label}%_CI_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def build_condition_summary(
    fold_auc_df: pd.DataFrame,
    observed_means: Mapping[str, float],
    boot_condition: np.ndarray,
    cond_idx: Mapping[str, int],
    models: Sequence[str],
    conditions: Sequence[str],
    ci: float,
) -> pd.DataFrame:
    rows = []
    label = int(round(ci * 100))
    for condition in conditions:
        per_fold = np.vstack(
            [fold_auc_df[f"AUROC_{series_key(model, condition)}"].to_numpy(dtype=float) for model in models]
        ).mean(axis=0)
        lo, hi = percentile_ci(boot_condition[:, cond_idx[condition]], ci)
        rows.append(
            {
                "condition": condition,
                "mean_AUROC_across_models": observed_means[condition],
                "SD_across_outer_folds_of_model_mean": float(per_fold.std(ddof=1)),
                f"bootstrap_{label}%_CI_lower": lo,
                f"bootstrap_{label}%_CI_upper": hi,
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_table(
    observed_means: Mapping[str, float],
    boot_condition: np.ndarray,
    cond_idx: Mapping[str, int],
    conditions: Sequence[str],
    ci: float,
) -> pd.DataFrame:
    rows = []
    label = int(round(ci * 100))
    for a, b in itertools.combinations(conditions, 2):
        observed = observed_means[a] - observed_means[b]
        boot = boot_condition[:, cond_idx[a]] - boot_condition[:, cond_idx[b]]
        lo, hi = percentile_ci(boot, ci)
        se, z, p = normal_approximation_p(observed, boot)
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "condition_1": a,
                "condition_2": b,
                "observed_delta_AUROC_mean_across_models": observed,
                f"bootstrap_{label}%_CI_lower": lo,
                f"bootstrap_{label}%_CI_upper": hi,
                "bootstrap_SE": se,
                "z_bootstrap": z,
                "p_value_two_sided": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_value_Holm"] = holm_adjust(out["p_value_two_sided"].to_numpy(dtype=float))
    return out


def main() -> None:
    args = parse_args()
    if not (0.5 < args.ci < 1.0):
        raise InputError("--ci must be between 0.5 and 1.0.")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, models, conditions, fold_labels = load_manifest(manifest_path)

    print(f"Models: {', '.join(models)}", file=sys.stderr)
    print(f"Conditions: {', '.join(conditions)}", file=sys.stderr)
    print(f"Outer folds: {', '.join(fold_labels)}", file=sys.stderr)

    lookup = build_path_lookup(manifest)
    folds: List[FoldData] = []
    for fold in fold_labels:
        print(f"Loading and aligning outer fold {fold}...", file=sys.stderr)
        folds.append(align_fold(fold, lookup, models, conditions))

    check_cross_fold_patient_overlap(folds)
    fold_auc_df = observed_fold_aucs(folds, models, conditions)
    model_condition_means = observed_model_condition_means(fold_auc_df, models, conditions)
    condition_means = observed_condition_means(model_condition_means, models, conditions)

    print("Running shared patient-cluster bootstrap...", file=sys.stderr)
    boot_model_condition, key_idx = run_bootstrap(
        folds, models, conditions, args.n_bootstrap, args.seed
    )
    boot_condition, cond_idx = bootstrap_condition_means(
        boot_model_condition, key_idx, models, conditions
    )

    model_summary = build_model_summary(
        fold_auc_df, boot_model_condition, key_idx, models, conditions, args.ci
    )
    condition_summary = build_condition_summary(
        fold_auc_df, condition_means, boot_condition, cond_idx, models, conditions, args.ci
    )
    pairwise = build_pairwise_table(
        condition_means, boot_condition, cond_idx, conditions, args.ci
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold": out_dir / "auroc_by_outer_fold.csv",
        "model": out_dir / "auroc_summary_by_model.csv",
        "condition": out_dir / "auroc_summary_across_models.csv",
        "pairwise": out_dir / "auroc_pairwise_comparisons.csv",
    }
    fold_auc_df.to_csv(paths["fold"], index=False)
    model_summary.to_csv(paths["model"], index=False)
    condition_summary.to_csv(paths["condition"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)

    print("\n=== AUROC summary by model ===")
    print(model_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== AUROC summary averaged across models ===")
    print(condition_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== Pairwise comparisons between conditions ===")
    print(pairwise.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
