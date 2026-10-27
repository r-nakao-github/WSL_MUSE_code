from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Support execution from the project root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mil_models import build_model
from train_mil import TestDataset, amp_autocast, get_transforms


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COLOR_CHOICES = ["inferno", "grayscale"]


def load_config(config_path: str | Path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def resolve_original_image_path(
    image_folder: str | Path,
    image_id: str,
) -> Path | None:
    image_folder = Path(image_folder)
    for path in sorted(image_folder.glob(f"{image_id}.*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return path
    return None


def get_model_paths(experiment_dir: str | Path, model_pattern: str) -> list[Path]:
    experiment_dir = Path(experiment_dir)
    model_paths = sorted(experiment_dir.glob(model_pattern))
    if not model_paths:
        raise FileNotFoundError(
            f"No model files matched: {experiment_dir / model_pattern}"
        )
    return model_paths


def _load_state_dict(model, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(state, strict=True)


def load_models(cfg, model_paths: list[Path], device: torch.device):
    models = []
    for model_path in model_paths:
        model = build_model(cfg, pretrain=False).to(device)
        _load_state_dict(model, model_path, device)
        model.eval()
        models.append(model)
    return models


def _batched_bag(sample: dict, device: torch.device) -> torch.Tensor:
    bag = sample["bag"]
    if bag.ndim == 4:
        bag = bag.unsqueeze(0)
    return bag.to(device, non_blocking=True)


def compute_patch_scores(
    model,
    sample: dict,
    device: torch.device,
    *,
    target_class: int,
) -> np.ndarray:
    """Return the model-specific patch score used for map visualization.

    Mean pooling
        Patch-level target-class probability.
    ABMIL / CLAM-SB
        Pre-softmax gated-attention score.
    DSMIL
        Patch-level target-class probability from the instance stream.
    """
    bag = _batched_bag(sample, device)
    mil_name = str(model.CFG.model.mil_name).lower()

    amp_enabled = bool(model.CFG.train.amp and device.type == "cuda")
    with torch.no_grad(), amp_autocast(enabled=amp_enabled):
        if mil_name == "mean_pooling":
            features = model.encoder(bag)
            patch_logits = model.classifier(features)
            if not 0 <= target_class < patch_logits.shape[-1]:
                raise ValueError(f"Invalid target class: {target_class}")
            scores = torch.softmax(patch_logits, dim=-1)[0, :, target_class]

        elif mil_name in {"abmil", "clam_sb"}:
            out = model(bag)
            scores = out["attention_raw"][0, :, 0]

        elif mil_name == "dsmil":
            out = model(bag)
            if not 0 <= target_class < out["instance_logits"].shape[-1]:
                raise ValueError(f"Invalid target class: {target_class}")
            scores = torch.softmax(out["instance_logits"], dim=-1)[
                0, :, target_class
            ]

        else:
            raise ValueError(f"Unsupported MIL model for mapping: {mil_name}")

    return scores.detach().float().cpu().numpy()


def pad_image_with_reflection(
    image: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, int, int, int, int]:
    height, width = image.shape[:2]
    new_height = math.ceil(height / patch_size) * patch_size
    new_width = math.ceil(width / patch_size) * patch_size

    pad_h = new_height - height
    pad_w = new_width - width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT,
    )
    return padded, top, bottom, left, right


def normalize_scores(
    scores: np.ndarray,
    score_min: float,
    score_max: float,
    color: str,
    center: float,
    scale_factor: float,
    flat_value: float = 0.5,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    denominator = score_max - score_min
    if denominator <= 0:
        normalized = np.full_like(scores, flat_value, dtype=np.float32)
    else:
        normalized = (scores - score_min) / denominator

    if color == "inferno":
        return np.clip(normalized, 0.0, 1.0)

    if color == "grayscale":
        shifted = (normalized - center) * scale_factor
        return torch.sigmoid(torch.tensor(shifted, dtype=torch.float32)).numpy()

    raise ValueError(f"Unsupported color mode: {color}")


def _render_score_patch(
    overlay: np.ndarray,
    heatmap_overlay: np.ndarray,
    grayscale_heatmap: np.ndarray,
    *,
    x_pos: int,
    y_pos: int,
    patch_size: int,
    score: float,
    color: str,
) -> None:
    height, width = overlay.shape[:2]
    x1 = max(0, x_pos)
    y1 = max(0, y_pos)
    x2 = min(width, x_pos + patch_size)
    y2 = min(height, y_pos + patch_size)
    if x1 >= x2 or y1 >= y2:
        return

    if color == "inferno":
        heatmap_patch = np.full(
            (y2 - y1, x2 - x1), score, dtype=np.float32
        )
        heatmap_patch = np.uint8(np.clip(255 * heatmap_patch, 0, 255))
        heatmap_patch = cv2.applyColorMap(
            heatmap_patch, cv2.COLORMAP_INFERNO
        )
        heatmap_overlay[y1:y2, x1:x2] = heatmap_patch
        return

    alpha = float(score) * 0.5
    white_patch = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
    original_patch = overlay[y1:y2, x1:x2]
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        original_patch, 1 - alpha, white_patch, alpha, 0
    )
    grayscale_heatmap[y1:y2, x1:x2] = np.maximum(
        grayscale_heatmap[y1:y2, x1:x2], score
    )


def validate_coords(coords: np.ndarray, n_scores: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    if coords.shape[0] < n_scores:
        raise ValueError(
            f"Only {coords.shape[0]} coordinates are available for {n_scores} scores."
        )
    coords = coords[:n_scores, :2]
    if not np.isfinite(coords).all():
        raise ValueError("Patch coordinates contain non-finite values.")
    return coords


def generate_heatmap_from_coords(
    image_path: str | Path,
    scores: np.ndarray,
    coords: np.ndarray,
    *,
    patch_size: int,
    image_resize: tuple[int, int],
    color: str,
    mirroring: bool,
) -> tuple[np.ndarray, np.ndarray]:
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    width, height = image_resize
    image = cv2.resize(original_image, (width, height))

    top = bottom = left = right = 0
    if mirroring:
        image, top, bottom, left, right = pad_image_with_reflection(
            image, patch_size
        )

    canvas_h, canvas_w = image.shape[:2]
    overlay = image.copy()
    heatmap_overlay = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    grayscale_heatmap = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    coords = validate_coords(coords, len(scores))
    for score, (x_coord, y_coord) in zip(scores, coords):
        _render_score_patch(
            overlay,
            heatmap_overlay,
            grayscale_heatmap,
            x_pos=int(round(float(x_coord))),
            y_pos=int(round(float(y_coord))),
            patch_size=patch_size,
            score=float(score),
            color=color,
        )

    if color == "inferno":
        heatmap_output = heatmap_overlay.copy()
        overlay = cv2.addWeighted(image, 0.5, heatmap_overlay, 0.5, 0)
    else:
        heatmap_output = np.uint8(np.clip(255 * grayscale_heatmap, 0, 255))

    if mirroring:
        y_slice = slice(top, canvas_h - bottom if bottom else canvas_h)
        x_slice = slice(left, canvas_w - right if right else canvas_w)
        overlay = overlay[y_slice, x_slice]
        heatmap_output = heatmap_output[y_slice, x_slice]

    return overlay, heatmap_output


def collect_global_min_max(
    image_ids: Iterable[str],
    dataset: TestDataset,
    models,
    device: torch.device,
    *,
    target_class: int,
) -> tuple[float, float]:
    global_min = float("inf")
    global_max = float("-inf")

    for image_id in image_ids:
        sample = dataset.get_by_id(image_id)
        score_list = [
            compute_patch_scores(
                model,
                sample,
                device,
                target_class=target_class,
            )
            for model in models
        ]
        mean_scores = np.mean(np.stack(score_list), axis=0)
        global_min = min(global_min, float(np.min(mean_scores)))
        global_max = max(global_max, float(np.max(mean_scores)))

    return global_min, global_max


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate patch-level maps from trained MIL models."
    )
    parser.add_argument("--test-csv", required=True, help="Test metadata CSV.")
    parser.add_argument(
        "--image-folder",
        required=True,
        help="Folder containing the original test images.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory containing model weights and the saved config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <experiment-dir>/mil_maps",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Saved configuration filename. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--model-pattern",
        default="fold*_best_auc_loss.pth",
        help="Glob pattern for model files in the experiment directory.",
    )
    parser.add_argument("--image-width", type=int, default=2736)
    parser.add_argument("--image-height", type=int, default=1824)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument(
        "--color", choices=COLOR_CHOICES, default="inferno"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Target class for Mean pooling and DSMIL patch probabilities.",
    )
    parser.add_argument(
        "--center",
        type=float,
        default=0.80,
        help="Center used only for grayscale normalization.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=30.0,
        help="Scale factor used only for grayscale normalization.",
    )
    parser.add_argument(
        "--flat-value",
        type=float,
        default=0.5,
        help="Normalized value used when all selected scores are identical.",
    )
    parser.add_argument(
        "--mirroring",
        action="store_true",
        help="Apply reflection padding before mapping.",
    )
    parser.add_argument(
        "--no-global-normalization",
        action="store_true",
        help="Normalize each image independently.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image limit for a quick check.",
    )
    return parser.parse_args()


def _resolve_config_path(
    experiment_dir: Path,
    config_name: str | None,
) -> Path:
    if config_name:
        return experiment_dir / config_name

    candidates = sorted(experiment_dir.glob("config_used_exp*.yaml"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one config_used_exp*.yaml in the experiment "
            f"directory, found {len(candidates)}."
        )
    return candidates[0]


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_dir / "mil_maps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_config_path(experiment_dir, args.config_name)
    cfg = load_config(config_path)
    mil_name = str(cfg.model.mil_name).lower()

    print(f"Loaded config: {config_path}")
    print(f"MIL model: {mil_name}")

    df_test = pd.read_csv(args.test_csv)
    if args.max_images is not None:
        df_test = df_test.head(args.max_images).copy()

    test_dataset = TestDataset(
        df_test,
        cfg,
        transform=get_transforms(data="test", CFG=cfg),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = get_model_paths(experiment_dir, args.model_pattern)
    expected_models = int(cfg.general.num_folds)
    if len(model_paths) != expected_models:
        raise ValueError(
            f"Expected {expected_models} fold model(s), found {len(model_paths)} "
            f"matching {args.model_pattern!r}."
        )
    models = load_models(cfg, model_paths, device)
    print(f"Loaded {len(models)} model(s).")

    image_ids = df_test["ID"].astype(str).tolist()
    if args.no_global_normalization:
        global_min = global_max = None
        print("Using per-image normalization.")
    else:
        global_min, global_max = collect_global_min_max(
            image_ids,
            test_dataset,
            models,
            device,
            target_class=args.target_class,
        )
        print(f"global_min: {global_min}, global_max: {global_max}")

    for image_id in image_ids:
        sample = test_dataset.get_by_id(image_id)
        score_list = [
            compute_patch_scores(
                model,
                sample,
                device,
                target_class=args.target_class,
            )
            for model in models
        ]
        mean_scores = np.mean(np.stack(score_list), axis=0)

        score_min = (
            float(np.min(mean_scores))
            if args.no_global_normalization
            else float(global_min)
        )
        score_max = (
            float(np.max(mean_scores))
            if args.no_global_normalization
            else float(global_max)
        )
        normalized_scores = normalize_scores(
            mean_scores,
            score_min=score_min,
            score_max=score_max,
            color=args.color,
            center=args.center,
            scale_factor=args.scale_factor,
            flat_value=args.flat_value,
        )

        image_path = resolve_original_image_path(args.image_folder, image_id)
        if image_path is None:
            raise FileNotFoundError(f"Original image not found for ID: {image_id}")

        coords = sample["coords"].detach().cpu().numpy()
        overlay, heatmap = generate_heatmap_from_coords(
            image_path=image_path,
            scores=normalized_scores,
            coords=coords,
            patch_size=args.patch_size,
            image_resize=(args.image_width, args.image_height),
            color=args.color,
            mirroring=args.mirroring,
        )

        prefix = f"{image_id}_{mil_name}"
        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        overlay_path = output_dir / f"{prefix}_image+heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(str(overlay_path), overlay)

        print(f"Saved: {heatmap_path}")
        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
