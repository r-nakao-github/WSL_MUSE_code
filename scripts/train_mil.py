from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Support execution from the project root:
#   python scripts/train_mil.py --config configs/config_abmil.yaml
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODULES_DIR = PROJECT_ROOT / "modules"
for path in (SCRIPT_DIR, MODULES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cv_splits import create_folds_by_case, create_folds_from_fixed_blocks
from mil_models import build_model


METADATA_COLUMNS = {"group", "image_id", "patch_idx", "patch_filename", "x", "y"}


def make_grad_scaler(*, enabled: bool):
    """Create a CUDA GradScaler across old and new PyTorch AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_autocast(*, enabled: bool):
    """Return a CUDA autocast context across old and new PyTorch AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(*, data: str, CFG):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if data == "train":
        transforms = [A.RandomCrop(CFG.preprocess.size, CFG.preprocess.size)]
        if CFG.aug.augmentation:
            transforms.extend(
                [
                    A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
                    A.VerticalFlip(p=CFG.aug.VerticalFlip.p),
                    A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
                    A.ShiftScaleRotate(
                        shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                        scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                        rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                        p=CFG.aug.ShiftScaleRotate.p,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                        contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                        p=CFG.aug.RandomBrightnessContrast.p,
                    ),
                    A.CLAHE(clip_limit=(1, 4), p=CFG.aug.CLAHE.p),
                    A.OneOf(
                        [
                            A.ImageCompression(
                                quality_lower=70,
                                quality_upper=100,
                                p=0.5,
                            ),
                            A.Downscale(scale_min=0.1, scale_max=0.15),
                        ],
                        p=CFG.aug.compress.p,
                    ),
                    A.CoarseDropout(
                        max_holes=CFG.aug.CoarseDropout.max_holes,
                        max_height=CFG.aug.CoarseDropout.max_height,
                        max_width=CFG.aug.CoarseDropout.max_width,
                        p=CFG.aug.CoarseDropout.p,
                    ),
                ]
            )
        transforms.extend([normalize, ToTensorV2()])
        return A.Compose(transforms)

    if data in {"valid", "test"}:
        return A.Compose(
            [
                A.CenterCrop(CFG.preprocess.size, CFG.preprocess.size),
                normalize,
                ToTensorV2(),
            ]
        )

    raise ValueError(f"Unknown transform split: {data}")


def _load_patch_metadata(metadata_path: str | Path) -> dict[str, pd.DataFrame]:
    metadata_path = Path(metadata_path).expanduser()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Patch metadata CSV not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    missing = METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Patch metadata CSV is missing columns: {', '.join(sorted(missing))}"
        )

    metadata = metadata.copy()
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["patch_idx"] = metadata["patch_idx"].astype(int)

    group_counts = metadata.groupby("image_id")["group"].nunique()
    ambiguous_ids = group_counts[group_counts > 1]
    if not ambiguous_ids.empty:
        examples = ambiguous_ids.index.astype(str).tolist()[:10]
        raise ValueError(
            "Each image_id must belong to exactly one metadata group. "
            f"IDs found in multiple groups include: {examples}"
        )

    duplicate_patches = metadata.duplicated(["image_id", "patch_idx"])
    if duplicate_patches.any():
        examples = metadata.loc[
            duplicate_patches, ["image_id", "patch_idx"]
        ].head(10).to_dict("records")
        raise ValueError(
            "Duplicate image_id/patch_idx entries were found in patch metadata. "
            f"Examples: {examples}"
        )

    metadata = metadata.sort_values(["image_id", "patch_idx"]).reset_index(drop=True)

    return {
        image_id: group.reset_index(drop=True)
        for image_id, group in metadata.groupby("image_id", sort=False)
    }


def _read_patch_image(img_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Patch image not found: {img_path}")
    return image


class PatchBagDataset(Dataset):
    """Base dataset that loads a fixed-size bag from patch metadata."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        patch_root: str | Path,
        metadata_csv: str | Path,
        n_patches: int,
        transform,
    ):
        self.df = df.reset_index(drop=True)
        self.ids = self.df["ID"].astype(str).to_numpy()
        self.patch_root = Path(patch_root)
        self.patch_meta_by_id = _load_patch_metadata(metadata_csv)
        self.n_patches = int(n_patches)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_bag(self, image_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if image_id not in self.patch_meta_by_id:
            raise KeyError(f"ID {image_id} not found in patch metadata CSV.")

        patch_df = self.patch_meta_by_id[image_id]
        if len(patch_df) < self.n_patches:
            raise ValueError(
                f"ID {image_id} has {len(patch_df)} patches, but "
                f"N_patch={self.n_patches}."
            )
        patch_df = patch_df.iloc[: self.n_patches]

        bag = []
        coords = []
        for _, row in patch_df.iterrows():
            img_path = (
                self.patch_root
                / str(row["group"])
                / str(row["patch_filename"])
            )
            image = _read_patch_image(img_path)
            image = self.transform(image=image)["image"]
            bag.append(image.float())
            coords.append([float(row["x"]), float(row["y"])])

        return torch.stack(bag), torch.tensor(coords, dtype=torch.float32)


class MILDataset(PatchBagDataset):
    """Labeled patch-bag dataset used for training and validation."""

    def __init__(self, df: pd.DataFrame, CFG, *, transform):
        super().__init__(
            df,
            patch_root=CFG.data.train_dir,
            metadata_csv=CFG.data.train_patch_metadata_csv,
            n_patches=CFG.N_patch,
            transform=transform,
        )
        self.labels = self.df["target"].to_numpy()

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        bag, _ = self._load_bag(image_id)
        return {
            "bag": bag,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "id": image_id,
        }


class TestDataset(PatchBagDataset):
    """Unlabeled patch-bag dataset used for test inference and map generation."""

    def __init__(self, df: pd.DataFrame, CFG, *, transform):
        super().__init__(
            df,
            patch_root=CFG.data.test_dir,
            metadata_csv=CFG.data.test_patch_metadata_csv,
            n_patches=CFG.N_patch,
            transform=transform,
        )

    def __getitem__(self, idx):
        return self.load_sample(self.ids[idx])

    def get_by_id(self, image_id):
        image_id = str(image_id)
        if image_id not in self.ids:
            raise ValueError(f"ID {image_id} not found in dataset.")
        return self.load_sample(image_id)

    def load_sample(self, image_id: str):
        image_id = str(image_id)
        bag, coords = self._load_bag(image_id)
        return {"bag": bag, "coords": coords, "id": image_id}


def get_cosine_scheduler(
    optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 1e-6,
):
    """Cosine LR schedule defined in optimizer-update steps.

    ``LambdaLR`` performs an initial scheduler step during construction.
    Therefore, the warmup factor uses ``current_step + 1`` so that the first
    optimizer update uses a small, non-zero learning rate and the final
    warmup update reaches the base learning rate.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps.")

    base_lr = float(optimizer.defaults["lr"])
    min_factor = float(min_lr) / base_lr

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / float(warmup_steps)

        cosine_steps = total_steps - warmup_steps
        if cosine_steps <= 1:
            return min_factor

        progress = (current_step - warmup_steps) / float(cosine_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def optimizer_steps_per_epoch(num_batches: int, ga_accum: int) -> int:
    """Return the number of optimizer updates in one epoch."""
    if num_batches < 1:
        raise ValueError("Training DataLoader must contain at least one batch.")
    if ga_accum < 1:
        raise ValueError("ga_accum must be at least 1.")
    return math.ceil(num_batches / ga_accum)


def accumulation_divisor(batch_idx: int, num_batches: int, ga_accum: int) -> int:
    """Return the number of micro-batches in the current accumulation group.

    The final group can contain fewer than ``ga_accum`` batches. Dividing the
    loss by the actual group size keeps the final optimizer update on the same
    gradient scale as all preceding updates.
    """
    remainder = num_batches % ga_accum
    if remainder and batch_idx >= num_batches - remainder:
        return remainder
    return ga_accum


def maybe_enable_grad_checkpointing(model) -> None:
    backbone = model.encoder.model
    if hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing()


def load_model_weights(model, model_path: Path, device: torch.device) -> None:
    """Load a model state dictionary saved by this training pipeline."""
    state = torch.load(model_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state_dict in checkpoint: {model_path}")
    model.load_state_dict(state, strict=True)


def train_fn(CFG, fold: int, folds: pd.DataFrame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"### fold: {fold} ###")

    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    val_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    print(
        f"Validation fold shape: {val_folds.shape}, "
        f"Train fold shape: {train_folds.shape}"
    )

    model = build_model(CFG).to(device)
    maybe_enable_grad_checkpointing(model)

    optimizer = AdamW(
        model.parameters(),
        lr=CFG.train.lr,
        weight_decay=CFG.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(CFG.train.amp and device.type == "cuda")
    scaler = make_grad_scaler(enabled=amp_enabled)
    softmax = nn.Softmax(dim=-1)

    best_auc_loss_auc = -np.inf
    best_auc_loss_loss = np.inf
    best_auc_loss_epoch = None
    best_preds = None
    best_valid_labels = None
    auc_tolerance = float(CFG.train.auc_tolerance)

    train_losses = []
    val_losses = []
    auc_scores = []

    num_workers = int(CFG.runtime.num_workers)
    pin_memory = bool(CFG.runtime.pin_memory)

    train_dataset = MILDataset(
        train_folds,
        CFG,
        transform=get_transforms(data="train", CFG=CFG),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    ga_accum = int(CFG.train.ga_accum)
    updates_per_epoch = optimizer_steps_per_epoch(len(train_loader), ga_accum)
    total_optimizer_steps = updates_per_epoch * int(CFG.train.epochs)

    if CFG.train.scheduler.name == "cosine":
        warmup_steps = 0
    elif CFG.train.scheduler.name == "cosine_warmup":
        warmup_epochs = int(CFG.train.scheduler.warmup)
        if warmup_epochs < 0 or warmup_epochs >= int(CFG.train.epochs):
            raise ValueError(
                "scheduler.warmup must be >= 0 and smaller than train.epochs."
            )
        warmup_steps = warmup_epochs * updates_per_epoch
    else:
        raise ValueError(f"Unknown scheduler: {CFG.train.scheduler.name}")

    scheduler = get_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr=float(CFG.train.scheduler.min_lr),
    )

    print(
        f"Gradient accumulation: {ga_accum} micro-batch(es) per optimizer update; "
        f"{updates_per_epoch} optimizer update(s) per epoch."
    )
    if warmup_steps > 0:
        print(
            f"LR warmup: {warmup_steps} optimizer update(s) "
            f"({CFG.train.scheduler.warmup} epoch(s))."
        )

    valid_dataset = MILDataset(
        val_folds,
        CFG,
        transform=get_transforms(data="valid", CFG=CFG),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            bags = batch["bag"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with amp_autocast(enabled=amp_enabled):
                out = model(bags, labels=labels)
                slide_loss = criterion(out["logits"], labels)
                instance_loss = out.get(
                    "instance_loss", torch.tensor(0.0, device=device)
                )
                instance_loss_weight = float(
                    getattr(CFG.train, "instance_loss_weight", 0.0)
                )
                loss = slide_loss + instance_loss_weight * instance_loss

            divisor = accumulation_divisor(
                batch_idx=batch_idx,
                num_batches=len(train_loader),
                ga_accum=ga_accum,
            )
            scaler.scale(loss / divisor).backward()

            should_step = (
                (batch_idx + 1) % ga_accum == 0
                or batch_idx == len(train_loader) - 1
            )
            if should_step:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # If AMP detected inf/NaN gradients, GradScaler skips the
                # optimizer update. Do not advance the LR schedule in that case.
                optimizer_step_skipped = amp_enabled and scale_after < scale_before
                if not optimizer_step_skipped:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Log the unscaled objective, not the accumulation-normalized loss.
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        avg_val_loss = 0.0
        logits_all = []
        labels_all = []

        for batch in tqdm(valid_loader, total=len(valid_loader)):
            bags = batch["bag"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.inference_mode(), amp_autocast(enabled=amp_enabled):
                out = model(bags, labels=labels)
                loss = criterion(out["logits"], labels)

            avg_val_loss += loss.item() / len(valid_loader)
            logits_all.append(out["logits"].detach().cpu())
            labels_all.append(labels.detach().cpu())

        logits_tensor = torch.cat(logits_all, dim=0).float()
        preds = softmax(logits_tensor).numpy()
        valid_labels = torch.cat(labels_all).numpy()
        auc_score = roc_auc_score(valid_labels, preds[:, 1])
        elapsed = time.time() - start_time

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        auc_scores.append(auc_score)

        print(f"  Epoch {epoch + 1} - val_auc_score: {auc_score:.6f}")
        print(
            f"  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.6f}, "
            f"avg_val_loss: {avg_val_loss:.6f}, time: {elapsed:.0f}s"
        )

        is_better = (
            auc_score > best_auc_loss_auc + auc_tolerance
            or (
                np.isclose(
                    auc_score,
                    best_auc_loss_auc,
                    atol=auc_tolerance,
                    rtol=0,
                )
                and avg_val_loss < best_auc_loss_loss
            )
        )
        if is_better:
            best_auc_loss_auc = auc_score
            best_auc_loss_loss = avg_val_loss
            best_auc_loss_epoch = epoch + 1
            best_preds = preds.copy()
            best_valid_labels = valid_labels.copy()
            model_path = (
                Path(CFG.data.output_dir)
                / f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth"
            )
            torch.save(model.state_dict(), model_path)
            print(
                f"  Epoch {epoch + 1} - Save Best AUC + Loss Model "
                f"(AUC: {best_auc_loss_auc:.6f}, "
                f"loss: {best_auc_loss_loss:.6f})"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(
        {
            "epoch": range(1, CFG.train.epochs + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "auc_score": auc_scores,
        }
    ).to_csv(Path(CFG.data.output_dir) / f"loss_fold{fold}.csv", index=False)

    if best_preds is None or best_valid_labels is None:
        raise RuntimeError("No best model was selected during training.")

    print(
        f"Fold {fold} Best AUC+Loss epoch: {best_auc_loss_epoch}, "
        f"AUC: {best_auc_loss_auc:.6f}, loss: {best_auc_loss_loss:.6f}"
    )

    for class_idx in range(best_preds.shape[1]):
        val_folds[f"pred_{class_idx}"] = best_preds[:, class_idx]

    return best_preds, best_valid_labels, val_folds


def infer(models, loader, device, *, amp: bool) -> np.ndarray:
    for model in models:
        model.eval()

    predictions = []
    for batch in tqdm(loader, total=len(loader)):
        bags = batch["bag"].to(device, non_blocking=True).float()

        amp_enabled = bool(amp and device.type == "cuda")
        with torch.inference_mode(), amp_autocast(enabled=amp_enabled):
            logits = torch.stack([model(bags)["logits"] for model in models]).mean(0)
            predictions.append(torch.softmax(logits, dim=-1))

    if not predictions:
        raise RuntimeError("No predictions were generated.")
    return torch.cat(predictions).cpu().numpy()


def oof_prediction(CFG, folds_df: pd.DataFrame, save_dir: str | Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    num_workers = int(CFG.runtime.num_workers)
    pin_memory = bool(CFG.runtime.pin_memory)

    preds = []
    valid_labels = []
    oof_frames = []

    for fold in range(CFG.general.num_folds):
        model_path = save_dir / f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth"

        if not model_path.exists():
            print(f"[Train] fold{fold} model not found. Starting training...")
            seed_torch(CFG.general.seed)
            fold_preds, fold_labels, oof_fold = train_fn(CFG, fold, folds_df)
        else:
            print(f"[OOF] Running inference for fold {fold}")
            val_df = folds_df[folds_df["fold"] == fold].reset_index(drop=True)

            model = build_model(CFG, pretrain=False).to(device)
            load_model_weights(model, model_path, device)

            val_dataset = MILDataset(
                val_df,
                CFG,
                transform=get_transforms(data="valid", CFG=CFG),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=CFG.train.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            fold_preds = infer(
                [model], val_loader, device, amp=bool(CFG.train.amp)
            )
            fold_labels = val_df["target"].to_numpy()
            oof_fold = pd.DataFrame(
                {
                    "ID": val_df["ID"],
                    "target": fold_labels,
                    "pred_0": fold_preds[:, 0],
                    "pred_1": fold_preds[:, 1],
                    "fold": fold,
                }
            )

        preds.append(fold_preds)
        valid_labels.append(fold_labels)
        oof_frames.append(oof_fold)

    return preds, valid_labels, pd.concat(oof_frames, ignore_index=True)


def submit(CFG, test_df: pd.DataFrame, save_dir: str | Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)

    models = []
    for fold in range(CFG.general.num_folds):
        model_path = save_dir / f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = build_model(CFG, pretrain=False).to(device)
        load_model_weights(model, model_path, device)
        model.eval()
        models.append(model)

    test_dataset = TestDataset(
        test_df,
        CFG,
        transform=get_transforms(data="test", CFG=CFG),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.test.batch_size,
        shuffle=False,
        num_workers=int(CFG.runtime.num_workers),
        pin_memory=bool(CFG.runtime.pin_memory),
    )

    test_preds = infer(
        models, test_loader, device, amp=bool(CFG.train.amp)
    )
    result = test_df.copy()
    for class_idx in range(test_preds.shape[1]):
        result[f"pred_{class_idx}"] = test_preds[:, class_idx]
    return result


def load_config(config_path: str | Path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    CFG = OmegaConf.load(config_path)
    output_dir = Path(CFG.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dst = output_dir / f"config_used_exp{CFG.general.exp_num}.yaml"
    shutil.copyfile(config_path, config_dst)
    print(f"Saved config to {config_dst}")
    return CFG


def resolve_folds_from_config(CFG) -> pd.DataFrame:
    fold_csv_path = Path(CFG.data.output_dir) / "fold.csv"
    method = str(CFG.cv.method)

    if method == "groupkfold":
        return create_folds_by_case(
            data_path=CFG.cv.groupkfold.train_csv,
            output_path=fold_csv_path,
            n_splits=CFG.general.num_folds,
        )

    if method == "fixed_blocks":
        block_csvs = list(CFG.cv.fixed_blocks.block_csvs)
        if len(block_csvs) != CFG.general.num_folds:
            raise ValueError(
                f"fixed_blocks requires exactly {CFG.general.num_folds} CSVs, "
                f"but got {len(block_csvs)}."
            )
        return create_folds_from_fixed_blocks(
            csv_paths=block_csvs,
            output_path=fold_csv_path,
        )

    raise ValueError(f"Unknown cv.method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a MIL model.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    CFG = load_config(args.config)

    folds = resolve_folds_from_config(CFG)
    print(folds.groupby(["fold", "case", "target"]).size().unstack(fill_value=0))

    preds, valid_labels, oof = oof_prediction(
        CFG,
        folds,
        save_dir=CFG.data.output_dir,
    )
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    oof_path = Path(CFG.data.output_dir) / f"oof_{CFG.general.exp_num}.csv"
    oof.to_csv(oof_path, index=False)
    print(f"OOF saved to {oof_path}")
    print(f"AUC Score_oof: {roc_auc_score(valid_labels, preds[:, 1])}")

    test_df = pd.read_csv(CFG.data.test_csv)
    test_predictions = submit(CFG, test_df, save_dir=CFG.data.output_dir)
    test_path = Path(CFG.data.output_dir) / f"test_{CFG.general.exp_num}.csv"
    test_predictions.to_csv(test_path, index=False)
    print(f"Test predictions saved to {test_path}")


if __name__ == "__main__":
    main()
