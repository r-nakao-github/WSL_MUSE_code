from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
METADATA_FIELDS = [
    "group",
    "image_id",
    "patch_idx",
    "patch_filename",
    "x",
    "y",
    "patch_size",
    "stride",
    "resize_scale",
    "mirroring",
    "original_width",
    "original_height",
    "resized_width",
    "resized_height",
    "processed_width",
    "processed_height",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
]


def pad_to_patch_size_with_reflection(image, patch_size: int):
    """Pad an image by reflection to multiples of patch_size."""
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
    return padded, {
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
    }


def _validate_params(resize_scale: float, patch_size: int, overlap: int) -> None:
    if resize_scale <= 0:
        raise ValueError("resize_scale must be > 0.")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0.")
    if not 0 <= overlap < patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size.")


def _iter_image_files(input_folder: Path):
    return sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def split_images_in_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    resize_scale: float = 0.5,
    patch_size: int = 256,
    overlap: int = 0,
    mirroring: bool = False,
) -> list[dict]:
    """Split all images in one folder and return patch metadata records."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    _validate_params(resize_scale, patch_size, overlap)
    stride = patch_size - overlap
    image_files = _iter_image_files(input_folder)
    if not image_files:
        print(f"[WARNING] No supported image files found in: {input_folder}")
        return []

    records = []
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        original_height, original_width = image.shape[:2]
        resized_width = int(original_width * resize_scale)
        resized_height = int(original_height * resize_scale)
        if resized_width <= 0 or resized_height <= 0:
            raise ValueError(f"Invalid resized shape for image: {image_path}")

        image = cv2.resize(image, (resized_width, resized_height))
        padding = {
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        if mirroring:
            image, padding = pad_to_patch_size_with_reflection(image, patch_size)

        processed_height, processed_width = image.shape[:2]
        if processed_height < patch_size or processed_width < patch_size:
            raise ValueError(
                f"Image is smaller than patch_size after resizing: {image_path.name}"
            )

        patch_idx = 0
        for y in range(0, processed_height - patch_size + 1, stride):
            for x in range(0, processed_width - patch_size + 1, stride):
                patch = image[y : y + patch_size, x : x + patch_size]
                patch_name = f"{image_path.stem}_{patch_idx}.png"
                save_path = output_folder / patch_name
                if not cv2.imwrite(str(save_path), patch):
                    raise IOError(f"Failed to save patch: {save_path}")

                records.append(
                    {
                        "group": output_folder.name,
                        "image_id": image_path.stem,
                        "patch_idx": patch_idx,
                        "patch_filename": patch_name,
                        "x": x,
                        "y": y,
                        "patch_size": patch_size,
                        "stride": stride,
                        "resize_scale": resize_scale,
                        "mirroring": int(mirroring),
                        "original_width": original_width,
                        "original_height": original_height,
                        "resized_width": resized_width,
                        "resized_height": resized_height,
                        "processed_width": processed_width,
                        "processed_height": processed_height,
                        **padding,
                    }
                )
                patch_idx += 1

        print(f"{patch_idx} patches saved for {image_path.name}")

    return records


def write_metadata_csv(records: list[dict], metadata_csv: str | Path) -> None:
    """Write patch metadata to CSV."""
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    with metadata_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Patch metadata saved to {metadata_csv}")


def process_all_subfolders(
    input_root: str | Path,
    output_root: str | Path,
    *,
    patch_size: int = 256,
    overlap: int = 0,
    resize_scale: float = 0.5,
    mirroring: bool = False,
    metadata_csv: str | Path | None = None,
) -> list[dict]:
    """Process each immediate subfolder of input_root as one image group."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {input_root}")

    subfolders = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not subfolders:
        raise ValueError(f"No subfolders found in: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for input_folder in subfolders:
        print(f"Processing folder: {input_folder}")
        all_records.extend(
            split_images_in_folder(
                input_folder,
                output_root / input_folder.name,
                resize_scale=resize_scale,
                patch_size=patch_size,
                overlap=overlap,
                mirroring=mirroring,
            )
        )

    metadata_csv = metadata_csv or output_root / "patch_metadata.csv"
    write_metadata_csv(all_records, metadata_csv)
    return all_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image patches and save their coordinates."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root folder containing subfolders of input images.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root folder for extracted patches.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Metadata CSV path. Default: <output-root>/patch_metadata.csv",
    )
    parser.add_argument("--resize-scale", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Patch overlap in pixels. Stride = patch_size - overlap.",
    )
    parser.add_argument(
        "--mirroring",
        action="store_true",
        help="Pad images by reflection before patch extraction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_all_subfolders(
        input_root=args.input_root,
        output_root=args.output_root,
        resize_scale=args.resize_scale,
        patch_size=args.patch_size,
        overlap=args.overlap,
        mirroring=args.mirroring,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
