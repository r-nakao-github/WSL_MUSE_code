from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


REQUIRED_COLUMNS = {"ID", "case", "target"}


def _load_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}"
        )
    return df


def create_folds_by_case(
    data_path: str | Path,
    output_path: str | Path,
    n_splits: int = 4,
) -> pd.DataFrame:
    """Create case-level cross-validation folds using GroupKFold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    data = _load_csv(data_path).copy()
    n_cases = data["case"].nunique()
    if n_splits > n_cases:
        raise ValueError(
            f"n_splits ({n_splits}) cannot exceed the number of unique cases "
            f"({n_cases})."
        )

    data["fold"] = -1
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(
        splitter.split(data, y=data["target"], groups=data["case"])
    ):
        data.loc[val_idx, "fold"] = fold

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Folds saved to {output_path}")
    return data


def create_folds_from_fixed_blocks(
    csv_paths: list[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """Create folds by concatenating pre-defined CSV blocks in the given order."""
    if len(csv_paths) < 2:
        raise ValueError("csv_paths must contain at least 2 CSV files.")

    frames = []
    seen_ids = set()

    for fold, csv_path in enumerate(csv_paths):
        df = _load_csv(csv_path).copy()
        duplicate_ids = set(df["ID"]).intersection(seen_ids)
        if duplicate_ids:
            preview = ", ".join(sorted(map(str, list(duplicate_ids)[:10])))
            raise ValueError(
                f"Duplicate IDs found across fixed fold CSVs. Examples: {preview}"
            )

        seen_ids.update(df["ID"].tolist())
        df["fold"] = fold
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Fixed folds saved to {output_path}")
    return data
