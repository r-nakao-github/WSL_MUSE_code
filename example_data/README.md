# Example data

This folder contains example metadata files that illustrate the expected CSV formats.

## Files

- `train.csv`: example metadata for training and validation
- `test.csv`: example metadata for test inference
- `auroc_comparison_manifest.csv`: example manifest for comparing AUROC across models and experimental conditions

## Image metadata

`train.csv` and `test.csv` contain the following columns:

- `ID`: image identifier without the file extension
- `case`: anonymized patient/case identifier
- `target`: binary class label (`0` or `1`)

## AUROC comparison manifest

`auroc_comparison_manifest.csv` contains the following columns:

- `model`: model name
- `condition`: experimental condition
- `fold`: outer-fold identifier
- `csv`: path to the corresponding held-out prediction CSV

The manifest must contain a complete set of model × condition × outer-fold combinations. Each prediction CSV specified in the manifest must contain `ID`, `case`, `target`, and `pred_1`.

These files are provided only as format examples and do not contain the original study data.