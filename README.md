# Weakly Supervised Learning for Bladder Cancer Classification on MUSE Images

## Overview

This repository contains code for weakly supervised classification of bladder cancer using microscopy with ultraviolet surface excitation (MUSE) images.

The pipeline supports four multiple-instance learning (MIL) approaches:

- Mean pooling
- Attention-based MIL (ABMIL)
- CLAM-SB
- DSMIL

The repository also includes utilities for patch extraction, patient-level cross-validation, nested cross-validation evaluation, patch-level score-map visualization, and AUROC comparison across experimental conditions.

The original medical image data used in the study are not included in this repository.

## Repository Structure

```text
WSL_MUSE_code/
├─ configs/
│  ├─ config_mean_pooling.yaml
│  ├─ config_abmil.yaml
│  ├─ config_clam_sb.yaml
│  └─ config_dsmil.yaml
├─ example_data/
│  ├─ test.csv
│  ├─ train.csv
│  └─ README.md
├─ modules/
│  └─ cv_splits.py
├─ scripts/
│  ├─ patch_images.py
│  ├─ train_mil.py
│  ├─ mil_models.py
│  ├─ visualize_mil_maps.py
│  ├─ evaluate_nested_predictions.py
│  └─ compare_auroc.py
├─ requirements.txt
├─ CITATION.cff
├─ LICENSE
└─ README.md
```

## Installation

A Python environment with CUDA-enabled PyTorch is recommended for model training.

```bash
git clone https://github.com/r-nakao-github/WSL_MUSE_code.git
cd WSL_MUSE_code
pip install -r requirements.txt
```

The public requirements file constrains Albumentations to the 1.x API used by the training code.

## Input Data

### Image-level metadata

Training/validation and test metadata CSV files are expected to contain:

- `ID`: image identifier
- `case`: anonymized patient/case identifier
- `target`: binary class label (`0` or `1`)

The example files under `example_data/` illustrate the expected format.

### Patch metadata

`scripts/patch_images.py` generates a `patch_metadata.csv` file containing patch filenames and spatial coordinates. The training and visualization scripts use this metadata to associate each image with its patch bag and, for map generation, with the original patch coordinates.

The patch root is expected to contain subdirectories matching the `group` column in `patch_metadata.csv`.

## Workflow

### 1. Patch extraction

Patch images are generated from original MUSE images with `scripts/patch_images.py`.

Example:

```bash
python scripts/patch_images.py \
  --input-root /path/to/original_images \
  --output-root /path/to/patched_images \
  --resize-scale 0.5 \
  --patch-size 224 \
  --overlap 0 \
  --mirroring
```

The script processes the immediate subdirectories under `--input-root` and creates corresponding patch subdirectories under `--output-root`.

### 2. Model training, validation, and test inference

Training is performed with `scripts/train_mil.py`.

Choose one of the model-specific configuration templates:

```bash
python scripts/train_mil.py --config configs/config_abmil.yaml
```

or:

```bash
python scripts/train_mil.py --config configs/config_clam_sb.yaml
python scripts/train_mil.py --config configs/config_dsmil.yaml
python scripts/train_mil.py --config configs/config_mean_pooling.yaml
```

The model architecture is selected by `model.mil_name` in the YAML file.

The training script supports two cross-validation modes:

- `fixed_blocks`: uses predefined metadata CSV files as validation folds.
- `groupkfold`: creates patient-level folds from a single metadata CSV using `GroupKFold`.

The supplied public configurations use `fixed_blocks` by default.

### Nested cross-validation used in the study

The study used five outer test folds. For each outer fold, the remaining data were divided into four predefined inner folds for model development.

Accordingly, one execution of `train_mil.py` corresponds to one outer-fold experiment:

```text
Outer fold 1
  ├─ inner fold 1
  ├─ inner fold 2
  ├─ inner fold 3
  └─ inner fold 4
       ↓
four trained models
       ↓
ensemble prediction on outer Test 1
```

The same procedure is repeated for outer Tests 1-5.

Thus, `general.num_folds: 4` in the example configurations refers to the number of inner folds within one outer-fold experiment, not the total number of outer test folds.

For each outer-fold run, update:

- `data.test_csv`
- `data.output_dir`
- `cv.fixed_blocks.block_csvs`
- patch-data paths as needed

The training script saves files including:

- `fold.csv`
- `config_used_exp*.yaml`
- `fold*_best_auc_loss.pth`
- `loss_fold*.csv`
- `oof_*.csv`
- `test_*.csv`

The test prediction is the ensemble average of the models trained in the inner folds.

## MIL Architectures

### Mean pooling

Patch features are averaged across the bag and classified with a linear classifier.

### ABMIL

ABMIL uses gated attention pooling to aggregate patch features into a bag-level representation.

### CLAM-SB

CLAM-SB uses gated attention pooling together with an instance-level clustering loss during training.

### DSMIL

DSMIL uses a dual-stream formulation combining instance-level and bag-level information.

All models use the backbone specified by `model.name` in the configuration file. The example configurations use a Swin Transformer backbone.

## Nested Cross-Validation Evaluation

`scripts/evaluate_nested_predictions.py` summarizes results across multiple outer test folds.

For each outer fold:

1. a decision threshold is derived from that outer experiment's OOF predictions;
2. the threshold is applied to the corresponding held-out outer test predictions;
3. AUROC and threshold-dependent metrics are calculated.

The default threshold method is the Youden index.

Example:

```bash
python scripts/evaluate_nested_predictions.py \
  --output-dirs \
  /path/to/results/test1 \
  /path/to/results/test2 \
  /path/to/results/test3 \
  /path/to/results/test4 \
  /path/to/results/test5 \
  --out-dir /path/to/results/nested_evaluation \
  --show-legend
```

Each supplied output directory must contain exactly one `oof_*.csv` and one `test_*.csv`.

Main outputs include:

- `metrics_per_outer_fold.csv`
- `metrics_summary_outer_folds.csv`
- `metrics_from_summed_confusion_matrix.csv`
- `combined_test_predictions_with_fold_thresholds.csv`
- `combined_oof_predictions.csv`
- `roc_curves_outer_folds.png`
- `confusion_matrix_sum_outer_folds.png`

AUROC is summarized as the mean and standard deviation across the outer test folds. Threshold-dependent metrics are also reported per outer fold, and the outer-fold confusion matrices are summed for an overall confusion-matrix summary.

## Patch-Level MIL Maps

`scripts/visualize_mil_maps.py` generates patch-level score maps from trained models.

Example:

```bash
python scripts/visualize_mil_maps.py \
  --test-csv /path/to/metadata/test.csv \
  --image-folder /path/to/original_test_images \
  --experiment-dir /path/to/results/abmil_test1 \
  --image-width 2736 \
  --image-height 1824 \
  --patch-size 224 \
  --color inferno
```

The script loads all matching inner-fold model checkpoints in the experiment directory and averages patch-level scores across the ensemble.

The score used for visualization depends on the MIL architecture:

- Mean pooling: patch-level target-class probability
- ABMIL: pre-softmax gated-attention score
- CLAM-SB: pre-softmax gated-attention score
- DSMIL: patch-level target-class probability from the instance stream

By default, scores are normalized using the global minimum and maximum across the selected test images for the experiment.

## AUROC Comparison Across Experimental Conditions

`scripts/compare_auroc.py` compares AUROC across multiple models and experimental conditions using held-out outer-fold prediction CSV files. Models, conditions, and outer folds are specified in a manifest CSV rather than being hard-coded in the script.

The manifest must contain the following columns:

- `model`: model name
- `condition`: experimental-condition name
- `fold`: outer-fold identifier
- `csv`: path to the corresponding held-out prediction CSV

Example manifest:

```csv
model,condition,fold,csv
ABMIL,Condition A,1,/path/to/abmil_condition_a_test1.csv
ABMIL,Condition A,2,/path/to/abmil_condition_a_test2.csv
ABMIL,Condition B,1,/path/to/abmil_condition_b_test1.csv
ABMIL,Condition B,2,/path/to/abmil_condition_b_test2.csv
CLAM-SB,Condition A,1,/path/to/clam_condition_a_test1.csv
CLAM-SB,Condition A,2,/path/to/clam_condition_a_test2.csv
CLAM-SB,Condition B,1,/path/to/clam_condition_b_test1.csv
CLAM-SB,Condition B,2,/path/to/clam_condition_b_test2.csv
```

Each prediction CSV must contain:

- `ID`
- `case`
- `target`
- `pred_1`

All model-condition combinations must contain the same set of outer folds. Within each outer fold, image IDs, case identifiers, and target labels must match exactly across all model-condition series. The script also checks that patient clusters do not overlap across outer test folds.

Example:

```bash
python scripts/compare_auroc.py \
  --manifest /path/to/auroc_comparison_manifest.csv \
  --n-bootstrap 10000 \
  --output-dir /path/to/auroc_comparison
```

For each bootstrap replicate and outer fold, patients are resampled with replacement within each target class. The same resampled patients are used for every model-condition series, preserving the paired structure of the comparison. AUROCs are first averaged equally across outer folds for each model-condition combination and then averaged equally across the prespecified models for each condition.

The script reports percentile bootstrap confidence intervals. Pairwise comparisons between conditions use bootstrap standard errors to calculate two-sided normal-approximation p values, followed by Holm correction across all pairwise comparisons.

Main outputs include:

- `auroc_by_outer_fold.csv`
- `auroc_summary_by_model.csv`
- `auroc_summary_across_models.csv`
- `auroc_pairwise_comparisons.csv`

## Configuration Files

Four model-specific templates are provided:

```text
configs/config_mean_pooling.yaml
configs/config_abmil.yaml
configs/config_clam_sb.yaml
configs/config_dsmil.yaml
```

Common parameters include:

- patch and metadata paths
- test metadata path
- output directory
- inner-fold definition
- image and patch sizes
- batch size
- gradient accumulation
- learning rate
- scheduler settings
- augmentation settings
- runtime worker settings

Model-specific parameters are included only where required.

The example paths are placeholders and must be replaced with local paths before execution.

## Reproducibility Notes

- Cross-validation splitting is performed at the patient/case level.
- The nested evaluation applies an independently derived OOF threshold to each corresponding outer test fold.
- Random seeds are set in the training configuration.
- The original medical image dataset is not included because it is not publicly available.
- Example metadata files are provided only to demonstrate the expected input format.

## Citation

If you use this code in your research, please cite the associated manuscript and the software release.

Software archive: https://doi.org/10.5281/zenodo.23992618

Full manuscript citation details will be added after publication.

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

This repository is released under the [MIT License](LICENSE).

The software is provided "as is", without warranty of any kind.

## Contact

For questions regarding this repository, please contact:

Ryuta Nakao  
Department of Pathology and Cell Regulation  
Kyoto Prefectural University of Medicine
