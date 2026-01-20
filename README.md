# Cox-MT

 Cox-MT: the  Cox proportional hazards model trained with a deep semi-supervised learning approach named  the mean-teacher (MT) method.

## Overview
This repository implements:
- A Cox proportional hazards model for survival analysis.
- A mean-teacher method for semi-supervised training.
- Single-modal and multi-modal Cox-MT models.

It also includes the scripts for processing the RNA-seq and whole slide image (WSI) data from The Cancer Genome Atlas (TCGA) and the Gene Expression Omnibus (GEO) as described in the paper. These data were used to train Cox-NT models for predicitng cancer prognosis. 


## Repository layout
- `train.py`: training entrypoint (student/teacher, losses, checkpoints).
- `config.py`: paths and hyperparameters.
- `src/models.py`: model definitions.
- `src/losses.py`: Cox loss and consistency loss.
- `src/dataset.py`: dataset and dataloaders.
- `src/utils.py`: EMA and C-index utilities.
- `preprocessing/extract_rna.py`: RNA preprocessing helper.
- `preprocessing/extract_wsi.py`: WSI preprocessing helper.

## Environment setup (conda)
Use the provided conda environment file:
```bash
conda env create -f environment.yml
conda activate coxmt
```

## Data preparation
The training script expects preprocessed files under `data/processed`.

### RNA preprocessing (TCGA + GEO)
```bash
python preprocessing/extract_rna.py \
  --tcga-expression data/TCGA_BRCA_FPKM.csv \
  --tcga-clinical data/TCGA_CDR.csv \
  --geo-expression data/GSE96058_FPKM.csv \
  --output-dir data/processed
```
Outputs:
- `data/processed/tcga_rna_features.csv`
- `data/processed/tcga_clinical_labels.csv`
- `data/processed/geo_rna_unlabeled.csv` (optional)

Expected formats:
- RNA CSV: rows are samples, columns are genes (float values).
- Clinical CSV: must include `PFI` (event) and `PFI.time` (time), indexed by sample ID.

### WSI preprocessing (optional)
```bash
python preprocessing/extract_wsi.py \
  --wsi-dir data/raw_wsi \
  --output-dir data/processed_wsi
```
Outputs:
- `data/processed_wsi/<sample_id>_avg.npy` (1x1024)
- `data/processed_wsi/<sample_id>_seq.npy` (Nx1024)

Move or symlink to `data/processed/wsi_features`, or update `config.py`.

## Configuration
Edit `config.py` before training:
- `model_type`: `single_modal` or `multi_modal`.
- `wsi_feature_type`: `avg` or `seq`.
  - `multi_modal` requires `seq`.
- `tcga_rna_path`, `tcga_clinical_path`, `geo_rna_path`: file paths.
- `wsi_dir`: WSI feature directory.
- `geo_wsi_dir`: unlabeled WSI feature directory (optional).
- `device`: `cuda` or `cpu`.
- `batch_size`, `epochs`, `lr`, `ema_alpha`, `consistency_w`, `noise_sigma`.

## Training
```bash
python train.py
```
Checkpoints are saved in `checkpoints/` as `best_student.pth`.

## Manual usage workflow
1. Prepare RNA and clinical CSVs with aligned sample IDs.
2. Run RNA preprocessing to create `data/processed` outputs.
3. (Optional) Run WSI preprocessing and place outputs under `data/processed/wsi_features`.
4. Set `model_type` and `wsi_feature_type` in `config.py`.
5. Update paths in `config.py` to point to your data.
6. Run training.
7. Use `best_student.pth` as the trained model checkpoint.

## Implementation details
- **Cox loss:** `src/losses.py` implements a negative partial log-likelihood using `logcumsumexp`.
- **Consistency loss:** MSE between student and teacher predictions.
- **EMA update:** `src/utils.py` updates the teacher model weights after each optimizer step.
- **C-index:** computed on labeled training data each epoch.

## Notes and constraints
- Multi-modal training requires WSI sequence features (`*_seq.npy`) with consistent sequence lengths.
- If `geo_wsi_dir` is not provided, unlabeled data is used only for RNA.
- The WSI preprocessing script uses DINOv2 via `torch.hub` and may need network access.
- For real WSI pipelines, replace the placeholder ROI logic in `preprocessing/extract_wsi.py`.
