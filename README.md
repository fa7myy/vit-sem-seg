# Vision Encoder Ablations for Segmentation

This repository contains the experimental codebase for studying the effect of pretrained vision encoders such as CLIP, DINOv2, and MAE on semantic segmentation.

---

## Research Goal

This project investigates the following question:

How much do pretrained vision encoders help segmentation?

The study measures the impact of different encoders on:

- Training speed and convergence
- Data efficiency in low data regimes
- Final segmentation accuracy
- Stability across random seeds
- Cross dataset generalization

Neck and head are kept fixed. Only the encoder is varied.

---

## Repository Structure

```
|-- README.md                     # Project overview and usage
|-- pyproject.toml                # Minimal Python package metadata
|-- run_experiment.py             # Experiment runner (swappable ViT-B backbones)
|-- eval_utils.py                 # Evaluation helper utilities (metrics, confusion matrix, FLOPs)
|-- thesis.egg-info/              # Local packaging metadata (generated)
|-- ViT-Adapter/                  # Upstream ViT-Adapter code (detection/segmentation/ops)
|   |-- detection/
|   |-- segmentation/
|   `-- wsdm2023/
`-- build/                        # (optional) build artifacts
```

---

## Environment Setup

Minimal setup (CUDA 11.8+ recommended):
1. Create env: `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows).
2. Install core deps: `pip install torch torchvision timm`.
3. Install OpenMMLab stack that ViT-Adapter expects: `pip install mmcv-full==1.4.2 mmsegmentation==0.20.2 mmdet==2.22.0 yapf==0.40.1`.
4. Build deformable attention ops used by ViT-Adapter:
   ```bash
   cd ViT-Adapter/detection/ops
   bash make.sh
   cd ../../..
   ```

## Running Experiments

### Entry Point
- `run_experiment.py`: standalone runner for linear-probe segmentation with swappable ViT-B backbones (DINOv2 / CLIP / MAE) using ViT-Adapter plus a 1×1 pixel head.

### Quick Sanity Check (no dataset)
```bash
python run_experiment.py --backbone dinov2 --dry-run
```
Prints output tensor shape to confirm the pipeline loads and runs.

### Full Training (VOC 2012)
```bash
python run_experiment.py \
  --data-root /path/to/VOCdevkit \
  --backbone dinov2 \
  --timm-model vit_base_patch14_dinov2.lvd142m \
  --img-size 512 \
  --batch-size 2 \
  --epochs 10 \
  --seed 42 \
  --output-dir runs
```
Notes:
- `--download/--no-download` toggles torchvision auto-download of VOC (defaults on).
- `--freeze-backbone` freezes the backbone (default is no freeze / backbone trainable).
- `--seed` controls Python/NumPy/PyTorch RNG seeds.
- `--deterministic` enables deterministic kernels for stricter reproducibility (typically slower).
- `--measure-inference-time/--no-measure-inference-time` controls synchronized eval timing.
- `--profile-flops` optionally estimates FLOPs per image (requires `fvcore`).
- `--save` saves both final and best checkpoints inside `<run_dir>/checkpoints/` as `<run_name>_final.pth` and `<run_name>_best.pth`.
- If interrupted with `Ctrl+C` while `--save` is enabled, an interrupted checkpoint is written to `<run_dir>/checkpoints/<run_name>_interrupted.pth`.

### Full Evaluation
```bash
python run_experiment.py \
  --data-root /path/to/VOCdevkit \
  --backbone clip \
  --timm-model clip_vit_base_patch16_224.openai \
  --img-size 512 \
  --eval-only
```

### Structured Logging (default on)

Each run writes artifacts under:

```text
<output-dir>/<run-name>/
```

Useful flags:
- `--output-dir runs` base folder for experiment artifacts.
- `--run-name clip_seed42_ft` explicit run folder name.
- `--save-logs/--no-save-logs` enable or disable JSON/CSV logging.
- `--target-miou 0.60` optional threshold used to compute epochs-to-converge.

Logged artifacts include:
- `run_config.json` full args + resolved backbone source + environment versions + dataset metadata.
- `load_report.json` matched/missing/unexpected checkpoint key statistics.
- `train_metrics.csv` epoch-level training loss and epoch time.
- `eval_metrics.csv` epoch-level `pixel_acc`, `mIoU`, `mean_class_acc`, inference timing.
- `confusion_matrix_epoch_XXX.csv` and `class_metrics_epoch_XXX.csv` for class-wise error analysis.
- `summary.json` final run summary (best mIoU epoch, final metrics, convergence info).
- `events.log` timestamped console log mirror.

### Backbones
- `--backbone dinov2` (default timm model: `vit_base_patch14_dinov2.lvd142m`, pretrain size 592)
- `--backbone clip` (default timm model: `clip_vit_base_patch16_224.openai`, pretrain size 224)
- `--backbone mae`  (default timm model: `mae_vit_base_patch16`, pretrain size 224)
- Custom weights: pass `--ckpt /path/to/model.pth` (overrides timm).

---

## Reproducibility

- All hyperparameters are stored in configuration files
- Random seeds are explicitly set
- Environment versions are logged at runtime
- Results are saved with encoder dataset and seed identifiers

This enables exact reproduction of all experiments.

---

## Evaluation Metrics

Models are evaluated across accuracy, efficiency, and learning behavior using the following metrics.

### Segmentation Quality
- **Mean Intersection over Union (mIoU)**  
  Measures overlap between predicted and ground truth masks. Primary accuracy metric.
- **Mean Class Accuracy**  
  Average per class accuracy. Evaluates performance balance across classes.

### Error Analysis
- **Confusion Matrix**  
  Shows class level prediction errors and label confusions.

### Efficiency
- **Inference Time**  
  Average time to process a single image.
- **Parameters and FLOPs**  
  Measures model size and computational cost.

### Training Dynamics
- **Epochs to Converge**  
  Number of epochs required to reach a target performance.
- **Loss Curves**  
  Tracks training stability and convergence behavior.

### Data Efficiency
- **mIoU at k Percent Data**  
  Performance when trained on k percent of the dataset. Evaluates low data learning ability.

All metrics are computed under identical training and evaluation settings for fair comparison.

---

## Dependencies

This project depends on the following external libraries:

- torch, torchvision
- timm
- mmcv-full==1.4.2
- mmdet==2.22.0
- mmsegmentation==0.20.2
- yapf (for config formatting; optional)

---

## Notes


- All encoder comparisons are conducted under identical training conditions
- ViT-Adapter code is vendored in `ViT-Adapter/` and left unmodified; the runner imports it in-place
- Ops must be built once per machine (see Environment Setup)

---

## Thesis Context

This repository supports the thesis titled:

How Much Do Vision Encoders Help Segmentation?

The codebase is designed to meet academic standards for controlled experimentation reproducibility and clarity of methodology.
