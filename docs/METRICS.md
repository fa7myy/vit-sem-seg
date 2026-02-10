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

## Where Metrics Are Logged

`run_experiment.py` writes all metric artifacts to:

```text
<output-dir>/<run-name>/
```

Evaluation computation and metric helpers are implemented in `eval_utils.py`.

Primary files:
- `train_metrics.csv`: epoch loss + epoch duration (for loss curves and stability analysis).
- `eval_metrics.csv`: `pixel_acc`, `mIoU`, `mean_class_acc`, inference timing, throughput.
  Use `--no-measure-inference-time` if you want to disable synchronized timing overhead.
- `class_metrics_epoch_XXX.csv`: per-class IoU and class accuracy.
- `confusion_matrix_epoch_XXX.csv`: class confusion matrix for each evaluation point.
- `summary.json`: best mIoU epoch, final metrics, and optional epochs-to-target-mIoU.
- `run_config.json`: full experiment configuration, seed, and environment/library versions.
  Parameter counts are always included; add `--profile-flops` to log FLOPs per image.
