# Original vs reviewed-label model comparison

Both checkpoints were evaluated after training on the same copied, hash-verified untouched test set. Test results were not used for checkpoint selection or tuning.

| Metric | Original checkpoint re-evaluated | Improved checkpoint | Difference | User-provided baseline reference |
|---|---:|---:|---:|---:|
| Precision | 0.25896 | 0.52603 | +0.26707 | 0.25896 |
| Recall | 0.71583 | 0.44159 | -0.27424 | 0.71583 |
| mAP50 | 0.37561 | 0.32036 | -0.05525 | 0.37561 |
| mAP50–95 | 0.23653 | 0.18515 | -0.05138 | 0.23653 |

## Per-class metrics

| Class | Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|
| Vegetative | Original | 0.30679 | 0.30000 | 0.31111 | 0.19834 |
| Vegetative | Improved | 1.00000 | 0.00000 | 0.26436 | 0.13569 |
| Tuber Bulking | Original | 0.28464 | 0.91892 | 0.49852 | 0.34053 |
| Tuber Bulking | Improved | 0.28291 | 0.62162 | 0.29097 | 0.17795 |
| Maturity | Original | 0.18547 | 0.92857 | 0.31720 | 0.17072 |
| Maturity | Improved | 0.29519 | 0.70315 | 0.40575 | 0.24180 |

## Artifacts

- Best checkpoint: `C:\Users\simra\potato_harvest\runs\fieldwise_yolov8n_1032\checkpoints\best.pt`
- Last checkpoint: `C:\Users\simra\potato_harvest\runs\fieldwise_yolov8n_1032\checkpoints\last.pt`
- Training plots and results: `C:\Users\simra\potato_harvest\runs\fieldwise_yolov8n_1032\training\train_batch2`
- Evaluation plots, confusion matrices, and prediction visualizations: `C:\Users\simra\potato_harvest\runs\fieldwise_yolov8n_1032\evaluation\models`

## Interpretation

Exploratory single-plot training using a mixture of verified ground truth and manually reviewed pseudo-labels. Pseudo-labels are not equivalent to independent ground truth; the internal validation is non-independent at plot level. Accuracy improvement is not claimed unless test metrics support it.
