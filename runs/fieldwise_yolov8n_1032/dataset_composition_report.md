# Reviewed-label exploratory dataset composition

This experiment uses manually reviewed pseudo-labels, which remain lower-confidence supervision than independently collected ground truth. No pending or rejected labels are used.

## Manifest audit

- Review rows: 3041
- Actual review statuses: `{'approved': 1027, 'edited': 4, 'rejected': 9, 'approved_empty': 1, 'pending': 2000}`
- Trainable reviewed items with valid image/label: 1032
- Reviewed pseudo labels selected from T1_T1: 287
- Reviewed pseudo labels excluded from T1_T2/T1_T3: 745
- Pending/rejected excluded from training.

## Parent-group split

- Plot group for training and internal validation: `T1_T1` only.
- Internal validation parent capture: `IMG20231228105347` (one parent image group; verified labels only).
- T1_T1 parent captures represented in selected tiles: 8.
- Internal validation is parent-separated but **not plot-independent**. Metrics are exploratory and cannot establish generalization to unseen plot groups.
- All T1_T2/T1_T3 reviewed images are excluded from train and validation because those plots are represented in the untouched test set.

## Split counts

| Split | Images | Boxes | Empty-label images | Vegetative | Tuber Bulking | Maturity |
|---|---:|---:|---:|---:|---:|---:|
| Train | 326 | 667 | 1 | 61 | 446 | 160 |
| Internal validation (non-independent) | 35 | 43 | 0 | 10 | 28 | 5 |
| Untouched test snapshot (evaluation only) | 59 | 85 | 0 | 20 | 37 | 28 |

## Source composition

| Split | Verified ground-truth images | Reviewed pseudo-label images |
|---|---:|---:|
| Train | 39 | 287 |
| Internal validation | 35 | 0 |

## Class balance and limitations

Training box distribution is {'Maturity': 160, 'Tuber Bulking': 446, 'Vegetative': 61}; Vegetative is the underrepresented class at 9.1% of training boxes. No oversampling or class weighting was added. Augmentations are kept mild to reduce distortion of growth-stage appearance.
Validation has 35 tiles and 43 boxes from a single parent capture, including 5 Maturity boxes. Per-class validation metrics will be noisy.
The experiment has one source plot in train/validation and two existing test plots. It can provide an exploratory comparison on the fixed test set, but three plot IDs are insufficient to estimate broad field generalization.

Test source image and label hashes are recorded in the manifest; the original test files were only read and copied to the experiment evaluation folder.
