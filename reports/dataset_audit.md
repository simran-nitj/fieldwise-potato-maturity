# Dataset audit

Read-only audit of the existing YOLO dataset. The `test` split was not modified or used for model selection.

## Split and class counts

| Split | Images | Instances | Vegetative | Tuber Bulking | Maturity | Empty or unlabeled | Capture groups |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 199 | 271 | 95 | 129 | 47 | 0 | 6 |
| val | 41 | 48 | 22 | 17 | 9 | 0 | 1 |
| test | 59 | 85 | 20 | 37 | 28 | 0 | 2 |

## Integrity findings

- **Images without label files:** 0
- **Missing image references from annotations.csv:** 0
- **Label files without images:** 0
- **Invalid label rows:** 0
- **Unreadable images:** 0

## Split leakage and duplicate checks

- Capture-prefix overlap: `{"train-val": [], "train-test": [], "val-test": []}`
- Source plot groups by split: `{"train": ["T1_T1", "T1_T2", "T1_T3"], "val": ["T1_T3"], "test": ["T1_T2", "T1_T3"]}`
- Source plot-group overlap: `{"train-val": ["T1_T3"], "train-test": ["T1_T2", "T1_T3"], "val-test": ["T1_T3"]}`
- Exact duplicate image pairs across splits: 0
- Near-duplicate pairs across splits (pHash Hamming distance ≤ 5): 0
- Similar-image class-count mismatches (within or across splits): 0

## Bounding-box size

Values are width/height in pixels and area as a fraction of image area.
- **Width (px):** `{'min': 113.0003, 'median': 323.4998, 'p90': 451.4, 'max': 637.0003}`
- **Height (px):** `{'min': 95.0003, 'median': 435.0, 'p90': 631.7001, 'max': 640.0}`
- **Area fraction:** `{'min': 0.0535, 'median': 0.3036, 'p90': 0.6085, 'max': 0.9922}`

## RGB vs green-filtered preprocessing

- Matched full-resolution RGB/preprocessed pairs: 63
- Notebook mask bounds: HSV [25, 40, 30]–[95, 255, 255]
- Non-black pixel fraction in preprocessed images: `{'min': 0.1047, 'median': 0.5109, 'p90': 0.7168, 'max': 0.8308}`
- Retention among yellow/brown-like pixels: `{'min': 0.0128, 'median': 0.079, 'p90': 0.2113, 'max': 0.2752}`
- Interpretation: the inspected preprocessed image keeps green plants while masking soil/background. The hue lower bound of 25 excludes many yellow colors and browning is also outside the green band, so maturity cues may be suppressed. The pixel statistic includes soil and is not foliage-specific. Keep RGB images available for a controlled comparison; do not replace current model inputs based on this audit alone.

## Recommendations

1. Preserve the existing test files and labels byte-for-byte. Capture prefixes are disjoint, but source `T1_T*` plot groups overlap across train/validation/test. Treat this as a field-group leakage risk and use plot-level grouping for any future split redesign.
2. Maturity is underrepresented in the current training annotations. Add manually reviewed training examples from diverse maturity stages, including yellowing/browning foliage, while holding test and validation groups out.
3. Inspect any invalid boxes and suspected similar-image label mismatches before excluding or correcting them. Do not automatically repair annotations from model predictions.
4. Maintain both original RGB and masked views. Compare them in a later controlled experiment using the same split and evaluation protocol.
5. The current dataset uses only a small number of capture groups; add independent field/date groups before interpreting test metrics as generalization across farms or seasons.

## CSV cross-check

`annotations.csv` contains 404 rows; parsed class counts: `{"Tuber Bulking": 183, "Vegetative": 137, "Maturity": 84}`.
