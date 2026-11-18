# Parent-photograph split report

| Split | Plot group | Parent capture IDs | Verified images | Reviewed pseudo images | Total images |
|---|---|---|---:|---:|---:|
| Train | T1_T1 | IMG20231227115125, IMG20231229110435, IMG20231231111259, IMG20240102112822, IMG20240104121308, IMG20240106121932, IMG20240107131745 | 39 | 287 | 326 |
| Internal validation | T1_T1 | IMG20231228105347 | 35 | 0 | 35 |
| Test (evaluation only) | T1_T2, T1_T3 | existing test capture groups | 59 | 0 | 59 |

Train and internal validation capture overlap: `[]` (must be empty).
The internal validation is not independent at plot level. T1_T2 and T1_T3 appear only in the untouched test snapshot and are never used for training, validation, checkpoint selection, or hyperparameter tuning.
