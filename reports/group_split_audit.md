# Dataset Group and Split Audit

Audit is read-only with respect to images, labels, review CSV, model weights, and existing dataset assignments.

## Summary

- Original parent photographs: **63**
- Unique source plot groups: **3**
- Verified dataset images: **299** (299 expected)
- Review manifest records: **3041**; trainable reviewed records: **1032**
- Reviewed pseudo boxes: **2193**
- Review statuses: `{'approved': 1027, 'edited': 4, 'rejected': 9, 'approved_empty': 1, 'pending': 2000}`
- Missing/invalid reviewed records: **0**

## Plot group totals

Counts include unique original parent photographs and image/box counts for each current verified split plus reviewed pseudo-label statuses.

| Plot group | Parent photos | Verified train images / boxes | Verified val images / boxes | Verified test images / boxes | Reviewed pseudo images / boxes | Classes in reviewed pseudo |
|---|---:|---:|---:|---:|---:|---|
| T1_T1 | 19 | 74 / 83 | 0 / 0 | 0 / 0 | 287 / 627 | {'Maturity': 156, 'Tuber Bulking': 427, 'Vegetative': 44} |
| T1_T2 | 23 | 90 / 128 | 0 / 0 | 28 / 38 | 381 / 789 | {'Vegetative': 157, 'Tuber Bulking': 503, 'Maturity': 129} |
| T1_T3 | 21 | 35 / 60 | 41 / 48 | 31 / 47 | 364 / 777 | {'Maturity': 140, 'Tuber Bulking': 413, 'Vegetative': 224} |

## Existing plot assignment and overlap

- Train: `['T1_T1', 'T1_T2', 'T1_T3']`
- Validation: `['T1_T3']`
- Test: `['T1_T2', 'T1_T3']`
- Train/validation overlap: `['T1_T3']`
- Train/test overlap: `['T1_T2', 'T1_T3']`
- Validation/test overlap: `['T1_T3']`

## Proposed strict assignment

- Candidate training groups outside current test: `['T1_T1']`
- Independent validation groups outside current test: `[]`
- Preserve existing test groups and samples: `['T1_T2', 'T1_T3']`
- **A complete group-disjoint three-way split is not possible with the available groups.** The two existing test groups must remain untouched. Only T1_T1 remains outside test; current validation is T1_T3, a test group. Do not train or tune on that assignment.

### Candidate split class distribution (not prepared)

The only group-disjoint training candidate is T1_T1. There is no eligible validation group. Existing test images and labels remain untouched.

| Candidate split | Plot groups | Images | Boxes | Vegetative | Tuber Bulking | Maturity |
|---|---|---:|---:|---:|---:|---:|
| Candidate train | T1_T1 | 361 | 710 | 71 | 474 | 165 |
| Validation | none available | 0 | 0 | 0 | 0 | 0 |
| Preserved existing test | T1_T2, T1_T3 | 59 | 85 | 20 | 37 | 28 |

### Class distributions in existing verified splits

| Existing split | Vegetative boxes | Tuber Bulking boxes | Maturity boxes |
|---|---:|---:|---:|
| existing_train | 95 | 129 | 47 |
| existing_val | 22 | 17 | 9 |
| existing_test | 20 | 37 | 28 |

## Parent photograph mapping and tile counts

Each row maps one original parent capture ID to its source plot folder. Tile counts list existing verified split membership and reviewed pseudo-label statuses; they are not randomly reassigned.

| Parent capture ID | Plot group | Verified train | Verified val | Verified test | Reviewed approved | Reviewed edited | Reviewed approved-empty |
|---|---|---:|---:|---:|---:|---:|---:|
| IMG20231224110533 | T1_T2 | 0 | 0 | 28 | 0 | 0 | 0 |
| IMG20231224110751 | T1_T3 | 0 | 0 | 31 | 0 | 0 | 0 |
| IMG20231225113918 | T1_T2 | 38 | 0 | 0 | 0 | 0 | 0 |
| IMG20231225114040 | T1_T3 | 35 | 0 | 0 | 0 | 0 | 0 |
| IMG20231227114230 | T1_T2 | 43 | 0 | 0 | 0 | 0 | 0 |
| IMG20231227114428 | T1_T3 | 0 | 41 | 0 | 0 | 0 | 0 |
| IMG20231227115125 | T1_T1 | 39 | 0 | 0 | 0 | 0 | 0 |
| IMG20231228105347 | T1_T1 | 35 | 0 | 0 | 0 | 0 | 0 |
| IMG20231228111007 | T1_T2 | 9 | 0 | 0 | 32 | 1 | 0 |
| IMG20231228111217 | T1_T3 | 0 | 0 | 0 | 38 | 3 | 0 |
| IMG20231229110435 | T1_T1 | 0 | 0 | 0 | 44 | 0 | 0 |
| IMG20231229112014 | T1_T2 | 0 | 0 | 0 | 44 | 0 | 0 |
| IMG20231229112233 | T1_T3 | 0 | 0 | 0 | 41 | 0 | 0 |
| IMG20231231111259 | T1_T1 | 0 | 0 | 0 | 44 | 0 | 0 |
| IMG20231231113016 | T1_T2 | 0 | 0 | 0 | 49 | 0 | 0 |
| IMG20231231113257 | T1_T3 | 0 | 0 | 0 | 46 | 0 | 0 |
| IMG20240102112822 | T1_T1 | 0 | 0 | 0 | 50 | 0 | 0 |
| IMG20240102115006 | T1_T2 | 0 | 0 | 0 | 49 | 0 | 0 |
| IMG20240102115240 | T1_T3 | 0 | 0 | 0 | 49 | 0 | 0 |
| IMG20240103114344 | T1_T2 | 0 | 0 | 0 | 49 | 0 | 0 |
| IMG20240103114547 | T1_T3 | 0 | 0 | 0 | 48 | 0 | 0 |
| IMG20240104121308 | T1_T1 | 0 | 0 | 0 | 47 | 0 | 0 |
| IMG20240104123203 | T1_T2 | 0 | 0 | 0 | 49 | 0 | 0 |
| IMG20240104123500 | T1_T3 | 0 | 0 | 0 | 54 | 0 | 0 |
| IMG20240106121932 | T1_T1 | 0 | 0 | 0 | 48 | 0 | 1 |
| IMG20240106123846 | T1_T2 | 0 | 0 | 0 | 53 | 0 | 0 |
| IMG20240106124149 | T1_T3 | 0 | 0 | 0 | 59 | 0 | 0 |
| IMG20240107131745 | T1_T1 | 0 | 0 | 0 | 53 | 0 | 0 |
| IMG20240109123048 | T1_T2 | 0 | 0 | 0 | 55 | 0 | 0 |
| IMG20240109123516 | T1_T3 | 0 | 0 | 0 | 26 | 0 | 0 |
| IMG20240112113135 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240112115343 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240112115630 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240114121818 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240114122127 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240116121452 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240116121712 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240119145149 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240119145538 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240122115923 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240122120223 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240123143907 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240123144046 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240125115144 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240126114120 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240128115636 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240130122409 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240130123959 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240130124252 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240201130711 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240201133138 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240203114404 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240203125612 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240203125843 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240205121347 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240205124640 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240205124901 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240207120121 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240207122218 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240207122513 | T1_T3 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240209113333 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240209121038 | T1_T2 | 0 | 0 | 0 | 0 | 0 | 0 |
| IMG20240211141324 | T1_T1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Missing metadata and validation

Original parent images without a unique group mapping: **0**.
Original parent images with verified tiles spanning multiple existing splits: **0**.
Reviewed-label/data issues: **0**.

## Recommendation

Collect additional independently labeled field plots, with parent-image IDs and plot/group IDs recorded at source. Preserve all images from T1_T2 and T1_T3 in the untouched test group set. At minimum, retain T1_T1 for training and obtain a separate independent plot for validation; more plots per split are preferable for stable field-level estimates. Assign groups before tiling, and keep every tile from a parent image in that parent's plot split.

Machine-readable counts and parent mapping: `reports/group_split_audit.json`.
