# Fieldwise — Agriculture Intelligence

Fieldwise is a local Flask research prototype for reviewing visible potato growth stages in field images and organizing a provisional harvest-planning estimate. It uses the project's existing YOLOv8n checkpoint and class mapping:

| ID | Stage |
|---:|---|
| 0 | Vegetative |
| 1 | Tuber Bulking |
| 2 | Maturity |

## Current project structure

```text
potato_harvest/
├── app/                         Flask application, routes, templates, and static assets
├── dataset/RGB images/          Original photographs and source plot folders
├── potato_yolo_dataset/
│   ├── data.yaml                Existing verified train/val/test definition
│   ├── images/{train,val,test}/ Verified dataset images
│   └── labels/{train,val,test}/ Verified YOLO labels
├── preannotations/
│   ├── labels/                  Original model-generated pseudo-labels (leave unchanged)
│   ├── metadata/                Model detections and confidence values
│   ├── approved_labels/         Human-approved/corrected YOLO labels
│   ├── manual_approval.csv      Human review status, reviewer, and timestamp
│   ├── progress.json            Pre-annotation processing records
│   └── reports/                 Pre-annotation summaries and exception lists
├── tiles_640_final/             Source tiles used for pseudo-annotation
├── notebooks/                   Training and evaluation notebook
├── scripts/                     Dataset audit, annotation, pre-annotation, and training tools
├── reports/                      Dataset audit outputs
├── runs/
│   ├── potato_yolov8n/           Original baseline checkpoint
│   ├── potato_yolov8n_finetune/  Previous separate training experiments
│   └── fieldwise_yolov8n_1032/   Reviewed-label T1_T1 exploratory experiment
├── verified_annotations/        Separate annotation workspace outputs
├── instance/outputs/             Local Flask assessment records and rendered files
├── tests/                         Flask, dataset, and pseudo-review tests
├── annotations.csv               Existing annotation source
├── run.py                         Flask application entry point
└── README.md
```

This tree lists the principal project data and code locations; individual folders also contain generated images, labels, reports, and experiment artifacts.

## Reviewed pseudo-label training status (2026-11-02)

The current review manifest contains 3,041 unique rows: 1,027 `approved`, 4 `edited`, 1 `approved_empty`, 9 `rejected`, and 2,000 `pending`. All 1,032 trainable review records have a readable source image and a valid saved label under `preannotations/approved_labels/`; the approved-empty record has an empty label. No duplicate image hashes were found among the reviewed items or against the existing 299 verified images. This reflects a mix of human-approved pseudo-labels and edits; it does not make pseudo-labels equivalent to independently verified ground truth.

There are only three source plots. Existing training includes `T1_T1`, `T1_T2`, and `T1_T3`; validation includes `T1_T3`; and test includes `T1_T2` and `T1_T3`. No fully plot-disjoint three-way split is possible while preserving the test plots. An explicitly exploratory run was completed using only T1_T1 for training and parent-grouped internal validation; it does not establish plot-independent validation or generalization to unseen field plots. See the experiment section below.

## Run the application

Use the existing Python environment with its compatible PyTorch/CUDA and Ultralytics installation. The project does not reinstall or change those packages.

```powershell
python -m pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:5001>. The model is loaded from `runs/potato_yolov8n/weights/best.pt`; set `POTATO_MODEL_PATH` to use another checkpoint. PyTorch CUDA is used when available, with CPU fallback. Inference runs one image at a time to limit VRAM use.

The assessment form accepts up to 12 JPG, JPEG, or PNG images at 8 MB each, plus field name/ID, required planting date, optional variety/location/notes, and an optional assumed crop duration from 60 to 180 days. The browser previews images before submission and validates type, count, and size; the server repeats the checks and verifies that each image can be decoded.

## Application workflow and storage

- Upload multiple images or run the project demo on three images from the existing YOLO test split.
- View original and annotated images, stage counts, bounding-box labels, per-box confidence, and per-image results. A zero-detection image remains visible and is reported as having no detections.
- Review growth-stage shares across the analyzed image sample. These percentages describe model boxes, not the percentage of the entire crop at each stage.
- Browse and search saved assessments from **Prediction history**. Dashboard totals are calculated from all readable saved assessment records.
- Print a report from the results page or download a standalone HTML report with embedded original and annotated images.
- Assessments are stored locally as JSON with image files under `instance/outputs/<assessment-id>/`. This is file-backed local persistence, not a remote account database.
- The Agri-IoT workbook, when present, is presented as historical context. It is not a live sensor feed and is not associated with a current assessment.

## Provisional harvest window

Fieldwise does not contain a trained harvest-time model. When a planting date, user-supplied assumed crop duration, and at least one detection are available, the application applies the existing transparent rule: it starts with planting date plus the assumed duration, shifts the calendar midpoint by at most 21 days using the observed stage mix, and applies a ±14-day planning margin. No variety-specific or regional agronomic data is invented or prefilled.

This is a preliminary planning estimate. YOLO detects visible stages; it does not directly predict days until harvest. Confirm readiness in the field and use agronomic judgment. Detection confidence is not model accuracy or harvest reliability.

## Current model and training experiment

The Flask application continues to use the original checkpoint at `runs/potato_yolov8n/weights/best.pt`. The new exploratory YOLOv8n experiment is in `runs/fieldwise_yolov8n_1032/`; it fine-tuned from that original checkpoint without overwriting it.

The manifest had 1,027 approved, 4 edited, and 1 approved-empty pseudo-labels. Only 287 reviewed images assigned to T1_T1 were eligible for this run. The other 745 reviewed images from T1_T2/T1_T3 were excluded because those groups occur in the untouched test set. The experiment combined 39 verified T1_T1 training images with those 287 reviewed pseudo-label images, including one negative image. Validation used 35 verified images (43 boxes) from parent capture `IMG20231228105347`; the other seven represented T1_T1 parent captures stayed in train. Train/validation are parent-disjoint but share plot T1_T1, so this is **not independent plot validation**. Eleven of the 19 T1_T1 source photographs have no usable training/approved tile in this experiment.

Training used CUDA on the RTX 3050 Laptop GPU, 640-pixel input, batch 2, AMP enabled, and mild augmentation. Ultralytics skipped its optional AMP helper-model check because the environment was offline; AMP remained enabled and the recorded losses were finite. It early-stopped after 21 epochs (best internal validation checkpoint at epoch 1). Training contains 326 images and 667 boxes: 61 Vegetative, 446 Tuber Bulking, and 160 Maturity; Vegetative is underrepresented. The single-parent internal validation has only 5 Maturity boxes, so its class metrics are noisy.

Both checkpoints were evaluated after training on the same hash-verified copy of the untouched 59-image, 85-box test set. The original test files were not modified:

| Metric | Original YOLOv8n | Reviewed-label YOLOv8n | Difference |
|---|---:|---:|---:|
| Precision | 0.25896 | 0.52603 | +0.26707 |
| Recall | 0.71583 | 0.44159 | −0.27424 |
| mAP50 | 0.37561 | 0.32036 | −0.05525 |
| mAP50–95 | 0.23653 | 0.18515 | −0.05138 |

The experiment does **not** demonstrate improved accuracy: both mAP metrics and recall fell, despite higher precision. Per-class test metrics show the new model had zero Vegetative recall at the evaluation threshold. Do not replace the original application checkpoint with this experiment.

The dataset YAML, exact image/label manifest, dataset report, parent-group assignment report, configuration, log, best/last checkpoints, training loss/metric curves, PR curves, confusion matrices, and test prediction visualizations are saved under `runs/fieldwise_yolov8n_1032/`. The detailed comparison is `evaluation/comparison_report.md` and `.json`.

To reproduce in a new output directory (the script refuses to overwrite existing runs):

```powershell
python scripts/train_reviewed_t1_t1.py --experiment runs/fieldwise_yolov8n_1032_repeat
```

The script validates review counts, corrected labels, class IDs, duplicate image content, plot membership, and the parent-group holdout before creating the experiment. It evaluates the original and new models on the test set only after training; test results are not used for model selection. The experiment cannot establish broad generalization from one training plot, and reviewed pseudo-labels are not equivalent to independently verified ground truth.

## Dataset audit and assisted annotation

### Visual pseudo-annotation review

The Flask application provides a visual review queue at <http://127.0.0.1:5001/review>. Start it with `python run.py`, then open **Pseudo-label review** in the sidebar. The page shows pending, approved, rejected, edited, and approved-empty counts, review progress, filename search, and status filters. Add a reviewer name before recording decisions.

- Switch between the original tile and the YOLO prediction overlay; use the zoom controls to inspect small objects.
- Press **A** to approve non-empty predictions, **R** to reject them, **E** to enter edit mode, **N/P** to navigate, and **S** to save edited boxes. In edit mode, draw a box by dragging, move a box by dragging its body, resize using its lower-right handle, select/change its class, or delete it.
- For an image with no predicted boxes, use **Approve Empty** only after confirming there are no relevant objects.
- Decisions are human actions; loading or viewing an item never approves it. Pending and rejected predictions are not training labels. The notebook accepts only rows marked `approved`, `approved_empty`, or `edited` with reviewer and timestamp, and uses the corresponding file under `preannotations/approved_labels/`. Reviewed labels remain separate from original predictions in `preannotations/labels/`; original images and verified labels are not edited.
- Review statuses and reviewer/timestamp values are stored in `preannotations/manual_approval.csv`. Corrected YOLO labels are atomically written under `preannotations/approved_labels/`, retaining the image's relative path. `approved` copies the suggested boxes only after human confirmation; `edited` saves the corrected boxes; `approved_empty` saves an empty label file. Rejected and pending items are excluded by the training notebook.

The review UI is an annotation aid. Human approval does not establish that labels are scientifically verified ground truth; inspect corrections carefully and retain group leakage safeguards when preparing training data.

Run the read-only dataset audit from the project root:

```powershell
python scripts/audit_dataset.py
```

Audit parent-image-to-plot mappings and whether the current groups support a leakage-safe split:

```powershell
python scripts/audit_group_splits.py
```

This writes `reports/group_split_audit.md` and `reports/group_split_audit.json`. The current audit finds only three source plots and no complete train/validation/test assignment that both preserves the existing test plots and keeps every plot group disjoint.

It writes `reports/dataset_audit.md` and `reports/dataset_audit.json`. The existing verified dataset has 299 images and 404 object instances across its train, validation, and test assignments. The audit checks YOLO label integrity, missing/orphan labels, box bounds and size distributions, class counts by split, duplicates, capture-prefix overlap, RGB/preprocessed pairing, and the historical green-mask behavior.

To manually review model suggestions, start the annotation workspace:

```powershell
python scripts/annotation_tool.py
```

Open <http://127.0.0.1:5055>. Suggestions remain pending until a person accepts/rejects them or draws corrected boxes. Review outputs are stored separately under `verified_annotations/`; original images, `annotations.csv`, existing labels, and model weights are preserved.

## Notebook experiment

`notebooks/potato-maturity-detection-yolov9.ipynb` contains the reproducible data inspection, verified-label preparation, training, and paired test evaluation workflow. The notebook uses only existing verified labels and explicitly approved corrected pseudo-labels. It will report unreviewed suggestions but not train on them. Run its cells in order. Each experiment writes to a new timestamped directory beneath `runs/potato_yolov8n_finetune/` and does not overwrite the original model.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover the Flask navigation, missing-model behavior, server-side upload validation, multiple image processing, zero detections, stored image URLs, history search, report generation, harvest-estimate inputs, dataset audit, and visual pseudo-annotation review including edit persistence, approve/reject, empty approval, and duplicate-record safeguards. A separate real-checkpoint upload test is part of the deployment verification for this project; it requires the trained checkpoint and compatible local ML environment.
