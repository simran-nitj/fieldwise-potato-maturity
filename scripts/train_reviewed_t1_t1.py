"""Exploratory YOLOv8n fine-tuning using reviewed labels from plot T1_T1 only.

This script refuses to overwrite its experiment directory, validates every
trainable review record, keeps all tiles from a capture together, snapshots the
existing test split into the experiment, trains, and only then evaluates both
models on that isolated test copy.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "runs" / "fieldwise_yolov8n_1032"
VERIFIED = ROOT / "potato_yolo_dataset"
TILES = ROOT / "tiles_640_final"
PRE = ROOT / "preannotations"
REVIEW_CSV = PRE / "manual_approval.csv"
APPROVED = PRE / "approved_labels"
RGB_ROOT = ROOT / "dataset" / "RGB images"
ORIGINAL_MODEL = ROOT / "runs" / "potato_yolov8n" / "weights" / "best.pt"
CLASS_NAMES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAPTURE_RE = re.compile(r"IMG[0-9]{14}", re.IGNORECASE)
ALLOWED_STATUS = {"approved", "edited", "approved_empty"}
EXPECTED_STATUS = {"approved": 1027, "edited": 4, "approved_empty": 1, "rejected": 9, "pending": 2000}
TEST_IMAGE_ROOT = VERIFIED / "images" / "test"
TEST_LABEL_ROOT = VERIFIED / "labels" / "test"
LOG_PATH = EXPERIMENT / "training.log"
DATASET_ROOT = EXPERIMENT / "dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"
CONFIG_PATH = EXPERIMENT / "training_config.json"
MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
REPORT_PATH = EXPERIMENT / "dataset_composition_report.md"
SPLIT_REPORT_PATH = EXPERIMENT / "parent_group_split_report.md"


def configure_experiment(value: str) -> None:
    global EXPERIMENT, LOG_PATH, DATASET_ROOT, DATA_YAML, CONFIG_PATH, MANIFEST_PATH, REPORT_PATH, SPLIT_REPORT_PATH
    requested = Path(value)
    resolved = (ROOT / requested).resolve() if not requested.is_absolute() else requested.resolve()
    if not resolved.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Experiment outputs must remain under the project runs/ directory")
    EXPERIMENT = resolved
    LOG_PATH = EXPERIMENT / "training.log"
    DATASET_ROOT = EXPERIMENT / "dataset"
    DATA_YAML = DATASET_ROOT / "data.yaml"
    CONFIG_PATH = EXPERIMENT / "training_config.json"
    MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
    REPORT_PATH = EXPERIMENT / "dataset_composition_report.md"
    SPLIT_REPORT_PATH = EXPERIMENT / "parent_group_split_report.md"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    os.replace(temp, path)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def capture_id(value: Path | str) -> str | None:
    match = CAPTURE_RE.search(Path(value).stem)
    return match.group(0).upper() if match else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_yolo(path: Path) -> list[int]:
    class_ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO values")
        values = [float(value) for value in fields]
        raw_class, cx, cy, width, height = values
        if not raw_class.is_integer() or int(raw_class) not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: invalid class ID")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite value")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized coordinates")
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box exceeds image bounds")
        class_ids.append(int(raw_class))
    return class_ids


def discover_plot_groups() -> tuple[dict[str, str], dict[str, set[str]]]:
    capture_to_group: dict[str, str] = {}
    capture_to_paths: dict[str, set[str]] = defaultdict(set)
    for image in sorted(path for path in RGB_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        cid = capture_id(image)
        if not cid:
            raise ValueError(f"Original parent image lacks capture ID: {image}")
        group = image.parent.name
        old = capture_to_group.get(cid)
        if old is not None and old != group:
            raise ValueError(f"Capture ID {cid} appears under conflicting plot folders: {old}, {group}")
        capture_to_group[cid] = group
        capture_to_paths[cid].add(str(image.relative_to(ROOT)))
    return capture_to_group, capture_to_paths


def collect_reviewed(capture_to_group: dict[str, str], verified_hashes: dict[str, str]):
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row.get("image_path", "") for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("manual_approval.csv has duplicate image_path rows")
    statuses = Counter(row.get("review_status", "").strip().lower() for row in rows)
    if dict(statuses) != EXPECTED_STATUS:
        raise ValueError(f"Review status counts differ from audited expectation: {dict(statuses)}")

    reviewed = []
    hash_owners: dict[str, str] = {}
    group_counts = defaultdict(lambda: {"images": 0, "boxes": 0, "classes": Counter(), "statuses": Counter()})
    errors = []
    for row in rows:
        status = row.get("review_status", "").strip().lower()
        if status not in ALLOWED_STATUS:
            continue
        rel = Path(row["image_path"])
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"Unsafe image path: {rel}")
            continue
        cid = capture_id(rel)
        group = capture_to_group.get(cid or "")
        source = TILES / rel
        label = (APPROVED / rel).with_suffix(".txt")
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            errors.append(f"{rel}: missing reviewer or review timestamp")
        if not source.is_file():
            errors.append(f"{rel}: missing source image")
            continue
        if not label.is_file():
            errors.append(f"{rel}: missing reviewed label")
            continue
        decoded = cv2.imread(str(source))
        if decoded is None:
            errors.append(f"{rel}: unreadable source image")
            continue
        class_ids = valid_yolo(label)
        if status in {"approved", "edited"} and not class_ids:
            errors.append(f"{rel}: {status} has an empty label")
        if status == "approved_empty" and class_ids:
            errors.append(f"{rel}: approved_empty label is not empty")
        digest = sha256(source)
        if digest in hash_owners:
            errors.append(f"Duplicate reviewed image content: {hash_owners[digest]} and {rel}")
        if digest in verified_hashes:
            errors.append(f"Duplicate reviewed/verified image content: {rel} and {verified_hashes[digest]}")
        hash_owners[digest] = str(rel)
        group_key = group or "UNKNOWN"
        group_counts[group_key]["images"] += 1
        group_counts[group_key]["boxes"] += len(class_ids)
        group_counts[group_key]["classes"].update(CLASS_NAMES[c] for c in class_ids)
        group_counts[group_key]["statuses"][status] += 1
        reviewed.append({"image_path": rel, "image": source, "label": label, "status": status, "capture_id": cid, "plot_group": group, "class_ids": class_ids, "source_kind": "reviewed_pseudo"})
    if errors:
        raise ValueError("Review audit failed:\n- " + "\n- ".join(errors[:100]))
    if len(reviewed) != sum(EXPECTED_STATUS[k] for k in ALLOWED_STATUS):
        raise ValueError(f"Expected 1,032 trainable review records, found {len(reviewed)}")
    return rows, statuses, group_counts, reviewed


def collect_verified_train(capture_to_group: dict[str, str], verified_hashes: dict[str, str]):
    items = []
    hashes: dict[str, str] = {}
    for image in sorted(path for path in (VERIFIED / "images" / "train").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        cid = capture_id(image)
        group = capture_to_group.get(cid or "")
        if group != "T1_T1":
            continue
        label = VERIFIED / "labels" / "train" / image.relative_to(VERIFIED / "images" / "train").with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Verified train label missing for {image}")
        if cv2.imread(str(image)) is None:
            raise ValueError(f"Verified image unreadable: {image}")
        class_ids = valid_yolo(label)
        digest = sha256(image)
        if digest in hashes:
            raise ValueError(f"Duplicate verified image content: {image} and {hashes[digest]}")
        hashes[digest] = str(image)
        items.append({"image": image, "label": label, "capture_id": cid, "plot_group": group, "class_ids": class_ids, "source_kind": "verified_ground_truth", "status": "verified"})
    if len(items) != 74:
        raise ValueError(f"Expected 74 existing verified T1_T1 training images, found {len(items)}")
    return items


def make_item_counts(items):
    classes = Counter(CLASS_NAMES[c] for item in items for c in item["class_ids"])
    return {"images": len(items), "boxes": sum(len(item["class_ids"]) for item in items), "empty_images": sum(not item["class_ids"] for item in items), "classes": dict(classes)}


def choose_internal_validation(items):
    by_parent = defaultdict(list)
    for item in items:
        by_parent[item["capture_id"]].append(item)
    candidates = []
    for parent, parent_items in by_parent.items():
        if not any(item["source_kind"] == "verified_ground_truth" for item in parent_items):
            continue
        # Keep a ground-truth-only parent holdout; a pseudo-label cannot become its target.
        if any(item["source_kind"] == "reviewed_pseudo" for item in parent_items):
            continue
        cls_counts = Counter(CLASS_NAMES[c] for item in parent_items for c in item["class_ids"])
        if set(cls_counts) != set(CLASS_NAMES.values()):
            continue
        remaining = [item for item in items if item["capture_id"] != parent]
        remaining_classes = {CLASS_NAMES[c] for item in remaining for c in item["class_ids"]}
        if remaining_classes != set(CLASS_NAMES.values()):
            continue
        candidates.append((sum(len(item["class_ids"]) for item in parent_items), len(parent_items), parent, parent_items))
    if not candidates:
        raise ValueError("T1_T1 has no verified-only parent capture that can serve as a three-class internal validation group while leaving all classes in training.")
    # Prefer the verified-only parent with the most labeled boxes; ties remain deterministic.
    _, _, selected_parent, validation_items = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[0]
    training_items = [item for item in items if item["capture_id"] != selected_parent]
    if set(item["capture_id"] for item in training_items) & {selected_parent}:
        raise AssertionError("Parent capture leakage between train and validation")
    return selected_parent, training_items, validation_items, {parent: len(values) for parent, values in by_parent.items()}


def test_snapshot():
    files = []
    for root in (TEST_IMAGE_ROOT, TEST_LABEL_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(root)
        allowed = IMAGE_EXTS if root == TEST_IMAGE_ROOT else {".txt"}
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed):
            files.append((path, str(path.relative_to(VERIFIED)), sha256(path)))
    image_files = [p for p in TEST_IMAGE_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(image_files) != 59:
        raise ValueError(f"Expected untouched test set of 59 images, found {len(image_files)}")
    for image in image_files:
        label = TEST_LABEL_ROOT / image.relative_to(TEST_IMAGE_ROOT).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Untouched test label missing: {label}")
        if cv2.imread(str(image)) is None:
            raise ValueError(f"Unreadable test image: {image}")
        valid_yolo(label)
    return files


def copy_item(item, split: str) -> dict:
    if item["source_kind"] == "verified_ground_truth":
        rel = Path("verified") / item["capture_id"] / item["image"].name
    else:
        rel = Path("reviewed_pseudo") / item["image_path"]
    image_dst = DATASET_ROOT / "images" / split / rel
    label_dst = DATASET_ROOT / "labels" / split / rel.with_suffix(".txt")
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    if image_dst.exists() or label_dst.exists():
        if not image_dst.is_file() or not label_dst.is_file() or sha256(image_dst) != sha256(item["image"]) or sha256(label_dst) != sha256(item["label"]):
            raise FileExistsError(f"Prepared destination differs from its source: {rel}")
    else:
        shutil.copy2(item["image"], image_dst)
        shutil.copy2(item["label"], label_dst)
    return {"split": split, "image": str(image_dst.relative_to(EXPERIMENT)), "label": str(label_dst.relative_to(EXPERIMENT)), "source_image": str(item["image"].relative_to(ROOT)), "source_label": str(item["label"].relative_to(ROOT)), "source_kind": item["source_kind"], "review_status": item["status"], "capture_id": item["capture_id"], "plot_group": item["plot_group"], "boxes": len(item["class_ids"]), "class_counts": dict(Counter(CLASS_NAMES[c] for c in item["class_ids"]))}


def copy_test_snapshot(snapshot):
    target_root = EXPERIMENT / "evaluation" / "untouched_test"
    copied = []
    for source, relative, before_hash in snapshot:
        rel = Path(relative)
        destination = target_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or sha256(destination) != before_hash:
                raise ValueError(f"Existing test snapshot differs from original: {destination}")
        else:
            shutil.copy2(source, destination)
        after_hash = sha256(destination)
        if after_hash != before_hash:
            raise ValueError(f"Test snapshot copy mismatch: {source}")
        copied.append({"source": relative, "snapshot": str(destination.relative_to(EXPERIMENT)), "sha256": before_hash})
    return copied


def make_train_config(batch: int, device_info: dict, selected_parent: str):
    return {
        "experiment": str(EXPERIMENT), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "YOLOv8n", "base_checkpoint": str(ORIGINAL_MODEL), "checkpoint_sha256_before": sha256(ORIGINAL_MODEL),
        "dataset_yaml": str(DATA_YAML), "dataset_report": str(REPORT_PATH), "manifest": str(MANIFEST_PATH),
        "epochs": 100, "imgsz": 640, "batch": batch, "device": 0, "workers": 0, "patience": 20,
        "seed": 42, "deterministic": True, "amp": True, "optimizer": "auto",
        "augmentation": {"hsv_h": 0.01, "hsv_s": 0.25, "hsv_v": 0.15, "degrees": 1.0, "translate": 0.05, "scale": 0.15, "fliplr": 0.5, "flipud": 0.0, "mosaic": 0.15, "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0},
        "validation": {"type": "internal parent-photograph holdout", "capture_id": selected_parent, "independent_plot": False, "test_used_for_checkpoint_selection": False},
        "group_policy": "train from T1_T1 only; all T1_T2/T1_T3 records excluded from train and validation because those plot groups are in the untouched test set",
        "gpu_diagnostic": device_info, "training_status": "prepared", "batch_attempts": [],
    }


def write_training_log(train_run_dir: Path, batch: int, attempts: list[dict]) -> None:
    results_csv = train_run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Per-epoch training log not found: {results_csv}")
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"])) if rows else None
    summary = [
        "Fieldwise reviewed-label YOLOv8n training log",
        f"Experiment: {EXPERIMENT}",
        f"Base checkpoint: {ORIGINAL_MODEL}",
        f"Training run: {train_run_dir}",
        f"Completed epochs: {len(rows)} / 100 maximum",
        f"Batch attempts: {json.dumps(attempts, default=json_default)}",
        f"Best epoch by validation mAP50-95: {best['epoch'] if best else 'unavailable'}",
        "AMP was enabled. Ultralytics skipped its optional AMP check because its helper model could not be downloaded in the offline environment; observed training metrics remained finite.",
        f"Per-epoch losses and validation metrics are logged in: {results_csv}",
        "",
        results_csv.read_text(encoding="utf-8-sig").strip(),
        "",
    ]
    LOG_PATH.write_text("\n".join(summary), encoding="utf-8")


def write_reports(review_rows, statuses, reviewed_group_counts, eligible_t1, selected_parent, train_items, val_items, all_verified_t1, test_copy_records, manifest_items):
    train_count, val_count, test_count = map(make_item_counts, (train_items, val_items, []))
    test_class_counts = Counter()
    for copied in test_copy_records:
        if Path(copied["source"]).is_relative_to(TEST_LABEL_ROOT.relative_to(VERIFIED)):
            label_path = EXPERIMENT / copied["snapshot"]
            test_class_counts.update(CLASS_NAMES[c] for c in valid_yolo(label_path))
    test_image_count = sum(1 for row in test_copy_records if Path(row["source"]).is_relative_to(TEST_IMAGE_ROOT.relative_to(VERIFIED)) and Path(row["source"]).suffix.lower() in IMAGE_EXTS)
    test_empty_count = sum(not valid_yolo(EXPERIMENT / copied["snapshot"]) for copied in test_copy_records if Path(copied["source"]).is_relative_to(TEST_LABEL_ROOT.relative_to(VERIFIED)))
    test_count = {"images": test_image_count, "boxes": sum(test_class_counts.values()), "empty_images": test_empty_count, "classes": dict(test_class_counts)}
    source_counts = {
        "verified_t1_t1_before_holdout": make_item_counts(all_verified_t1),
        "reviewed_pseudo_t1_t1": make_item_counts(eligible_t1),
        "reviewed_pseudo_all_groups": {"images": len([row for row in review_rows if row["review_status"].strip().lower() in ALLOWED_STATUS]), "boxes": sum(counts["boxes"] for counts in reviewed_group_counts.values())},
        "reviewed_pseudo_excluded_test_plot_groups": {"images": sum(value["images"] for group, value in reviewed_group_counts.items() if group in {"T1_T2", "T1_T3"}), "groups": ["T1_T2", "T1_T3"]},
    }
    dataset_manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": "T1_T1 only for train/internal validation; test is an exact isolated copy of the original test split; pending/rejected excluded", "review_status_counts": dict(statuses), "selected_reviewed_statuses": {key: statuses[key] for key in sorted(ALLOWED_STATUS)}, "plot_group_reviewed_counts": {group: {"images": value["images"], "boxes": value["boxes"], "classes": dict(value["classes"]), "statuses": dict(value["statuses"])} for group, value in reviewed_group_counts.items()}, "selected_validation_capture": selected_parent, "selected_validation_labels": "verified ground truth only", "splits": {"train": train_count, "val": val_count, "test_snapshot": test_count}, "source_counts": source_counts, "test_snapshot_hashes": test_copy_records, "items": manifest_items}
    atomic_json(MANIFEST_PATH, dataset_manifest)
    report = [
        "# Reviewed-label exploratory dataset composition", "",
        "This experiment uses manually reviewed pseudo-labels, which remain lower-confidence supervision than independently collected ground truth. No pending or rejected labels are used.", "",
        "## Manifest audit", "", f"- Review rows: {len(review_rows)}", f"- Actual review statuses: `{dict(statuses)}`", f"- Trainable reviewed items with valid image/label: {sum(statuses[k] for k in ALLOWED_STATUS)}", f"- Reviewed pseudo labels selected from T1_T1: {len(eligible_t1)}", f"- Reviewed pseudo labels excluded from T1_T2/T1_T3: {source_counts['reviewed_pseudo_excluded_test_plot_groups']['images']}", "- Pending/rejected excluded from training.", "",
        "## Parent-group split", "", f"- Plot group for training and internal validation: `T1_T1` only.", f"- Internal validation parent capture: `{selected_parent}` (one parent image group; verified labels only).", f"- T1_T1 parent captures represented in selected tiles: {len(set(item['capture_id'] for item in all_verified_t1 + eligible_t1))}.", "- Internal validation is parent-separated but **not plot-independent**. Metrics are exploratory and cannot establish generalization to unseen plot groups.", "- All T1_T2/T1_T3 reviewed images are excluded from train and validation because those plots are represented in the untouched test set.", "",
        "## Split counts", "", "| Split | Images | Boxes | Empty-label images | Vegetative | Tuber Bulking | Maturity |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, count in (("Train", train_count), ("Internal validation (non-independent)", val_count), ("Untouched test snapshot (evaluation only)", test_count)):
        report.append(f"| {name} | {count['images']} | {count['boxes']} | {count['empty_images']} | {count['classes'].get('Vegetative', 0)} | {count['classes'].get('Tuber Bulking', 0)} | {count['classes'].get('Maturity', 0)} |")
    report += ["", "## Source composition", "", "| Split | Verified ground-truth images | Reviewed pseudo-label images |", "|---|---:|---:|"]
    report.append(f"| Train | {sum(i['source_kind']=='verified_ground_truth' for i in train_items)} | {sum(i['source_kind']=='reviewed_pseudo' for i in train_items)} |")
    report.append(f"| Internal validation | {sum(i['source_kind']=='verified_ground_truth' for i in val_items)} | {sum(i['source_kind']=='reviewed_pseudo' for i in val_items)} |")
    report += ["", "## Class balance and limitations", ""]
    total_boxes = train_count["boxes"]
    minority_share = train_count["classes"].get("Vegetative", 0) / total_boxes if total_boxes else 0
    report += [f"Training box distribution is {train_count['classes']}; Vegetative is the underrepresented class at {minority_share:.1%} of training boxes. No oversampling or class weighting was added. Augmentations are kept mild to reduce distortion of growth-stage appearance.", f"Validation has {val_count['images']} tiles and {val_count['boxes']} boxes from a single parent capture, including {val_count['classes'].get('Maturity', 0)} Maturity boxes. Per-class validation metrics will be noisy.", "The experiment has one source plot in train/validation and two existing test plots. It can provide an exploratory comparison on the fixed test set, but three plot IDs are insufficient to estimate broad field generalization.", "", "Test source image and label hashes are recorded in the manifest; the original test files were only read and copied to the experiment evaluation folder.", ""]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    split_report = ["# Parent-photograph split report", "", "| Split | Plot group | Parent capture IDs | Verified images | Reviewed pseudo images | Total images |", "|---|---|---|---:|---:|---:|"]
    train_caps = sorted({item["capture_id"] for item in train_items})
    val_caps = sorted({item["capture_id"] for item in val_items})
    split_report.append(f"| Train | T1_T1 | {', '.join(train_caps)} | {sum(i['source_kind']=='verified_ground_truth' for i in train_items)} | {sum(i['source_kind']=='reviewed_pseudo' for i in train_items)} | {len(train_items)} |")
    split_report.append(f"| Internal validation | T1_T1 | {', '.join(val_caps)} | {sum(i['source_kind']=='verified_ground_truth' for i in val_items)} | 0 | {len(val_items)} |")
    split_report.append(f"| Test (evaluation only) | T1_T2, T1_T3 | existing test capture groups | 59 | 0 | 59 |")
    split_report += ["", f"Train and internal validation capture overlap: `{sorted(set(train_caps) & set(val_caps))}` (must be empty).", "The internal validation is not independent at plot level. T1_T2 and T1_T3 appear only in the untouched test snapshot and are never used for training, validation, checkpoint selection, or hyperparameter tuning.", ""]
    SPLIT_REPORT_PATH.write_text("\n".join(split_report), encoding="utf-8")
    return dataset_manifest


def evaluate(model_path: Path, name: str):
    model = YOLO(str(model_path))
    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=640, batch=2, device=0, workers=0, plots=True, save=True, project=str(EXPERIMENT / "evaluation" / "models"), name=name, exist_ok=False, cache=False, verbose=True)
    box = metrics.box
    per_class = {}
    for class_id, class_name in CLASS_NAMES.items():
        p, r, map50, map5095 = box.class_result(class_id)
        per_class[class_name] = {"precision": float(p), "recall": float(r), "map50": float(map50), "map50_95": float(map5095)}
    return {"checkpoint": str(model_path), "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map), "per_class": per_class, "evaluation_directory": str(EXPERIMENT / "evaluation" / "models" / name), "results_dict": {str(k): float(v) for k, v in metrics.results_dict.items()}}


def main() -> None:
    if EXPERIMENT.exists():
        unexpected = []
        for existing in EXPERIMENT.rglob("*"):
            if existing.is_file() and not (existing.is_relative_to(DATASET_ROOT / "images") or existing.is_relative_to(DATASET_ROOT / "labels")):
                unexpected.append(str(existing))
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite existing experiment artifacts: {unexpected[:10]}")
    for required in (REVIEW_CSV, APPROVED, VERIFIED, TILES, RGB_ROOT, ORIGINAL_MODEL, TEST_IMAGE_ROOT, TEST_LABEL_ROOT):
        if not required.exists():
            raise FileNotFoundError(required)
    if ORIGINAL_MODEL.resolve() == (EXPERIMENT / "checkpoints" / "best.pt").resolve():
        raise AssertionError("Experiment checkpoint path aliases the original model")

    # Recheck the previous cudaErrorUnknown with a synchronized operation; never classify it as OOM.
    gpu_diagnostic = {"cuda_available": bool(torch.cuda.is_available()), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "device_count": torch.cuda.device_count()}
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(f"CUDA GPU 0 is not available; this requested run does not silently fall back to CPU: {gpu_diagnostic}")
    gpu_diagnostic["device_name"] = torch.cuda.get_device_name(0)
    if "3050" not in gpu_diagnostic["device_name"]:
        raise RuntimeError(f"Expected RTX 3050 GPU, found {gpu_diagnostic['device_name']}")
    torch.cuda.init()
    probe = torch.arange(250_000, device="cuda", dtype=torch.float32)
    probe_result = (probe.square().mean()).item()
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gpu_diagnostic.update({"synchronized_cuda_probe": float(probe_result), "free_memory_gib": free_bytes / 1024**3, "total_memory_gib": total_bytes / 1024**3, "diagnostic_status": "passed"})
    if "cudaErrorUnknown" in str(gpu_diagnostic):
        raise RuntimeError(f"CUDA unknown error during diagnostic: {gpu_diagnostic}")

    capture_to_group, capture_paths = discover_plot_groups()
    plot_groups = set(capture_to_group.values())
    if plot_groups != {"T1_T1", "T1_T2", "T1_T3"}:
        raise ValueError(f"Unexpected source plot groups: {plot_groups}")
    test_before = test_snapshot()
    test_groups = {capture_to_group.get(capture_id(Path(row[1]).name) or "") for row in test_before if Path(row[1]).parts[0] == "images"}
    if test_groups != {"T1_T2", "T1_T3"}:
        raise ValueError(f"Expected original test plot groups T1_T2/T1_T3, found {test_groups}")
    original_checkpoint_hash = sha256(ORIGINAL_MODEL)
    verified_hashes = {}
    for split in ("train", "val", "test"):
        for image in sorted(p for p in (VERIFIED / "images" / split).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            digest = sha256(image)
            if digest in verified_hashes:
                raise ValueError(f"Duplicate verified image bytes: {image} and {verified_hashes[digest]}")
            verified_hashes[digest] = str(image)
    review_rows, statuses, reviewed_group_counts, reviewed = collect_reviewed(capture_to_group, verified_hashes)
    verified_t1 = collect_verified_train(capture_to_group, verified_hashes)
    if any(item["plot_group"] in {"T1_T2", "T1_T3"} for item in verified_t1):
        raise AssertionError("Test plot group entered verified training pool")
    reviewed_t1 = [item for item in reviewed if item["plot_group"] == "T1_T1"]
    reviewed_excluded = [item for item in reviewed if item["plot_group"] in {"T1_T2", "T1_T3"}]
    unknown = [item for item in reviewed if item["plot_group"] not in {"T1_T1", "T1_T2", "T1_T3"}]
    if unknown:
        raise ValueError(f"Reviewed images have unknown groups: {[str(x['image_path']) for x in unknown[:20]]}")
    if len(reviewed_t1) != 287 or len(reviewed_excluded) != 745:
        raise ValueError(f"Expected 287 T1_T1 and 745 excluded reviewed images, got {len(reviewed_t1)} and {len(reviewed_excluded)}")
    candidate_pool = verified_t1 + reviewed_t1
    selected_parent, train_items, val_items, parent_image_counts = choose_internal_validation(candidate_pool)
    if selected_parent != "IMG20231228105347":
        raise ValueError(f"Internal validation parent changed from the audited candidate: {selected_parent}")
    if any(item["source_kind"] != "verified_ground_truth" for item in val_items):
        raise AssertionError("Internal validation must use only verified labels")
    train_groups = {item["plot_group"] for item in train_items}
    val_groups = {item["plot_group"] for item in val_items}
    if train_groups != {"T1_T1"} or val_groups != {"T1_T1"}:
        raise AssertionError("Train/validation must be restricted to T1_T1")
    if {item["capture_id"] for item in train_items} & {item["capture_id"] for item in val_items}:
        raise AssertionError("Parent image leakage between train and validation")
    if set(CLASS_NAMES.values()) - {CLASS_NAMES[c] for item in train_items for c in item["class_ids"]}:
        raise ValueError("Training pool is missing at least one class")
    if set(CLASS_NAMES.values()) - {CLASS_NAMES[c] for item in val_items for c in item["class_ids"]}:
        raise ValueError("Internal validation is missing at least one class")
    if (EXPERIMENT / "checkpoints").exists():
        raise FileExistsError("Experiment checkpoint directory unexpectedly exists")

    print("PRE-FLIGHT PASSED")
    print("review_statuses", dict(statuses))
    print("verified_T1_T1", make_item_counts(verified_t1))
    print("reviewed_T1_T1", make_item_counts(reviewed_t1))
    print("reviewed_excluded_test_groups", len(reviewed_excluded))
    print("internal_validation_parent", selected_parent, make_item_counts(val_items))
    print("train", make_item_counts(train_items))
    print("represented_parent_captures", len(parent_image_counts), "all_T1_T1_source_parents", sum(1 for x in capture_to_group.values() if x == "T1_T1"))
    print("gpu_diagnostic", gpu_diagnostic)

    # Only now create a new isolated experiment directory.
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    train_manifest = [copy_item(item, "train") for item in train_items]
    val_manifest = [copy_item(item, "val") for item in val_items]
    test_copy_records = copy_test_snapshot(test_before)
    train_caps = {item["capture_id"] for item in train_items}
    val_caps = {item["capture_id"] for item in val_items}
    if train_caps & val_caps:
        raise AssertionError("Prepared train and validation parent IDs overlap")
    if any(capture_to_group.get(cid) in {"T1_T2", "T1_T3"} for cid in train_caps | val_caps):
        raise AssertionError("Test plot group entered prepared training or validation")
    if len([p for p in (EXPERIMENT / "evaluation" / "untouched_test" / "images" / "test").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) != 59:
        raise ValueError("Test snapshot image count mismatch")

    yaml_data = {"path": str(EXPERIMENT.resolve()), "train": "dataset/images/train", "val": "dataset/images/val", "test": "evaluation/untouched_test/images/test", "names": CLASS_NAMES}
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest_items = train_manifest + val_manifest
    dataset_manifest = write_reports(review_rows, statuses, reviewed_group_counts, reviewed_t1, selected_parent, train_items, val_items, verified_t1, test_copy_records, manifest_items)
    if sha256(ORIGINAL_MODEL) != original_checkpoint_hash:
        raise AssertionError("Original checkpoint hash changed during preparation")

    config = make_train_config(2, gpu_diagnostic, selected_parent)
    config["review_status_counts"] = dict(statuses)
    config["selected_training"] = make_item_counts(train_items)
    config["selected_validation"] = make_item_counts(val_items)
    config["excluded_reviewed_images_other_plot_groups"] = len(reviewed_excluded)
    config["test_snapshot_count"] = len(test_copy_records)
    atomic_json(CONFIG_PATH, config)

    print("Starting YOLOv8n training from the original checkpoint. Test set remains unused until training finishes.")
    attempted_batches = []
    train_result = None
    train_run_dir = None
    for batch in (2, 1):
        attempt = {"batch": batch, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        attempted_batches.append(attempt)
        config["training_status"] = "running"
        config["batch_attempts"] = attempted_batches
        atomic_json(CONFIG_PATH, config)
        run_name = f"train_batch{batch}"
        try:
            trainer_model = YOLO(str(ORIGINAL_MODEL))
            with LOG_PATH.open("a", encoding="utf-8") as log_file:
                with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
                    train_result = trainer_model.train(
                        data=str(DATA_YAML), epochs=100, imgsz=640, batch=batch, workers=0, device=0,
                        patience=20, seed=42, deterministic=True, amp=True, optimizer="auto",
                        project=str(EXPERIMENT / "training"), name=run_name, exist_ok=False,
                        cache=False, plots=True, save=True, verbose=True,
                        hsv_h=0.01, hsv_s=0.25, hsv_v=0.15, degrees=1.0, translate=0.05,
                        scale=0.15, fliplr=0.5, flipud=0.0, mosaic=0.15, close_mosaic=10,
                        mixup=0.0, copy_paste=0.0,
                    )
            train_run_dir = Path(trainer_model.trainer.save_dir)
            attempt["status"] = "completed"
            attempt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            config["batch"] = batch
            config["training_run_dir"] = str(train_run_dir)
            break
        except RuntimeError as exc:
            error_text = str(exc)
            attempt["error"] = error_text
            attempt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in error_text.lower() or "out of memory" in error_text.lower()
            if "cudaerrorunknown" in error_text.lower() or "cuda error: unknown error" in error_text.lower():
                attempt["status"] = "cuda_unknown_error"
                config["training_status"] = "failed_cuda_unknown"
                config["batch_attempts"] = attempted_batches
                config["cuda_error_diagnostic_after_failure"] = {"cuda_available": bool(torch.cuda.is_available()), "initialized": bool(torch.cuda.is_initialized())}
                atomic_json(CONFIG_PATH, config)
                raise RuntimeError("CUDA reported cudaErrorUnknown; it was not treated as OOM and no CPU fallback was attempted.") from exc
            if not is_oom or batch == 1:
                attempt["status"] = "failed"
                config["training_status"] = "failed"
                config["batch_attempts"] = attempted_batches
                atomic_json(CONFIG_PATH, config)
                raise
            attempt["status"] = "cuda_oom_retrying_batch_1"
            config["batch_attempts"] = attempted_batches
            atomic_json(CONFIG_PATH, config)
            print("Confirmed CUDA out-of-memory at batch 2; retrying with batch 1.")
            del trainer_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    if train_result is None or train_run_dir is None:
        raise RuntimeError("Training did not complete")

    best_src = train_run_dir / "weights" / "best.pt"
    last_src = train_run_dir / "weights" / "last.pt"
    if not best_src.is_file() or not last_src.is_file():
        raise FileNotFoundError(f"Training did not produce both checkpoints: {best_src}, {last_src}")
    checkpoint_dir = EXPERIMENT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    shutil.copy2(best_src, best_checkpoint)
    shutil.copy2(last_src, last_checkpoint)
    config["training_status"] = "completed"
    config["best_checkpoint"] = str(best_checkpoint)
    config["last_checkpoint"] = str(last_checkpoint)
    config["training_run_dir"] = str(train_run_dir)
    config["batch_attempts"] = attempted_batches
    config["amp"] = True
    config["amp_check"] = "Ultralytics optional check skipped because its helper model could not be downloaded offline; AMP remained enabled and recorded losses were finite."
    config["checkpoint_hashes"] = {"best": sha256(best_checkpoint), "last": sha256(last_checkpoint), "original_after_training": sha256(ORIGINAL_MODEL)}
    config["original_checkpoint_unchanged"] = original_checkpoint_hash == sha256(ORIGINAL_MODEL)
    atomic_json(CONFIG_PATH, config)
    write_training_log(train_run_dir, config["batch"], attempted_batches)
    if not config["original_checkpoint_unchanged"]:
        raise AssertionError("Original checkpoint hash changed")

    print("Training completed. Starting final-only test evaluation of original and improved models.")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            baseline_metrics = evaluate(ORIGINAL_MODEL, "original_baseline")
            improved_metrics = evaluate(best_checkpoint, "reviewed_t1_t1_best")
    test_after = test_snapshot()
    if [(row[1], row[2]) for row in test_before] != [(row[1], row[2]) for row in test_after]:
        raise AssertionError("Original untouched test files changed during experiment")
    comparison = {
        "test_set": {"images": 59, "split": str(TEST_IMAGE_ROOT), "plot_groups": ["T1_T2", "T1_T3"], "original_files_unchanged": True, "test_used_for_training_or_checkpoint_selection": False},
        "provided_original_baseline_reference": {"precision": 0.25896, "recall": 0.71583, "map50": 0.37561, "map50_95": 0.23653},
        "original_re_evaluated_on_test": baseline_metrics,
        "improved_model_on_same_test": improved_metrics,
        "deltas_vs_re_evaluated_original": {key: improved_metrics[key] - baseline_metrics[key] for key in ("precision", "recall", "map50", "map50_95")},
        "interpretation": "Exploratory single-plot training using a mixture of verified ground truth and manually reviewed pseudo-labels. Pseudo-labels are not equivalent to independent ground truth; the internal validation is non-independent at plot level. Accuracy improvement is not claimed unless test metrics support it.",
    }
    atomic_json(EXPERIMENT / "evaluation" / "comparison_report.json", comparison)
    table = ["# Original vs reviewed-label model comparison", "", "Both checkpoints were evaluated after training on the same copied, hash-verified untouched test set. Test results were not used for checkpoint selection or tuning.", "", "| Metric | Original checkpoint re-evaluated | Improved checkpoint | Difference | User-provided baseline reference |", "|---|---:|---:|---:|---:|"]
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP50"), ("map50_95", "mAP50–95")):
        table.append(f"| {label} | {baseline_metrics[key]:.5f} | {improved_metrics[key]:.5f} | {improved_metrics[key]-baseline_metrics[key]:+.5f} | {comparison['provided_original_baseline_reference'][key]:.5f} |")
    table += ["", "## Per-class metrics", "", "| Class | Model | Precision | Recall | mAP50 | mAP50–95 |", "|---|---|---:|---:|---:|---:|"]
    for cls in CLASS_NAMES.values():
        for model_name, metrics in (("Original", baseline_metrics), ("Improved", improved_metrics)):
            values = metrics["per_class"][cls]
            table.append(f"| {cls} | {model_name} | {values['precision']:.5f} | {values['recall']:.5f} | {values['map50']:.5f} | {values['map50_95']:.5f} |")
    table += ["", "## Artifacts", "", f"- Best checkpoint: `{best_checkpoint}`", f"- Last checkpoint: `{last_checkpoint}`", f"- Training plots and results: `{train_run_dir}`", f"- Evaluation plots, confusion matrices, and prediction visualizations: `{EXPERIMENT / 'evaluation' / 'models'}`", "", "## Interpretation", "", comparison["interpretation"], ""]
    (EXPERIMENT / "evaluation" / "comparison_report.md").write_text("\n".join(table), encoding="utf-8")
    config["evaluation_status"] = "completed"
    config["comparison_report"] = str(EXPERIMENT / "evaluation" / "comparison_report.md")
    config["test_original_files_unchanged"] = True
    atomic_json(CONFIG_PATH, config)
    print("EXPERIMENT COMPLETE")
    print(json.dumps({"train": make_item_counts(train_items), "validation": make_item_counts(val_items), "test": {"images": 59, "boxes": 85}, "best_checkpoint": str(best_checkpoint), "last_checkpoint": str(last_checkpoint), "baseline": baseline_metrics, "improved": improved_metrics, "deltas": comparison["deltas_vs_re_evaluated_original"]}, indent=2, default=json_default))


if __name__ == "__main__":
    # The run directory is created only after every preflight check passes.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="runs/fieldwise_yolov8n_1032", help="New output directory below runs/; existing experiments are never overwritten")
    configure_experiment(parser.parse_args().experiment)
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
