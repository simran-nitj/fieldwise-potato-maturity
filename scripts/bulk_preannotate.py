"""Bulk YOLO pre-annotation into a separate, resumable review workspace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAPTURE_RE = re.compile(r"IMG\d{14}", re.IGNORECASE)
TILE_RE = re.compile(r"_tile_(\d+)", re.IGNORECASE)
SCHEMA_VERSION = 1


def image_key(value: str | Path) -> tuple[str, int | str]:
    text = str(value).replace("\\", "/")
    cap = CAPTURE_RE.search(Path(text).name)
    tile = TILE_RE.search(Path(text).stem)
    if cap and tile:
        return cap.group(0).upper(), int(tile.group(1))
    return Path(text).stem.lower(), ""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value):
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def checkpoint_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_yolo_label(path: Path):
    """Return whether the file contains only valid, non-empty YOLO rows."""
    try:
        lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return False
    if not lines:
        return False
    for parts in lines:
        if len(parts) != 5:
            return False
        try:
            class_value = float(parts[0])
            box = [float(value) for value in parts[1:]]
        except ValueError:
            return False
        if not class_value.is_integer() or int(class_value) not in CLASS_NAMES:
            return False
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in box):
            return False
        cx, cy, width, height = box
        if width <= 0 or height <= 0 or cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            return False
    return True


def valid_csv_annotations(root: Path, source: Path):
    """Identify source tiles that have complete, valid existing CSV boxes."""
    csv_path = root / "annotations.csv"
    if not csv_path.is_file():
        return set()
    grouped, invalid_keys = {}, set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").replace("\\", "/")
            key = image_key(filename)
            source_path = source / Path(filename)
            try:
                class_id = int(row.get("class_id", ""))
                coords = [float(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")]
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("unreadable source image")
                height, width = image.shape[:2]
                xmin, ymin, xmax, ymax = coords
                if class_id not in CLASS_NAMES or not all(math.isfinite(x) for x in coords) or xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmin >= xmax or ymin >= ymax:
                    raise ValueError("invalid class or pixel box")
            except (ValueError, KeyError, TypeError):
                invalid_keys.add(key)
                continue
            grouped.setdefault(key, 0)
            grouped[key] += 1
    return {key for key in grouped if key not in invalid_keys and grouped[key] > 0}


def existing_valid_keys(root: Path, source: Path):
    keys = valid_csv_annotations(root, source)
    yolo_root = root / "potato_yolo_dataset"
    for split in ("train", "val", "test"):
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        labels_by_stem = {path.stem.lower(): path for path in label_dir.rglob("*.txt")}
        for image_path in image_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels_by_stem.get(image_path.stem.lower())
            if label and valid_yolo_label(label):
                keys.add(image_key(image_path.stem))
    return keys


def to_detection_records(result):
    detections = []
    if result.boxes is None:
        return detections
    height, width = result.orig_shape
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(box.conf.item()),
            "x_center": ((x1 + x2) / 2) / width,
            "y_center": ((y1 + y2) / 2) / height,
            "width": (x2 - x1) / width,
            "height": (y2 - y1) / height,
        })
    return detections


class BulkPreannotator:
    def __init__(self, root: Path, source: Path, output: Path, checkpoint: Path, confidence: float, batch_size: int, retries: int, device: str):
        self.root = root.resolve()
        self.source = source.resolve()
        self.output = output.resolve()
        self.checkpoint = checkpoint.resolve()
        self.confidence = confidence
        self.batch_size = batch_size
        self.retries = retries
        self.requested_device = device
        self.torch = None
        self.model = None
        self.device = "cpu"
        self.started_at = utc_now()
        for folder in ("labels", "metadata", "zero_detections", "reports"):
            (self.output / folder).mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output / "progress.json"

    def discover(self):
        if not self.source.is_dir():
            raise FileNotFoundError(f"Source image directory not found: {self.source}")
        paths = sorted(path for path in self.source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        existing = existing_valid_keys(self.root, self.source)
        candidates = [path for path in paths if image_key(path.relative_to(self.source)) not in existing]
        return paths, existing, candidates

    def load_model(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint}")
        import torch
        from ultralytics import YOLO

        self.torch = torch
        cuda_ok = torch.cuda.is_available()
        if self.requested_device == "cpu" or not cuda_ok:
            self.device = "cpu"
        else:
            self.device = "0"
        self.model = YOLO(str(self.checkpoint))
        names = {int(key): str(value).strip() for key, value in self.model.names.items()}
        if names != CLASS_NAMES:
            raise RuntimeError(f"Checkpoint classes do not match project classes: {names}")
        return names

    def _signature(self, candidates):
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source.relative_to(self.root).as_posix() if self.source.is_relative_to(self.root) else self.source.name,
            "checkpoint_sha256": checkpoint_sha256(self.checkpoint),
            "confidence_threshold": self.confidence,
            "image_size": 640,
            "eligible_images": [path.relative_to(self.source).as_posix() for path in candidates],
        }

    def load_progress(self, candidates):
        signature = self._signature(candidates)
        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if progress.get("signature") != signature:
                raise RuntimeError("Existing progress uses a different source, checkpoint, threshold, or eligible-image list. Use a separate output directory to avoid mixing runs.")
            records = progress.get("records", {})
        else:
            records = {}
        for path in candidates:
            relative = path.relative_to(self.source).as_posix()
            records.setdefault(relative, {"status": "pending", "attempts": 0, "last_error": None})
        progress = {"signature": signature, "started_at": progress.get("started_at", utc_now()) if self.progress_path.exists() else utc_now(), "updated_at": utc_now(), "records": records}
        self.save_progress(progress)
        return progress

    def save_progress(self, progress):
        progress["updated_at"] = utc_now()
        atomic_write_json(self.progress_path, progress)

    def _predict_batch(self, paths):
        results = self.model.predict(
            source=[str(path) for path in paths],
            imgsz=640,
            batch=len(paths),
            conf=self.confidence,
            device=self.device,
            verbose=False,
            stream=False,
        )
        if len(results) != len(paths):
            raise RuntimeError(f"Model returned {len(results)} results for {len(paths)} images")
        return [to_detection_records(result) for result in results]

    def _predict_single(self, path, device):
        result = self.model.predict(source=str(path), imgsz=640, batch=1, conf=self.confidence, device=device, verbose=False)[0]
        return to_detection_records(result)

    def _relative_output(self, folder, source_path):
        return self.output / folder / source_path.relative_to(self.source)

    def _write_result(self, source_path, detections, status, attempts, error=None, device_used=None):
        relative = source_path.relative_to(self.source).as_posix()
        labels_path = self._relative_output("labels", source_path).with_suffix(".txt")
        metadata_path = self._relative_output("metadata", source_path).with_suffix(".json")
        if status == "zero_detections":
            atomic_write_text(labels_path, "")
            zero_path = self._relative_output("zero_detections", source_path)
            zero_path.parent.mkdir(parents=True, exist_ok=True)
            if not zero_path.exists():
                shutil.copy2(source_path, zero_path)
        elif status == "success":
            lines = [f"{d['class_id']} {d['x_center']:.6f} {d['y_center']:.6f} {d['width']:.6f} {d['height']:.6f}" for d in detections]
            atomic_write_text(labels_path, "\n".join(lines) + "\n")
            zero_path = None
        else:
            zero_path = None
        metadata = {
            "original_path": (self.source.relative_to(self.root) / source_path.relative_to(self.source)).as_posix() if self.source.is_relative_to(self.root) else source_path.relative_to(self.source).as_posix(),
            "status": status,
            "processed_at_utc": utc_now(),
            "confidence_threshold": self.confidence,
            "model_checkpoint": self.checkpoint.name,
            "device_used": device_used or ("CUDA" if self.device == "0" else "CPU"),
            "attempts": attempts,
            "detections": detections,
            "last_error": error,
            "label_file": labels_path.relative_to(self.output).as_posix() if status in {"success", "zero_detections"} else None,
            "zero_detection_image": zero_path.relative_to(self.output).as_posix() if zero_path else None,
        }
        atomic_write_json(metadata_path, metadata)
        return metadata

    def _count_classes(self, records):
        counts = {name: 0 for name in CLASS_NAMES.values()}
        for record in records.values():
            for name, value in record.get("class_counts", {}).items():
                if name in counts:
                    counts[name] += int(value)
        return counts

    def _write_summary(self, progress, total_tiles, excluded_count, candidates, write_markdown=False):
        records = progress["records"]
        statuses = [records[path.relative_to(self.source).as_posix()].get("status", "pending") for path in candidates]
        summary = {
            "updated_at_utc": utc_now(),
            "model_checkpoint": self.checkpoint.name,
            "confidence_threshold": self.confidence,
            "device_requested": self.requested_device,
            "device_selected": "CUDA" if self.device == "0" else "CPU",
            "batch_size": self.batch_size,
            "total_tiles_scanned": total_tiles,
            "valid_annotated_images_excluded": excluded_count,
            "eligible_images": len(candidates),
            "processed_successfully": statuses.count("success"),
            "zero_detections": statuses.count("zero_detections"),
            "failed": statuses.count("failed"),
            "pending": statuses.count("pending"),
            "total_detection_instances": sum(self._count_classes(records).values()),
            "class_instances": self._count_classes(records),
            "processing_records": len(records),
        }
        atomic_write_json(self.output / "reports" / "summary.json", summary)
        failed = [(path.relative_to(self.source).as_posix(), records[path.relative_to(self.source).as_posix()]) for path in candidates if records[path.relative_to(self.source).as_posix()].get("status") == "failed"]
        with (self.output / "reports" / "failed_images.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_path", "attempts", "last_error"])
            writer.writerows((path, record.get("attempts", 0), record.get("last_error", "")) for path, record in failed)
        zero = [(path.relative_to(self.source).as_posix(), records[path.relative_to(self.source).as_posix()]) for path in candidates if records[path.relative_to(self.source).as_posix()].get("status") == "zero_detections"]
        with (self.output / "reports" / "zero_detections.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_path", "metadata_path"])
            writer.writerows((path, (Path("metadata") / Path(path)).with_suffix(".json").as_posix()) for path, _ in zero)
        if write_markdown:
            lines = ["# Bulk pre-annotation summary", "", f"- Total tiles scanned: {summary['total_tiles_scanned']}", f"- Valid existing annotations excluded: {summary['valid_annotated_images_excluded']}", f"- Eligible images: {summary['eligible_images']}", f"- Processed successfully: {summary['processed_successfully']}", f"- Zero detections: {summary['zero_detections']}", f"- Failed after retries: {summary['failed']}", f"- Pending: {summary['pending']}", f"- Device selected: {summary['device_selected']}", f"- Confidence threshold: {summary['confidence_threshold']}", "", "## Predicted class instances", ""]
            lines.extend(f"- {name}: {count}" for name, count in summary["class_instances"].items())
            lines += ["", "Predicted boxes are unverified pre-annotations, not ground-truth labels. Do not train on them until manually reviewed.", ""]
            (self.output / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
            stale_temp = self.output / "reports" / "summary.md.tmp"
            stale_temp.unlink(missing_ok=True)
        return summary

    def run(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        if self.batch_size < 1 or self.retries < 1:
            raise ValueError("Batch size and retries must be at least 1")
        all_paths, existing_keys, candidates = self.discover()
        if not candidates:
            raise RuntimeError("No eligible images found after excluding valid existing annotations")
        eligible_keys = {image_key(path.relative_to(self.source)) for path in candidates}
        collision_keys = eligible_keys & existing_keys
        if collision_keys:
            raise RuntimeError(f"Internal eligibility error: {len(collision_keys)} already-labeled images were not excluded")
        progress = self.load_progress(candidates)
        names = self.load_model()
        print(f"Checkpoint class names verified: {names}")
        print(f"Scanned {len(all_paths)} tiles; excluded {len(existing_keys)} with valid annotations; eligible: {len(candidates)}")
        print(f"Device: {'CUDA:0' if self.device == '0' else 'CPU'}; requested batch: {self.batch_size}; confidence: {self.confidence:.2f}")

        to_process = []
        for path in candidates:
            record = progress["records"][path.relative_to(self.source).as_posix()]
            status = record.get("status", "pending")
            if status in {"pending", "failed"}:
                to_process.append(path)
        total_batches = math.ceil(len(to_process) / self.batch_size)
        for batch_index, start in enumerate(range(0, len(to_process), self.batch_size), 1):
            chunk = to_process[start:start + self.batch_size]
            batch_error = None
            try:
                predictions = self._predict_batch(chunk)
                for path, detections in zip(chunk, predictions):
                    relative = path.relative_to(self.source).as_posix()
                    attempts = progress["records"][relative].get("attempts", 0) + 1
                    status = "success" if detections else "zero_detections"
                    metadata = self._write_result(path, detections, status, attempts)
                    progress["records"][relative] = {"status": status, "attempts": attempts, "detection_count": len(detections), "class_counts": dict(Counter(d["class_name"] for d in detections)), "metadata_path": self._relative_output("metadata", path).with_suffix(".json").relative_to(self.output).as_posix(), "last_error": None, "processed_at_utc": metadata["processed_at_utc"]}
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"
                if self.device == "0" and self.torch is not None:
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass
                for path in chunk:
                    relative = path.relative_to(self.source).as_posix()
                    attempts = progress["records"][relative].get("attempts", 0) + 1
                    succeeded = False
                    last_error = batch_error
                    for retry in range(1, self.retries + 1):
                        attempts += 1
                        retry_device = "cpu" if self.device == "0" else self.device
                        try:
                            detections = self._predict_single(path, retry_device)
                            status = "success" if detections else "zero_detections"
                            metadata = self._write_result(path, detections, status, attempts, device_used="CPU" if retry_device == "cpu" else "CUDA")
                            progress["records"][relative] = {"status": status, "attempts": attempts, "detection_count": len(detections), "class_counts": dict(Counter(d["class_name"] for d in detections)), "metadata_path": self._relative_output("metadata", path).with_suffix(".json").relative_to(self.output).as_posix(), "last_error": None, "processed_at_utc": metadata["processed_at_utc"], "batch_fallback_error": batch_error}
                            succeeded = True
                            break
                        except Exception as retry_exc:
                            last_error = f"{type(retry_exc).__name__}: {retry_exc}"
                            if self.device == "0" and self.torch is not None:
                                try:
                                    self.torch.cuda.empty_cache()
                                except Exception:
                                    pass
                    if not succeeded:
                        metadata = self._write_result(path, [], "failed", attempts, last_error)
                        progress["records"][relative] = {"status": "failed", "attempts": attempts, "detection_count": 0, "class_counts": {}, "metadata_path": self._relative_output("metadata", path).with_suffix(".json").relative_to(self.output).as_posix(), "last_error": last_error, "processed_at_utc": metadata["processed_at_utc"]}
                        print(f"FAILED after {attempts} attempts: {relative} :: {last_error}")
            self.save_progress(progress)
            summary = self._write_summary(progress, len(all_paths), len(existing_keys), candidates)
            done = sum(progress["records"][p.relative_to(self.source).as_posix()].get("status") in {"success", "zero_detections", "failed"} for p in candidates)
            print(f"Batch {batch_index}/{total_batches} saved; {done}/{len(candidates)} have terminal records; success={summary['processed_successfully']} zero={summary['zero_detections']} failed={summary['failed']}", flush=True)

        # Retry any records still marked failed after the first pass. A final retry
        # pass is separate so partial batches cannot silently leave failure records.
        failed_paths = [p for p in candidates if progress["records"][p.relative_to(self.source).as_posix()].get("status") == "failed"]
        for path in failed_paths:
            relative = path.relative_to(self.source).as_posix()
            record = progress["records"][relative]
            last_error = record.get("last_error")
            for _ in range(self.retries):
                try:
                    detections = self._predict_single(path, "cpu" if self.device == "0" else self.device)
                    attempts = record.get("attempts", 0) + 1
                    status = "success" if detections else "zero_detections"
                    retry_device = "cpu" if self.device == "0" else self.device
                    metadata = self._write_result(path, detections, status, attempts, device_used="CPU" if retry_device == "cpu" else "CUDA")
                    progress["records"][relative] = {"status": status, "attempts": attempts, "detection_count": len(detections), "class_counts": dict(Counter(d["class_name"] for d in detections)), "metadata_path": self._relative_output("metadata", path).with_suffix(".json").relative_to(self.output).as_posix(), "last_error": None, "processed_at_utc": metadata["processed_at_utc"]}
                    break
                except Exception as retry_exc:
                    last_error = f"{type(retry_exc).__name__}: {retry_exc}"
                    record["attempts"] = record.get("attempts", 0) + 1
            else:
                progress["records"][relative]["attempts"] = record.get("attempts", 0)
                progress["records"][relative]["last_error"] = last_error
                self._write_result(path, [], "failed", record.get("attempts", 0), last_error)
            self.save_progress(progress)
            self._write_summary(progress, len(all_paths), len(existing_keys), candidates)

        summary = self._write_summary(progress, len(all_paths), len(existing_keys), candidates, write_markdown=True)
        self.save_progress(progress)
        recorded = set(progress["records"])
        expected = {path.relative_to(self.source).as_posix() for path in candidates}
        summary["every_eligible_image_has_record"] = expected == recorded
        summary["missing_processing_records"] = sorted(expected - recorded)
        atomic_write_json(self.output / "reports" / "summary.json", summary)
        print("FINAL SUMMARY")
        print(json.dumps(summary, indent=2))
        return summary


def main():
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source or (root / "tiles_640_final")
    output = args.output or (root / "preannotations")
    checkpoint = args.model or Path(os.environ.get("POTATO_MODEL_PATH", root / "runs" / "potato_yolov8n" / "weights" / "best.pt"))
    runner = BulkPreannotator(root, source, output, checkpoint, args.confidence, args.batch_size, args.retries, args.device)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted. Completed batch records are saved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
