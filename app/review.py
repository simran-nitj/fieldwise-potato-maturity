"""Read-only pseudo-label inspection and explicitly human-driven review writes."""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

CLASSES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
REVIEW_STATUSES = {"pending", "approved", "rejected", "edited", "approved_empty"}
REVIEWED_STATUSES = REVIEW_STATUSES - {"pending"}
CSV_REQUIRED_COLUMNS = ("image_path", "review_status", "reviewer", "reviewed_at")


class ReviewDataError(RuntimeError):
    """Raised when review sources are incomplete or inconsistent."""


class ReviewStore:
    def __init__(self, project_root: Path, preannotation_root: Path):
        self.project_root = Path(project_root).resolve()
        self.root = Path(preannotation_root).resolve()
        self.tiles = (self.project_root / "tiles_640_final").resolve()
        self.progress_path = self.root / "progress.json"
        self.csv_path = self.root / "manual_approval.csv"
        self.labels_path = self.root / "labels"
        self.metadata_path = self.root / "metadata"
        self.approved_path = self.root / "approved_labels"
        self._lock = threading.RLock()

    def _relative_image(self, value: str) -> PurePosixPath:
        rel = PurePosixPath(str(value).replace("\\", "/"))
        if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
            raise ReviewDataError(f"Unsafe image path in review manifest: {value!r}")
        if Path(rel.name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            raise ReviewDataError(f"Unsupported image path in review manifest: {value!r}")
        return rel

    def _load_progress(self):
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewDataError("Pre-annotation progress.json is missing or invalid.") from exc
        records = data.get("records")
        if not isinstance(records, dict):
            raise ReviewDataError("Pre-annotation progress.json has no image records.")
        return records

    def _read_rows(self):
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                missing = set(CSV_REQUIRED_COLUMNS) - set(fieldnames)
                if missing:
                    raise ReviewDataError(f"manual_approval.csv is missing columns: {sorted(missing)}")
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ReviewDataError("manual_approval.csv could not be read.") from exc

        seen = set()
        for row in rows:
            rel = self._relative_image(row.get("image_path", "")).as_posix()
            if rel in seen:
                raise ReviewDataError(f"Duplicate review record for {rel}.")
            seen.add(rel)
            status = (row.get("review_status") or "pending").strip().lower()
            if status not in REVIEW_STATUSES:
                raise ReviewDataError(f"Unsupported review status {status!r} for {rel}.")
            row["image_path"] = rel
            row["review_status"] = status
            row["reviewer"] = (row.get("reviewer") or "").strip()
            row["reviewed_at"] = (row.get("reviewed_at") or "").strip()

        progress = self._load_progress()
        progress_ids = {self._relative_image(key).as_posix() for key in progress}
        if seen != progress_ids:
            missing_rows = sorted(progress_ids - seen)
            extra_rows = sorted(seen - progress_ids)
            raise ReviewDataError(f"Review manifest and progress records differ (missing rows={len(missing_rows)}, extra rows={len(extra_rows)}).")
        self._fieldnames = fieldnames
        return rows, progress

    def dashboard(self):
        with self._lock:
            rows, _ = self._read_rows()
        counts = {status: 0 for status in REVIEW_STATUSES}
        for row in rows:
            counts[row["review_status"]] += 1
        total = len(rows)
        reviewed = sum(counts[status] for status in REVIEWED_STATUSES)
        return {
            "total": total,
            "pending": counts["pending"],
            "approved": counts["approved"],
            "rejected": counts["rejected"],
            "edited": counts["edited"],
            "approved_empty": counts["approved_empty"],
            "reviewed": reviewed,
            "progress_percent": round(100 * reviewed / total, 1) if total else 0.0,
        }

    def list_items(self, status="pending", query=""):
        if status not in {"all", *REVIEW_STATUSES}:
            raise ValueError("Unknown review filter.")
        needle = query.strip().casefold()
        with self._lock:
            rows, progress = self._read_rows()
        filtered = []
        for row in rows:
            if status != "all" and row["review_status"] != status:
                continue
            if needle and needle not in Path(row["image_path"]).name.casefold():
                continue
            state = progress[row["image_path"]]
            filtered.append({
                "image_path": row["image_path"],
                "filename": Path(row["image_path"]).name,
                "review_status": row["review_status"],
                "detection_count": int(state.get("detection_count", 0)),
            })
        return filtered

    def get_item(self, image_id: str):
        rel = self._relative_image(image_id).as_posix()
        with self._lock:
            rows, progress = self._read_rows()
        by_id = {row["image_path"]: row for row in rows}
        if rel not in by_id:
            raise KeyError("Unknown pseudo-annotation image.")
        row = by_id[rel]
        record = progress[rel]
        original_path = self.tiles.joinpath(*PurePosixPath(rel).parts)
        if self.tiles not in original_path.resolve().parents or not original_path.is_file():
            raise ReviewDataError(f"Original tile is missing: {rel}")
        metadata_rel = PurePosixPath("metadata") / PurePosixPath(rel).with_suffix(".json")
        metadata_file = self.root.joinpath(*metadata_rel.parts)
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewDataError(f"Pre-annotation metadata is missing or invalid for {rel}.") from exc
        prediction_file = self.root / "labels" / PurePosixPath(rel).with_suffix(".txt")
        predicted_rows = self._read_yolo_file(prediction_file)
        detections = metadata.get("detections", [])
        predicted_boxes = self._build_boxes(predicted_rows, detections, show_confidence=True)

        current_boxes = predicted_boxes
        if row["review_status"] == "edited":
            corrected_file = self.approved_path / PurePosixPath(rel).with_suffix(".txt")
            corrected_rows = self._read_yolo_file(corrected_file)
            current_boxes = self._build_boxes(corrected_rows, [], show_confidence=False)
        return {
            "image_path": rel,
            "filename": Path(rel).name,
            "review_status": row["review_status"],
            "reviewer": row["reviewer"],
            "reviewed_at": row["reviewed_at"],
            "detection_count": int(record.get("detection_count", len(predicted_boxes))),
            "class_counts": record.get("class_counts", {}),
            "original_url": f"/review/image/{rel}",
            "predicted_boxes": predicted_boxes,
            "boxes": current_boxes,
            "metadata": {
                "confidence_threshold": metadata.get("confidence_threshold"),
                "processed_at_utc": metadata.get("processed_at_utc"),
                "status": metadata.get("status"),
            },
        }

    def _read_yolo_file(self, path: Path):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReviewDataError(f"YOLO label file is missing: {path.name}") from exc
        rows = []
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ReviewDataError(f"Invalid YOLO row {line_no} in {path.name}.")
            try:
                class_value, cx, cy, width, height = map(float, parts)
            except ValueError as exc:
                raise ReviewDataError(f"Non-numeric YOLO row {line_no} in {path.name}.") from exc
            if not class_value.is_integer() or int(class_value) not in CLASSES:
                raise ReviewDataError(f"Invalid class ID at row {line_no} in {path.name}.")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, width, height)) or width <= 0 or height <= 0:
                raise ReviewDataError(f"Invalid normalized box at row {line_no} in {path.name}.")
            if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
                raise ReviewDataError(f"Box exceeds image bounds at row {line_no} in {path.name}.")
            rows.append((int(class_value), cx, cy, width, height))
        return rows

    @staticmethod
    def _build_boxes(rows, detections, show_confidence):
        result = []
        for index, (class_id, cx, cy, width, height) in enumerate(rows):
            detection = detections[index] if show_confidence and index < len(detections) else {}
            confidence = detection.get("confidence")
            result.append({
                "id": index,
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "confidence": float(confidence) if confidence is not None else None,
                "manual": False,
            })
        return result

    @staticmethod
    def _atomic_write(path: Path, payload: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _atomic_write_manifest(self, rows):
        path = self.csv_path
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def update(self, image_id: str, action: str, reviewer: str, boxes=None):
        reviewer = (reviewer or "").strip()
        if not reviewer or len(reviewer) > 100:
            raise ValueError("Enter your name or reviewer ID (1–100 characters).")
        rel = self._relative_image(image_id).as_posix()
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with self._lock:
            rows, progress = self._read_rows()
            row_by_id = {row["image_path"]: row for row in rows}
            if rel not in row_by_id:
                raise KeyError("Unknown pseudo-annotation image.")
            row = row_by_id[rel]
            original_label = self.root / "labels" / PurePosixPath(rel).with_suffix(".txt")
            destination = self.approved_path / PurePosixPath(rel).with_suffix(".txt")

            if action == "approve":
                if row["review_status"] == "edited":
                    raise ValueError("This image has corrected boxes. Keep its edited status or reopen Edit and save corrections.")
                predicted_rows = self._read_yolo_file(original_label)
                if int(progress[rel].get("detection_count", len(predicted_rows))) == 0 and not predicted_rows:
                    raise ValueError("Use Approve Empty to explicitly review an image with no predicted boxes.")
                label_payload = original_label.read_bytes()
                new_status = "approved"
            elif action == "reject":
                label_payload = None
                new_status = "rejected"
            elif action == "approve_empty":
                state = progress[rel]
                predicted = self._read_yolo_file(original_label)
                if int(state.get("detection_count", len(predicted))) != 0 or predicted:
                    raise ValueError("Approve Empty is available only for an image with zero predicted boxes.")
                label_payload = b""
                new_status = "approved_empty"
            elif action == "edit":
                if not isinstance(boxes, list):
                    raise ValueError("Corrected box list is required.")
                lines = []
                for box in boxes:
                    try:
                        class_id = int(box["class_id"])
                        cx, cy = float(box["cx"]), float(box["cy"])
                        width, height = float(box["width"]), float(box["height"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("Each corrected box needs a class ID and normalized coordinates.") from exc
                    if class_id not in CLASSES:
                        raise ValueError(f"Unknown class ID {class_id}.")
                    coords = (cx, cy, width, height)
                    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords) or width <= 0 or height <= 0:
                        raise ValueError("Box coordinates must be normalized, finite, and have positive size.")
                    if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
                        raise ValueError("Every corrected box must remain inside the image.")
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
                label_payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
                new_status = "edited"
            else:
                raise ValueError("Unsupported review action.")

            if label_payload is not None:
                self._atomic_write(destination, label_payload)
            row["review_status"] = new_status
            row["reviewer"] = reviewer
            row["reviewed_at"] = timestamp
            self._atomic_write_manifest(rows)
            return {"image_path": rel, "review_status": new_status, "reviewer": reviewer, "reviewed_at": timestamp}
