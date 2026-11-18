"""Local browser annotation workspace. Model boxes remain suggestions until reviewed."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template_string, request, send_file

CLASSES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CAPTURE_RE = re.compile(r"IMG\d{14}", re.IGNORECASE)


def capture_group(path: Path) -> str:
    match = CAPTURE_RE.search(path.name)
    return match.group(0).upper() if match else path.stem.split("_tile_")[0]


def tile_key(path: Path) -> str:
    group = capture_group(path)
    match = re.search(r"_tile_(\d+)", path.stem, re.IGNORECASE)
    return f"{group}_tile_{int(match.group(1)):03d}" if match else path.stem.lower()


class AnnotationWorkspace:
    def __init__(self, root: Path, source_dir: Path | None = None, output_dir: Path | None = None):
        self.root = Path(root).resolve()
        self.source_dir = Path(source_dir or self.root / "tiles_640_final").resolve()
        self.output_dir = Path(output_dir or self.root / "verified_annotations").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.csv_path = self.output_dir / "annotations.csv"
        self._lock = threading.RLock()
        self._model = None
        self._model_error = None
        self._candidates = self._discover()
        self._candidate_ids = {p.relative_to(self.source_dir).as_posix(): p for p in self._candidates}

    def _split_groups(self, split):
        image_dir = self.root / "potato_yolo_dataset" / "images" / split
        return {capture_group(p) for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES} if image_dir.exists() else set()

    def _source_plot_by_capture(self):
        source_root = self.root / "dataset" / "RGB images"
        mapping = {}
        if source_root.exists():
            for path in source_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    mapping[path.stem.upper()] = path.parent.name
        return mapping

    def _labeled_keys(self):
        keys = set()
        for split in ("train", "val", "test"):
            image_dir = self.root / "potato_yolo_dataset" / "images" / split
            label_dir = self.root / "potato_yolo_dataset" / "labels" / split
            if not image_dir.exists() or not label_dir.exists():
                continue
            for image in image_dir.rglob("*"):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = label_dir / f"{image.stem}.txt"
                if label.exists() and label.stat().st_size > 0:
                    keys.add(tile_key(image))
        annotation_csv = self.root / "annotations.csv"
        if annotation_csv.exists():
            try:
                with annotation_csv.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        name = (row.get("filename") or "").replace("\\", "/")
                        if name:
                            keys.add(tile_key(Path(name)))
            except (OSError, UnicodeError, csv.Error):
                pass
        return keys

    def _read_progress(self):
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _discover(self):
        if not self.source_dir.exists():
            return []
        protected = self._split_groups("val") | self._split_groups("test")
        source_plot_by_capture = self._source_plot_by_capture()
        protected_plot_groups = {source_plot_by_capture[capture] for capture in protected if capture in source_plot_by_capture}
        labeled = self._labeled_keys()
        progress = self._read_progress()
        candidates = []
        for path in sorted(self.source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            group = capture_group(path)
            rel = path.relative_to(self.source_dir).as_posix()
            source_plot_group = source_plot_by_capture.get(group)
            if group in protected or source_plot_group in protected_plot_groups or tile_key(path) in labeled:
                continue
            if progress.get(rel, {}).get("complete"):
                continue
            candidates.append(path)
        return candidates

    def image_id(self, image_id):
        path = self._candidate_ids.get(image_id)
        if path is None:
            raise KeyError("Unknown or protected image")
        return path

    def get_model(self, model_path: Path):
        with self._lock:
            if self._model is not None:
                return self._model
            if self._model_error:
                raise RuntimeError(self._model_error)
            if not model_path.is_file():
                raise RuntimeError("Trained checkpoint not found. Set POTATO_MODEL_PATH to a valid best.pt file.")
            try:
                import torch
                from ultralytics import YOLO

                model = YOLO(str(model_path))
                names = {int(key): str(value).strip() for key, value in model.names.items()}
                if names != CLASSES:
                    raise RuntimeError(f"Checkpoint class names do not match: {names}")
                self.device = "0" if torch.cuda.is_available() else "cpu"
                self._model = model
                return model
            except Exception as exc:
                self._model_error = f"Could not load checkpoint ({type(exc).__name__}). Check Ultralytics/PyTorch compatibility."
                raise RuntimeError(self._model_error) from exc

    def item(self, image_id, model_path):
        path = self.image_id(image_id)
        with self._lock:
            progress = self._read_progress()
            saved = progress.get(image_id)
            if saved:
                return {"id": image_id, "image_url": f"/image?id={image_id}", "boxes": saved.get("boxes", []), "complete": saved.get("complete", False), "saved": True}
            model = self.get_model(model_path)
            result = model.predict(source=str(path), device=self.device, imgsz=640, conf=0.05, batch=1, verbose=False)[0]
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id not in CLASSES:
                        continue
                    x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                    height, width = result.orig_shape
                    boxes.append({"x": max(0, x1 / width), "y": max(0, y1 / height), "w": max(0, (x2 - x1) / width), "h": max(0, (y2 - y1) / height), "class_id": class_id, "confidence": float(box.conf.item()), "source": "model", "status": "pending"})
            return {"id": image_id, "image_url": f"/image?id={image_id}", "boxes": boxes, "complete": False, "saved": False, "device": "CUDA" if self.device == "0" else "CPU"}

    def save(self, image_id, boxes, complete=True):
        path = self.image_id(image_id)
        if not isinstance(boxes, list):
            raise ValueError("Boxes must be a list")
        cleaned = []
        for box in boxes:
            class_id = int(box.get("class_id", -1))
            x, y, w, h = [float(box.get(key, -1)) for key in ("x", "y", "w", "h")]
            source, status = box.get("source"), box.get("status")
            if class_id not in CLASSES or not all(0 <= value <= 1 for value in (x, y, w, h)) or w <= 0 or h <= 0 or x + w > 1.00001 or y + h > 1.00001:
                raise ValueError("Invalid class or normalized bounding box")
            if source not in {"model", "manual"} or status not in {"pending", "accepted", "rejected"}:
                raise ValueError("Invalid box review state")
            confidence = box.get("confidence")
            cleaned.append({"x": x, "y": y, "w": w, "h": h, "class_id": class_id, "confidence": float(confidence) if confidence is not None else None, "source": source, "status": status})
        if complete and any(box["source"] == "model" and box["status"] == "pending" for box in cleaned):
            raise ValueError("Accept or reject every model suggestion before marking this image reviewed")
        record = {"complete": bool(complete), "boxes": cleaned}
        with self._lock:
            progress = self._read_progress()
            progress[image_id] = record
            self._write_json_atomic(self.progress_path, progress)
            if complete:
                self._write_yolo(path, cleaned)
            self._write_csv(progress)
        return record

    @staticmethod
    def _write_json_atomic(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _write_yolo(self, image_path, boxes):
        rel = image_path.relative_to(self.source_dir)
        output = self.output_dir / "labels" / "train" / rel.parent / f"{rel.stem}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for box in boxes:
            if box["source"] == "manual" or box["status"] == "accepted":
                cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
                lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {box['w']:.6f} {box['h']:.6f}")
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_csv(self, progress):
        rows = []
        for image_id, record in sorted(progress.items()):
            if not record.get("complete"):
                continue
            for box in record.get("boxes", []):
                if box["source"] != "manual" and box["status"] != "accepted":
                    continue
                rows.append([image_id, box["class_id"], CLASSES[box["class_id"]], box["x"] + box["w"] / 2, box["y"] + box["h"] / 2, box["w"], box["h"], box["source"], box["confidence"] if box["confidence"] is not None else ""])
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "class_id", "class_name", "x_center", "y_center", "width", "height", "source", "confidence"])
            writer.writerows(rows)
        temporary.replace(self.csv_path)


def create_app(root=None, source_dir=None, output_dir=None, model_path=None):
    project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    workspace = AnnotationWorkspace(project_root, source_dir, output_dir)
    checkpoint = Path(model_path or os.environ.get("POTATO_MODEL_PATH", project_root / "runs" / "potato_yolov8n" / "weights" / "best.pt"))
    app = Flask(__name__)

    @app.get("/")
    def home():
        page = Path(__file__).with_name("annotation_tool.html").read_text(encoding="utf-8")
        return render_template_string(page)

    @app.get("/api/candidates")
    def candidates():
        progress = workspace._read_progress()
        return jsonify([{"id": p.relative_to(workspace.source_dir).as_posix(), "name": p.name, "status": "Saved progress" if p.relative_to(workspace.source_dir).as_posix() in progress else "Not started"} for p in workspace._candidates])

    @app.get("/api/item")
    def item():
        image_id = request.args.get("id", "")
        try:
            return jsonify(workspace.item(image_id, checkpoint))
        except (KeyError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/image")
    def image():
        try:
            return send_file(workspace.image_id(request.args.get("id", "")))
        except KeyError:
            return jsonify({"error": "Unknown or protected image"}), 404

    @app.post("/api/save")
    def save():
        data = request.get_json(silent=True) or {}
        try:
            saved = workspace.save(str(data.get("id", "")), data.get("boxes", []), bool(data.get("complete", True)))
            return jsonify({"saved": True, "complete": saved["complete"]})
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/status")
    def status():
        return jsonify({"checkpoint_configured": checkpoint.is_file(), "classes": list(CLASSES.values()), "source": workspace.source_dir.name, "device": "CUDA when available, otherwise CPU"})

    app.workspace = workspace
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=None, help="Image directory; defaults to tiles_640_final")
    parser.add_argument("--output", type=Path, default=None, help="Verified-label directory; defaults to verified_annotations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    app = create_app(args.root, args.source, args.output)
    print(f"Annotation tool listening at http://{args.host}:{args.port}")
    print("Validation/test capture and source-plot groups are excluded; model boxes require individual human review.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
