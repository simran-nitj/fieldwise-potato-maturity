import threading
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}


class Predictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.device = "cpu"
        self.error = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not self.model_path.is_file():
            self.error = "Trained model checkpoint was not found. Set POTATO_MODEL_PATH or restore best.pt."
            return
        try:
            import torch
            from ultralytics import YOLO

            self.device = "0" if torch.cuda.is_available() else "cpu"
            model = YOLO(str(self.model_path))
            names = model.names
            mapped = {int(k): str(v).strip() for k, v in names.items()}
            if mapped != CLASS_NAMES:
                self.error = f"Checkpoint classes did not match the expected project classes: {mapped}"
                return
            self.model = model
        except Exception as exc:  # surfaced on dashboard without exposing local paths
            self.error = f"Model could not be loaded ({type(exc).__name__}). Check the installed Ultralytics/PyTorch environment."

    @property
    def ready(self):
        return self.model is not None

    def predict(self, image_path, output_path):
        if not self.ready:
            raise RuntimeError(self.error or "Model is unavailable.")
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            raise ValueError("This file is not a readable image.")
        with self._lock:
            result = self.model.predict(source=image, device=self.device, imgsz=640, batch=1, verbose=False)[0]
        counts = {name: 0 for name in CLASS_NAMES.values()}
        confidences = []
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                name = CLASS_NAMES.get(class_id, "Unknown")
                confidence = float(box.conf.item())
                if name in counts:
                    counts[name] += 1
                confidences.append(confidence)
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({"class": name, "confidence": round(confidence, 4), "box": xyxy})
        annotated = result.plot()
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError("Could not save annotated image.")
        return {
            "counts": counts,
            "total": len(detections),
            "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "detections": detections,
        }
