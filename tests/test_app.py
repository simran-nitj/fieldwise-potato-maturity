import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import create_app
from app.harvest_estimator import estimate_harvest


class HarvestEstimatorTests(unittest.TestCase):
    def test_valid_estimate_is_transparent(self):
        result = estimate_harvest("2026-01-01", {"Vegetative": 2, "Tuber Bulking": 3, "Maturity": 5}, 100)
        self.assertTrue(result["available"])
        self.assertEqual(result["assumed_duration_days"], 100)
        self.assertIn("observed stage mix", result["explanation"])
        self.assertIn("Provisional", result["disclaimer"])

    def test_invalid_or_insufficient_inputs(self):
        self.assertFalse(estimate_harvest("not-a-date", {"Maturity": 3}, 100)["available"])
        self.assertFalse(estimate_harvest("2026-01-01", {"Maturity": 3}, None)["available"])
        self.assertFalse(estimate_harvest("2026-01-01", {"Maturity": 0}, 100)["available"])
        self.assertFalse(estimate_harvest("2026-01-01", {"Maturity": 3}, 40)["available"])


class FakePredictor:
    model_path = Path("fake-model.pt")
    device = "cpu"
    ready = True
    error = None

    def predict(self, image_path, output_path):
        image = cv2.imread(str(image_path))
        self.assert_readable(image)
        if not cv2.imwrite(str(output_path), image):
            raise OSError("fixture annotation write failed")
        no_detection = "empty" in Path(image_path).name
        counts = {"Vegetative": 0 if no_detection else 1, "Tuber Bulking": 0, "Maturity": 0}
        detections = [] if no_detection else [{"class": "Vegetative", "confidence": 0.91, "box": [1.0, 2.0, 30.0, 40.0]}]
        return {"counts": counts, "total": len(detections), "average_confidence": 0.91 if detections else None, "detections": detections}

    @staticmethod
    def self_assert_readable(_image):
        return None

    def assert_readable(self, image):
        if image is None:
            raise ValueError("bad image")


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        root = Path(self.temp.name)
        (root / "uploads").mkdir()
        (root / "outputs").mkdir()
        self.app = create_app({
            "TESTING": True,
            "PROJECT_ROOT": root,
            "UPLOAD_DIR": root / "uploads",
            "OUTPUT_DIR": root / "outputs",
            "MODEL_PATH": root / "missing.pt",
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def image_bytes(color=(40, 120, 60)):
        image = np.full((48, 64, 3), color, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("image fixture encoding failed")
        return encoded.tobytes()

    def test_dashboard_and_graceful_missing_model(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model inference is unavailable", response.data)
        self.assertIn(b"SAVED ASSESSMENTS", response.data)

    def test_navigation_pages_and_empty_history(self):
        for path in ("/history", "/about"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
        self.assertIn(b"archive is empty", self.client.get("/history").data)

    def test_missing_invalid_and_oversized_uploads(self):
        missing = self.client.post("/assessment", data={})
        self.assertEqual(missing.status_code, 302)
        invalid = self.client.post("/assessment", data={"planting_date": "2026-01-01", "images": (io.BytesIO(b"data"), "bad.gif")}, content_type="multipart/form-data")
        page = self.client.get(invalid.location, follow_redirects=True)
        self.assertIn(b"Unsupported image type", page.data)
        unreadable = self.client.post("/assessment", data={"planting_date": "2026-01-01", "images": (io.BytesIO(b"not an image"), "broken.jpg")}, content_type="multipart/form-data")
        page = self.client.get(unreadable.location, follow_redirects=True)
        self.assertIn(b"Could not read broken.jpg", page.data)
        too_large = create_app({"TESTING": True, "PROJECT_ROOT": Path(self.temp.name), "UPLOAD_DIR": Path(self.temp.name) / "uploads", "OUTPUT_DIR": Path(self.temp.name) / "outputs", "MODEL_PATH": Path(self.temp.name) / "missing.pt", "MAX_CONTENT_LENGTH": 100})
        response = too_large.test_client().post("/assessment", data={"images": (io.BytesIO(b"x" * 1000), "large.png")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)

    def test_assessment_validation(self):
        payload = self.image_bytes()
        bad_date = self.client.post("/assessment", data={"planting_date": "nope", "images": (io.BytesIO(payload), "leaf.png")}, content_type="multipart/form-data")
        self.assertIn(b"valid planting date", self.client.get(bad_date.location, follow_redirects=True).data)
        bad_duration = self.client.post("/assessment", data={"planting_date": "2026-01-01", "assumed_duration_days": "50", "images": (io.BytesIO(payload), "leaf.png")}, content_type="multipart/form-data")
        self.assertIn(b"between 60 and 180", self.client.get(bad_duration.location, follow_redirects=True).data)

    def test_multiple_images_zero_detection_history_images_and_report(self):
        model = FakePredictor()
        data = {
            "field_name": "North Field",
            "planting_date": "2026-01-01",
            "assumed_duration_days": "100",
            "variety": "Kufri Jyoti",
            "region": "Demo village",
            "images": [
                (io.BytesIO(self.image_bytes()), "leaf.png"),
                (io.BytesIO(self.image_bytes((80, 80, 20))), "empty-sample.png"),
            ],
        }
        with patch("app.routes.predictor", return_value=model):
            response = self.client.post("/assessment", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        result_location = response.location
        result = self.client.get(result_location)
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"Vegetative", result.data)
        self.assertIn(b"91.0%", result.data)
        self.assertIn(b"No detections in this image", result.data)
        image_urls = re.findall(rb'src="([^"]*assessment-image[^"]*)"', result.data)
        self.assertEqual(len(image_urls), 4)
        for image_url in image_urls:
            response = self.client.get(image_url.decode())
            self.assertEqual(response.status_code, 200)
            response.close()
        assessment_id = result_location.rsplit("/", 1)[-1]
        output = Path(self.temp.name) / "outputs" / assessment_id
        saved = json.loads((output / "assessment.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["stage_counts"]["Vegetative"], 1)
        self.assertEqual(saved["images"][0]["detections"][0]["class"], "Vegetative")
        print_report = self.client.get(f"/report/{assessment_id}/print")
        self.assertEqual(print_report.status_code, 200)
        self.assertIn(b"Original image", print_report.data)
        download = self.client.get(f"/report/{assessment_id}")
        self.assertEqual(download.status_code, 200)
        self.assertIn(b"data:image/jpeg;base64,", download.data)
        history = self.client.get("/history?q=North")
        self.assertEqual(history.status_code, 200)
        self.assertIn(b"North Field", history.data)
        dashboard = self.client.get("/")
        self.assertIn(b">1</strong>", dashboard.data)
        self.assertIn(b">2</strong>", dashboard.data)

    def test_assessment_image_path_traversal_is_rejected(self):
        response = self.client.get("/assessment-image/" + "a" * 32 + "/..%5Csecret.txt")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
