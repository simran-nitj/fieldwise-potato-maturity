import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import create_app


class PseudoReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.pre = self.root / "preannotations"
        self.tiles = self.root / "tiles_640_final"
        for name in ("labels", "metadata", "approved_labels"):
            (self.pre / name).mkdir(parents=True)
        self.images = {
            "capture_a/IMG20240101000000_tile_001.jpg": [
                {"class_id": 0, "class_name": "Vegetative", "confidence": 0.72, "x_center": 0.3, "y_center": 0.4, "width": 0.2, "height": 0.3},
                {"class_id": 1, "class_name": "Tuber Bulking", "confidence": 0.61, "x_center": 0.7, "y_center": 0.6, "width": 0.25, "height": 0.2},
            ],
            "capture_b/IMG20240102000000_tile_002.jpg": [
                {"class_id": 2, "class_name": "Maturity", "confidence": 0.84, "x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.35},
            ],
            "capture_c/IMG20240103000000_tile_003.jpg": [],
        }
        records = {}
        for rel, detections in self.images.items():
            image_path = self.tiles / Path(rel)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((80, 120, 3), (50, 130, 60), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            image_path.write_bytes(encoded.tobytes())
            label_path = self.pre / "labels" / Path(rel).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{d['class_id']} {d['x_center']} {d['y_center']} {d['width']} {d['height']}" for d in detections]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            metadata_path = self.pre / "metadata" / Path(rel).with_suffix(".json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps({"status": "zero_detections" if not detections else "success", "confidence_threshold": 0.15, "detections": detections}), encoding="utf-8")
            records[rel] = {"status": "zero_detections" if not detections else "success", "detection_count": len(detections), "class_counts": {}}
        (self.pre / "progress.json").write_text(json.dumps({"records": records}), encoding="utf-8")
        with (self.pre / "manual_approval.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "review_status", "reviewer", "reviewed_at"])
            writer.writeheader()
            writer.writerows({"image_path": key, "review_status": "pending", "reviewer": "", "reviewed_at": ""} for key in records)
        self.original_label_hashes = {key: hashlib.sha256((self.pre / "labels" / Path(key).with_suffix(".txt")).read_bytes()).hexdigest() for key in self.images}
        for folder in ("uploads", "outputs"):
            (self.root / folder).mkdir(exist_ok=True)
        self.app = create_app({"TESTING": True, "PROJECT_ROOT": self.root, "PREANNOTATION_DIR": self.pre, "TILES_DIR": self.tiles, "UPLOAD_DIR": self.root / "uploads", "OUTPUT_DIR": self.root / "outputs", "MODEL_PATH": self.root / "missing.pt"})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def read_csv_rows(self):
        with (self.pre / "manual_approval.csv").open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def item(self, rel):
        response = self.client.get("/review/api/item", query_string={"image_path": rel})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def action(self, rel, action, **extra):
        response = self.client.post("/review/api/action", query_string={"image_path": rel}, json={"action": action, "reviewer": "Reviewer A", **extra})
        return response

    def test_dashboard_filters_multiple_single_zero_and_originals_are_read_only(self):
        page = self.client.get("/review")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Pseudo-annotation review", page.data)
        response = self.client.get("/review/api/items?status=pending")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["items"]), 3)
        self.assertEqual(response.get_json()["counts"]["pending"], 3)
        multi, single, empty = [self.item(path) for path in self.images]
        self.assertEqual(len(multi["predicted_boxes"]), 2)
        self.assertEqual(multi["predicted_boxes"][0]["class_name"], "Vegetative")
        self.assertEqual(multi["predicted_boxes"][0]["confidence"], 0.72)
        self.assertEqual(len(single["predicted_boxes"]), 1)
        self.assertEqual(single["predicted_boxes"][0]["class_name"], "Maturity")
        self.assertEqual(empty["detection_count"], 0)
        self.assertEqual(empty["predicted_boxes"], [])
        image_response = self.client.get(multi["original_url"])
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.mimetype, "image/jpeg")
        self.assertEqual(self.read_csv_rows()[0]["review_status"], "pending")
        self.assertEqual(self.original_label_hashes, {key: hashlib.sha256((self.pre / "labels" / Path(key).with_suffix(".txt")).read_bytes()).hexdigest() for key in self.images})

    def test_approve_reject_and_refresh_persist_without_duplicate_rows(self):
        approved_id, rejected_id = list(self.images)[:2]
        approved_bytes = (self.pre / "labels" / Path(approved_id).with_suffix(".txt")).read_bytes()
        approve = self.action(approved_id, "approve")
        self.assertEqual(approve.status_code, 200, approve.get_data(as_text=True))
        self.assertEqual(approve.get_json()["item"]["review_status"], "approved")
        approved_path = self.pre / "approved_labels" / Path(approved_id).with_suffix(".txt")
        self.assertEqual(approved_path.read_bytes(), approved_bytes)
        reject = self.action(rejected_id, "reject")
        self.assertEqual(reject.status_code, 200)
        self.assertEqual(reject.get_json()["item"]["review_status"], "rejected")
        self.assertFalse((self.pre / "approved_labels" / Path(rejected_id).with_suffix(".txt")).exists())
        self.assertEqual(self.item(approved_id)["review_status"], "approved")
        self.assertEqual(self.item(rejected_id)["review_status"], "rejected")
        rows = self.read_csv_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["image_path"] for row in rows}), 3)
        self.assertEqual(rows[0]["reviewer"], "Reviewer A")
        self.assertTrue(rows[0]["reviewed_at"])
        counts = self.client.get("/review/api/items?status=all").get_json()["counts"]
        self.assertEqual(counts["approved"], 1)
        self.assertEqual(counts["rejected"], 1)
        self.assertAlmostEqual(counts["progress_percent"], 66.7, places=1)

    def test_edit_class_add_remove_save_and_reopen_corrected_annotations(self):
        rel = next(iter(self.images))
        edited_boxes = [
            {"class_id": 2, "cx": 0.45, "cy": 0.5, "width": 0.3, "height": 0.4},
            {"class_id": 1, "cx": 0.7, "cy": 0.7, "width": 0.1, "height": 0.12},
        ]
        response = self.action(rel, "edit", boxes=edited_boxes)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["item"]["review_status"], "edited")
        destination = self.pre / "approved_labels" / Path(rel).with_suffix(".txt")
        self.assertEqual(destination.read_text(encoding="utf-8").splitlines()[0], "2 0.450000 0.500000 0.300000 0.400000")
        reopened = self.item(rel)
        self.assertEqual(reopened["review_status"], "edited")
        self.assertEqual(len(reopened["boxes"]), 2)
        self.assertEqual(reopened["boxes"][0]["class_id"], 2)
        self.assertIsNone(reopened["boxes"][0]["confidence"])
        self.assertEqual(len(reopened["predicted_boxes"]), 2)
        self.assertEqual(self.original_label_hashes[rel], hashlib.sha256((self.pre / "labels" / Path(rel).with_suffix(".txt")).read_bytes()).hexdigest())
        self.assertFalse(list((self.pre / "approved_labels").rglob("*.tmp")))

    def test_approve_empty_only_accepts_zero_detection_image(self):
        multi, _, zero = list(self.images)
        wrong = self.action(multi, "approve_empty")
        self.assertEqual(wrong.status_code, 400)
        response = self.action(zero, "approve_empty")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["item"]["review_status"], "approved_empty")
        path = self.pre / "approved_labels" / Path(zero).with_suffix(".txt")
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), b"")
        self.assertEqual(self.item(zero)["review_status"], "approved_empty")

    def test_edit_validation_and_manifest_duplicate_guard(self):
        rel = next(iter(self.images))
        invalid = self.action(rel, "edit", boxes=[{"class_id": 9, "cx": 0.5, "cy": 0.5, "width": 0.2, "height": 0.2}])
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(self.item(rel)["review_status"], "pending")
        rows = self.read_csv_rows()
        with (self.pre / "manual_approval.csv").open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"{rows[0]['image_path']},pending,,\n")
        response = self.client.get("/review/api/items")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"Duplicate review record", response.data)


if __name__ == "__main__":
    unittest.main()
