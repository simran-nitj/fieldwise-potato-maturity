import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_dataset import parse_label
from scripts.annotation_tool import AnnotationWorkspace, create_app


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), np.full((64, 64, 3), 80, dtype=np.uint8))


class DatasetAuditTests(unittest.TestCase):
    def test_valid_and_invalid_yolo_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            label = Path(temp) / "box.txt"
            label.write_text("2 0.5 0.5 0.5 0.6\n3 0.5 0.5 0.2 0.2\n0 0.1 0.2 0 0.3\n", encoding="utf-8")
            rows, problems = parse_label(label, 640, 640)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["class_id"], 2)
            self.assertEqual(len(problems), 2)


class AnnotationWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.source = self.root / "tiles_640_final"
        self.labels_image = "IMG20240101120000_tile_001.jpg"
        make_image(self.source / "IMG20240101120000" / self.labels_image)
        make_image(self.source / "IMG20240101120000" / "IMG20240101120000_tile_002.jpg")
        make_image(self.source / "IMG20240102120000" / "IMG20240102120000_tile_001.jpg")
        make_image(self.source / "IMG20240103120000" / "IMG20240103120000_tile_001.jpg")
        make_image(self.source / "IMG20240104120000" / "IMG20240104120000_tile_001.jpg")
        raw_root = self.root / "dataset" / "RGB images" / "Year2023-24_RGB"
        make_image(raw_root / "T1_T1" / "IMG20240101120000.jpg")
        make_image(raw_root / "T1_T3" / "IMG20240102120000.jpg")
        make_image(raw_root / "T1_T2" / "IMG20240103120000.jpg")
        make_image(raw_root / "T1_T3" / "IMG20240104120000.jpg")
        make_image(self.root / "potato_yolo_dataset" / "images" / "train" / "IMG20240101120000_IMG20240101120000_tile_001.jpg")
        make_image(self.root / "potato_yolo_dataset" / "images" / "val" / "IMG20240102120000_IMG20240102120000_tile_001.jpg")
        make_image(self.root / "potato_yolo_dataset" / "images" / "test" / "IMG20240103120000_IMG20240103120000_tile_001.jpg")
        train_label = self.root / "potato_yolo_dataset" / "labels" / "train" / "IMG20240101120000_IMG20240101120000_tile_001.txt"
        train_label.parent.mkdir(parents=True, exist_ok=True)
        train_label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        self.workspace = AnnotationWorkspace(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_discovery_skips_labeled_validation_and_test_groups(self):
        ids = {path.relative_to(self.source).as_posix() for path in self.workspace._candidates}
        self.assertEqual(ids, {
            "IMG20240101120000/IMG20240101120000_tile_002.jpg",
        })

    def test_only_human_accepted_and_manual_boxes_reach_yolo(self):
        image_id = "IMG20240101120000/IMG20240101120000_tile_002.jpg"
        pending = [{"x": .1, "y": .2, "w": .3, "h": .4, "class_id": 0, "confidence": .31, "source": "model", "status": "pending"}]
        with self.assertRaisesRegex(ValueError, "Accept or reject"):
            self.workspace.save(image_id, pending, complete=True)
        boxes = [
            {**pending[0], "status": "accepted"},
            {"x": .5, "y": .1, "w": .2, "h": .2, "class_id": 2, "confidence": .8, "source": "model", "status": "rejected"},
            {"x": .2, "y": .2, "w": .1, "h": .2, "class_id": 1, "confidence": None, "source": "manual", "status": "accepted"},
        ]
        self.workspace.save(image_id, boxes, complete=True)
        label = self.root / "verified_annotations" / "labels" / "train" / "IMG20240101120000" / "IMG20240101120000_tile_002.txt"
        lines = label.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("0 ", lines[0])
        self.assertIn("1 ", lines[1])
        self.assertEqual(len((self.root / "verified_annotations" / "annotations.csv").read_text(encoding="utf-8").splitlines()), 3)

    def test_local_browser_routes_render_and_expose_only_eligible_items(self):
        app = create_app(self.root, self.source, self.root / "tool-output", self.root / "missing.pt")
        client = app.test_client()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Assisted annotation review", page.data)
        response = client.get("/api/candidates")
        self.assertEqual(response.status_code, 200)
        ids = {row["id"] for row in response.get_json()}
        self.assertEqual(ids, {"IMG20240101120000/IMG20240101120000_tile_002.jpg"})


if __name__ == "__main__":
    unittest.main()
