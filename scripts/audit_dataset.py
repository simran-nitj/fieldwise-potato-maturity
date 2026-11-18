"""Read-only integrity and split audit for the potato YOLO dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CAPTURE_RE = re.compile(r"IMG\d{14}", re.IGNORECASE)


def capture_group(path: Path) -> str:
    match = CAPTURE_RE.search(path.name)
    return match.group(0).upper() if match else path.stem.split("_tile_")[0]


def parse_label(path: Path, width: int, height: int):
    rows, problems = [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return rows, [f"unreadable: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {line_number}: expected 5 fields, found {len(parts)}")
            continue
        try:
            class_raw = float(parts[0])
            box = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append(f"line {line_number}: non-numeric value")
            continue
        if not class_raw.is_integer() or int(class_raw) not in CLASSES:
            problems.append(f"line {line_number}: invalid class id {parts[0]}")
            continue
        if not all(np.isfinite(box)):
            problems.append(f"line {line_number}: non-finite box value")
            continue
        cx, cy, bw, bh = box
        if not all(0 <= value <= 1 for value in box):
            problems.append(f"line {line_number}: normalized value outside [0, 1]")
            continue
        if bw <= 0 or bh <= 0:
            problems.append(f"line {line_number}: zero/negative width or height")
            continue
        if cx - bw / 2 < -1e-6 or cy - bh / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6 or cy + bh / 2 > 1 + 1e-6:
            problems.append(f"line {line_number}: box extends outside image")
            continue
        rows.append({"class_id": int(class_raw), "cx": cx, "cy": cy, "w": bw, "h": bh, "pixels_w": bw * width, "pixels_h": bh * height, "area_fraction": bw * bh})
    return rows, problems


def image_phash(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct.flatten()[1:]
    median = float(np.median(values))
    return sum((int(value > median) << index) for index, value in enumerate(values))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path):
    dataset = root / "potato_yolo_dataset"
    split_data, all_items = {}, []
    findings = {"invalid_labels": [], "missing_labels": [], "missing_images": [], "orphan_labels": [], "unreadable_images": []}
    distribution, box_widths, box_heights, box_areas = {}, [], [], []
    for split in ("train", "val", "test"):
        image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
        labels = sorted(label_dir.rglob("*.txt")) if label_dir.exists() else []
        by_stem = {p.stem.lower(): p for p in labels}
        seen_stems = {p.stem.lower() for p in images}
        counts = Counter()
        empty_images, instance_count, split_items = [], 0, []
        for image_path in images:
            label_path = by_stem.get(image_path.stem.lower())
            capture = capture_group(image_path)
            try:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("OpenCV could not decode image")
                height, width = img.shape[:2]
            except Exception as exc:
                findings["unreadable_images"].append({"split": split, "file": image_path.relative_to(root).as_posix(), "reason": str(exc)})
                continue
            if label_path is None:
                findings["missing_labels"].append({"split": split, "image": image_path.relative_to(root).as_posix()})
                rows = []
                empty_images.append(image_path.relative_to(root).as_posix())
            else:
                rows, problems = parse_label(label_path, width, height)
                findings["invalid_labels"].extend({"split": split, "file": label_path.relative_to(root).as_posix(), "problem": p} for p in problems)
                if not rows:
                    empty_images.append(image_path.relative_to(root).as_posix())
            for row in rows:
                counts[row["class_id"]] += 1
                box_widths.append(row["pixels_w"])
                box_heights.append(row["pixels_h"])
                box_areas.append(row["area_fraction"])
            instance_count += len(rows)
            entry = {"split": split, "path": image_path, "relative": image_path.relative_to(root).as_posix(), "capture": capture, "counts": Counter(row["class_id"] for row in rows), "hash": sha256(image_path), "phash": image_phash(image_path)}
            split_items.append(entry)
            all_items.append(entry)
        for label_path in labels:
            if label_path.stem.lower() not in seen_stems:
                findings["orphan_labels"].append({"split": split, "label": label_path.relative_to(root).as_posix()})
        distribution[split] = {"images": len(images), "instances": instance_count, "per_class": {CLASSES[c]: counts[c] for c in CLASSES}, "empty_or_unlabeled_images": len(empty_images), "empty_or_unlabeled_examples": empty_images[:20], "capture_groups": sorted({i["capture"] for i in split_items})}
        split_data[split] = split_items

    exact_duplicates, near_duplicates, all_near_pairs = [], [], []
    for idx, first in enumerate(all_items):
        for second in all_items[idx + 1:]:
            if first["hash"] == second["hash"]:
                if first["split"] != second["split"]:
                    exact_duplicates.append((first, second, 0))
                all_near_pairs.append((first, second, 0))
            elif first["phash"] is not None and second["phash"] is not None:
                distance = (first["phash"] ^ second["phash"]).bit_count()
                if distance <= 5:
                    all_near_pairs.append((first, second, distance))
                    if first["split"] != second["split"]:
                        near_duplicates.append((first, second, distance))

    group_sets = {split: {item["capture"] for item in items} for split, items in split_data.items()}
    group_overlaps = {f"{a}-{b}": sorted(group_sets[a] & group_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    source_rgb = root / "dataset" / "RGB images"
    source_plot_by_capture = {}
    if source_rgb.exists():
        for raw_path in source_rgb.rglob("*"):
            if raw_path.is_file() and raw_path.suffix.lower() in IMAGE_SUFFIXES:
                source_plot_by_capture[raw_path.stem.upper()] = raw_path.parent.name
    source_plot_sets = {
        split: {source_plot_by_capture[item["capture"]] for item in items if item["capture"] in source_plot_by_capture}
        for split, items in split_data.items()
    }
    source_plot_overlaps = {f"{a}-{b}": sorted(source_plot_sets[a] & source_plot_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    preprocessed = root / "preprocessed_images"
    mask_stats = []
    if source_rgb.exists() and preprocessed.exists():
        for raw_path in source_rgb.rglob("*"):
            if not raw_path.is_file() or raw_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            filtered_path = preprocessed / f"{raw_path.stem}.jpg"
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            filtered = cv2.imread(str(filtered_path), cv2.IMREAD_COLOR) if filtered_path.exists() else None
            if raw is None or filtered is None or raw.shape[:2] != filtered.shape[:2]:
                continue
            retained = np.any(filtered > 8, axis=2)
            hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
            yellow_brown_like = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35)
            mask_stats.append({"name": raw_path.name, "retained_fraction": float(retained.mean()), "yellow_brown_retained_fraction": float(retained[yellow_brown_like].mean()) if yellow_brown_like.any() else None, "yellow_brown_pixel_fraction": float(yellow_brown_like.mean())})

    annotation_csv = root / "annotations.csv"
    csv_class_counts = Counter()
    csv_rows = 0
    csv_missing_images = []
    if annotation_csv.exists():
        with annotation_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                csv_rows += 1
                try:
                    csv_class_counts[CLASSES[int(row["class_id"])]] += 1
                except (KeyError, ValueError):
                    pass
                filename = (row.get("filename") or "").replace("\\", "/")
                annotated_image = (row.get("annotated_image") or "").replace("\\", "/")
                if filename and not (root / "tiles_640_final" / Path(filename)).is_file():
                    csv_missing_images.append({"field": "filename", "value": filename})
                if annotated_image and not (root / Path(annotated_image)).is_file():
                    csv_missing_images.append({"field": "annotated_image", "value": annotated_image})
    findings["missing_images"] = csv_missing_images

    def describe(values):
        if not values:
            return None
        ordered = sorted(values)
        return {"min": round(min(ordered), 4), "median": round(statistics.median(ordered), 4), "p90": round(float(np.percentile(ordered, 90)), 4), "max": round(max(ordered), 4)}

    similar_label_mismatches = []
    for first, second, distance in all_near_pairs:
        if first["counts"] != second["counts"]:
            similar_label_mismatches.append({"file_a": first["relative"], "file_b": second["relative"], "phash_distance": distance, "classes_a": {CLASSES[k]: v for k, v in first["counts"].items()}, "classes_b": {CLASSES[k]: v for k, v in second["counts"].items()}})

    return {
        "class_names": CLASSES,
        "splits": distribution,
        "issues": findings,
        "duplicate_images": {"exact_cross_split": [{"a": a["relative"], "b": b["relative"]} for a, b, _ in exact_duplicates], "near_cross_split_phash_le_5": [{"a": a["relative"], "b": b["relative"], "distance": d} for a, b, d in near_duplicates]},
        "capture_group_overlap": group_overlaps,
        "source_plot_groups_by_split": {split: sorted(groups) for split, groups in source_plot_sets.items()},
        "source_plot_group_overlap": source_plot_overlaps,
        "bounding_boxes_pixels_width": describe(box_widths), "bounding_boxes_pixels_height": describe(box_heights), "bounding_boxes_area_fraction": describe(box_areas),
        "annotation_csv": {"rows": csv_rows, "per_class": dict(csv_class_counts)},
        "similar_image_label_mismatches": similar_label_mismatches[:100],
        "preprocessing": {"matched_rgb_pairs": len(mask_stats), "hsv_lower": [25, 40, 30], "hsv_upper": [95, 255, 255], "retained_pixel_fraction": describe([x["retained_fraction"] for x in mask_stats]), "yellow_brown_like_pixel_retention": describe([x["yellow_brown_retained_fraction"] for x in mask_stats if x["yellow_brown_retained_fraction"] is not None]), "interpretation": "The notebook's green mask is HSV [25, 40, 30] to [95, 255, 255]. Yellow/brown color ranges include soil and cannot identify foliage alone. Review RGB images before relying on a green-only mask for maturity cues."},
    }


def markdown_report(report):
    lines = ["# Dataset audit", "", "Read-only audit of the existing YOLO dataset. The `test` split was not modified or used for model selection.", "", "## Split and class counts", "", "| Split | Images | Instances | Vegetative | Tuber Bulking | Maturity | Empty or unlabeled | Capture groups |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for split in ("train", "val", "test"):
        d = report["splits"][split]
        lines.append(f"| {split} | {d['images']} | {d['instances']} | {d['per_class']['Vegetative']} | {d['per_class']['Tuber Bulking']} | {d['per_class']['Maturity']} | {d['empty_or_unlabeled_images']} | {len(d['capture_groups'])} |")
    lines += ["", "## Integrity findings", ""]
    issue_names = [("missing_labels", "Images without label files"), ("missing_images", "Missing image references from annotations.csv"), ("orphan_labels", "Label files without images"), ("invalid_labels", "Invalid label rows"), ("unreadable_images", "Unreadable images")]
    for key, label in issue_names:
        lines.append(f"- **{label}:** {len(report['issues'][key])}")
        for row in report["issues"][key][:8]:
            lines.append(f"  - `{row}`")
    lines += ["", "## Split leakage and duplicate checks", "", f"- Capture-prefix overlap: `{json.dumps(report['capture_group_overlap'])}`", f"- Source plot groups by split: `{json.dumps(report['source_plot_groups_by_split'])}`", f"- Source plot-group overlap: `{json.dumps(report['source_plot_group_overlap'])}`", f"- Exact duplicate image pairs across splits: {len(report['duplicate_images']['exact_cross_split'])}", f"- Near-duplicate pairs across splits (pHash Hamming distance ≤ 5): {len(report['duplicate_images']['near_cross_split_phash_le_5'])}", f"- Similar-image class-count mismatches (within or across splits): {len(report['similar_image_label_mismatches'])}"]
    for pair in report["duplicate_images"]["near_cross_split_phash_le_5"][:12]:
        lines.append(f"  - near duplicate (distance {pair['distance']}): `{pair['a']}` ↔ `{pair['b']}`")
    for pair in report["similar_image_label_mismatches"][:8]:
        lines.append(f"  - possible label inconsistency: `{pair['file_a']}` ↔ `{pair['file_b']}`; {pair['classes_a']} vs {pair['classes_b']}")
    lines += ["", "## Bounding-box size", "", "Values are width/height in pixels and area as a fraction of image area."]
    for label, key in (("Width (px)", "bounding_boxes_pixels_width"), ("Height (px)", "bounding_boxes_pixels_height"), ("Area fraction", "bounding_boxes_area_fraction")):
        lines.append(f"- **{label}:** `{report[key]}`")
    lines += ["", "## RGB vs green-filtered preprocessing", "", f"- Matched full-resolution RGB/preprocessed pairs: {report['preprocessing']['matched_rgb_pairs']}", f"- Notebook mask bounds: HSV {report['preprocessing']['hsv_lower']}–{report['preprocessing']['hsv_upper']}", f"- Non-black pixel fraction in preprocessed images: `{report['preprocessing']['retained_pixel_fraction']}`", f"- Retention among yellow/brown-like pixels: `{report['preprocessing']['yellow_brown_like_pixel_retention']}`", "- Interpretation: the inspected preprocessed image keeps green plants while masking soil/background. The hue lower bound of 25 excludes many yellow colors and browning is also outside the green band, so maturity cues may be suppressed. The pixel statistic includes soil and is not foliage-specific. Keep RGB images available for a controlled comparison; do not replace current model inputs based on this audit alone.", "", "## Recommendations", "", "1. Preserve the existing test files and labels byte-for-byte. Capture prefixes are disjoint, but source `T1_T*` plot groups overlap across train/validation/test. Treat this as a field-group leakage risk and use plot-level grouping for any future split redesign.", "2. Maturity is underrepresented in the current training annotations. Add manually reviewed training examples from diverse maturity stages, including yellowing/browning foliage, while holding test and validation groups out.", "3. Inspect any invalid boxes and suspected similar-image label mismatches before excluding or correcting them. Do not automatically repair annotations from model predictions.", "4. Maintain both original RGB and masked views. Compare them in a later controlled experiment using the same split and evaluation protocol.", "5. The current dataset uses only a small number of capture groups; add independent field/date groups before interpreting test metrics as generalization across farms or seasons.", "", "## CSV cross-check", "", f"`annotations.csv` contains {report['annotation_csv']['rows']} rows; parsed class counts: `{json.dumps(report['annotation_csv']['per_class'])}`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--report", type=Path, default=None, help="Markdown output path (default: reports/dataset_audit.md)")
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    report_path = args.report or root / "reports" / "dataset_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(result), encoding="utf-8")
    (report_path.parent / "dataset_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Dataset audit written to {report_path}")
    print(json.dumps({"splits": result["splits"], "issues": {k: len(v) for k, v in result["issues"].items()}, "capture_group_overlap": result["capture_group_overlap"], "duplicates": result["duplicate_images"]}, indent=2))


if __name__ == "__main__":
    main()
