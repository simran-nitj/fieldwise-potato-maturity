"""Read-only audit of original plot groups and the current reviewed dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
RGB_ROOT = ROOT / "dataset" / "RGB images"
VERIFIED = ROOT / "potato_yolo_dataset"
TILES = ROOT / "tiles_640_final"
PRE = ROOT / "preannotations"
APPROVED_LABELS = PRE / "approved_labels"
REVIEW_CSV = PRE / "manual_approval.csv"
OUTPUT_JSON = ROOT / "reports" / "group_split_audit.json"
OUTPUT_MD = ROOT / "reports" / "group_split_audit.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAPTURE_RE = re.compile(r"IMG[0-9]{14}", re.IGNORECASE)
CLASSES = {0: "Vegetative", 1: "Tuber Bulking", 2: "Maturity"}
TRAINABLE = {"approved", "edited", "approved_empty"}


def capture_id(path: Path | str) -> str | None:
    match = CAPTURE_RE.search(Path(path).stem)
    return match.group(0).upper() if match else None


def parse_label(path: Path) -> list[int]:
    classes: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        values = [float(value) for value in fields]
        class_float, cx, cy, width, height = values
        if not class_float.is_integer() or int(class_float) not in CLASSES:
            raise ValueError(f"{path}:{line_number}: invalid class")
        if not all(0 <= value <= 1 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: invalid normalized box")
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box outside image")
        classes.append(int(class_float))
    return classes


def new_group_record() -> dict:
    return {
        "original_parent_images": 0,
        "verified": {s: {"images": 0, "boxes": 0, "classes": Counter()} for s in ("train", "val", "test")},
        "reviewed_pseudo": {"images": 0, "boxes": 0, "classes": Counter(), "statuses": Counter()},
        "parent_ids": [],
    }


def main() -> None:
    parents: dict[str, dict] = {}
    groups_by_capture: dict[str, set[str]] = defaultdict(set)
    for image in sorted(p for p in RGB_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        cid = capture_id(image)
        if cid is None:
            continue
        # The directory immediately containing the original photograph is the source plot ID.
        plot = image.parent.name
        groups_by_capture[cid].add(plot)
        parents.setdefault(cid, {"capture_id": cid, "plot_group": plot, "path": str(image.relative_to(ROOT)), "source_file_count": 0})
        parents[cid]["source_file_count"] += 1
        if parents[cid]["plot_group"] != plot:
            parents[cid].setdefault("plot_group_conflicts", []).append(plot)

    groups = {plot: new_group_record() for values in groups_by_capture.values() for plot in values}
    for cid, info in parents.items():
        for plot in groups_by_capture[cid]:
            groups[plot]["original_parent_images"] += 1
            groups[plot]["parent_ids"].append(cid)

    verified_seen: dict[str, list[str]] = defaultdict(list)
    verified_image_info: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        image_root = VERIFIED / "images" / split
        label_root = VERIFIED / "labels" / split
        for image in sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            cid = capture_id(image)
            label = label_root / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing verified label for {image}")
            image_data = cv2.imread(str(image))
            if image_data is None:
                raise ValueError(f"Unreadable verified image: {image}")
            cls_ids = parse_label(label)
            if cid is None or cid not in groups_by_capture:
                group = "UNKNOWN"
            else:
                mapped = groups_by_capture[cid]
                group = next(iter(mapped)) if len(mapped) == 1 else "CONFLICT"
            record = groups.setdefault(group, new_group_record())
            record["verified"][split]["images"] += 1
            record["verified"][split]["boxes"] += len(cls_ids)
            record["verified"][split]["classes"].update(CLASSES[c] for c in cls_ids)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            verified_seen[digest].append(f"{split}/{image.name}")
            verified_image_info[f"{split}/{image.name}"] = {"plot_group": group, "capture_id": cid, "boxes": len(cls_ids), "classes": Counter(CLASSES[c] for c in cls_ids)}

    review_rows = list(csv.DictReader(REVIEW_CSV.open(encoding="utf-8-sig", newline="")))
    review_paths = [row["image_path"] for row in review_rows]
    if len(review_paths) != len(set(review_paths)):
        raise ValueError("Review CSV contains duplicate image paths")
    reviewed_records = []
    review_seen: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    for row in review_rows:
        status = row["review_status"].strip().lower()
        if status not in TRAINABLE:
            continue
        rel = Path(row["image_path"])
        cid = capture_id(rel)
        plot_set = groups_by_capture.get(cid or "", set())
        plot = next(iter(plot_set)) if len(plot_set) == 1 else ("CONFLICT" if plot_set else "UNKNOWN")
        source_image = TILES / rel
        label = (APPROVED_LABELS / rel).with_suffix(".txt")
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            errors.append(f"{rel}: missing reviewer or timestamp")
        if not source_image.is_file() or not label.is_file():
            errors.append(f"{rel}: missing source image or approved label")
            cls_ids: list[int] = []
        else:
            if cv2.imread(str(source_image)) is None:
                errors.append(f"{rel}: unreadable source image")
            cls_ids = parse_label(label)
            if status in {"approved", "edited"} and not cls_ids:
                errors.append(f"{rel}: {status} label is empty")
            if status == "approved_empty" and cls_ids:
                errors.append(f"{rel}: approved_empty label contains boxes")
            digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
            review_seen[digest].append(rel.as_posix())
            if digest in verified_seen:
                errors.append(f"{rel}: duplicate image content with verified {verified_seen[digest]}")
        record = groups.setdefault(plot, new_group_record())
        record["reviewed_pseudo"]["images"] += 1
        record["reviewed_pseudo"]["boxes"] += len(cls_ids)
        record["reviewed_pseudo"]["classes"].update(CLASSES[c] for c in cls_ids)
        record["reviewed_pseudo"]["statuses"][status] += 1
        reviewed_records.append({"image_path": rel.as_posix(), "capture_id": cid, "plot_group": plot, "status": status, "boxes": len(cls_ids), "classes": Counter(CLASSES[c] for c in cls_ids)})

    for digest, paths in review_seen.items():
        if len(paths) > 1:
            errors.append(f"duplicate reviewed image content: {paths}")

    plot_membership = {}
    for split in ("train", "val", "test"):
        plot_membership[split] = sorted(group for group, rec in groups.items() if rec["verified"][split]["images"])
    plot_membership["reviewed_pseudo"] = sorted(group for group, rec in groups.items() if rec["reviewed_pseudo"]["images"])
    overlaps = {
        f"{a}-{b}": sorted(set(plot_membership[a]) & set(plot_membership[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    test_groups = set(plot_membership["test"])
    non_test_groups = set(groups) - test_groups
    proposed = {
        "train": sorted(non_test_groups - set(plot_membership["val"])),
        "validation": sorted((non_test_groups & set(plot_membership["val"])) - set(plot_membership["train"])),
        "test": sorted(test_groups),
        "valid_complete_split": False,
        "reason": "Existing test groups are T1_T2 and T1_T3. Of the three known plots, only T1_T1 remains outside test, so there is no independent plot group for both training and validation. Existing validation is T1_T3 and overlaps test.",
    }

    def serial_group(data: dict) -> dict:
        return {
            "original_parent_images": data["original_parent_images"],
            "parent_ids": sorted(data["parent_ids"]),
            "verified": {s: {"images": d["images"], "boxes": d["boxes"], "classes": dict(d["classes"])} for s, d in data["verified"].items()},
            "reviewed_pseudo": {"images": data["reviewed_pseudo"]["images"], "boxes": data["reviewed_pseudo"]["boxes"], "classes": dict(data["reviewed_pseudo"]["classes"]), "statuses": dict(data["reviewed_pseudo"]["statuses"])},
        }

    parent_tile_counts = defaultdict(lambda: {"plot_group": None, "verified": Counter(), "reviewed_pseudo": Counter()})
    for cid in parents:
        parent_tile_counts[cid]["plot_group"] = next(iter(groups_by_capture[cid])) if len(groups_by_capture[cid]) == 1 else "CONFLICT"
    for key, info in verified_image_info.items():
        cid = info["capture_id"] or "UNKNOWN"
        parent_tile_counts[cid]["plot_group"] = info["plot_group"]
        parent_tile_counts[cid]["verified"][key.split("/")[0]] += 1
    for item in reviewed_records:
        cid = item["capture_id"] or "UNKNOWN"
        parent_tile_counts[cid]["plot_group"] = item["plot_group"]
        parent_tile_counts[cid]["reviewed_pseudo"][item["status"]] += 1
    parent_split_spans = {
        cid: sorted(split for split, count in data["verified"].items() if count)
        for cid, data in parent_tile_counts.items()
        if sum(1 for count in data["verified"].values() if count) > 1
    }
    unmapped_parent_ids = sorted(cid for cid in parents if len(groups_by_capture[cid]) != 1)

    statuses = Counter(row["review_status"].strip().lower() for row in review_rows)
    candidate_train_classes = Counter()
    candidate_train_images = 0
    candidate_train_boxes = 0
    for group in proposed["train"]:
        data = groups[group]
        candidate_train_images += data["verified"]["train"]["images"] + data["reviewed_pseudo"]["images"]
        candidate_train_boxes += data["verified"]["train"]["boxes"] + data["reviewed_pseudo"]["boxes"]
        candidate_train_classes.update(data["verified"]["train"]["classes"])
        candidate_train_classes.update(data["reviewed_pseudo"]["classes"])
    result = {
        "scope": "All original RGB parent photos, all verified dataset images, and all human-reviewed trainable pseudo-label records.",
        "counts": {"original_parent_images": len(parents), "unique_plot_groups": len({group for values in groups_by_capture.values() for group in values}), "verified_images": sum(1 for _ in verified_image_info), "review_manifest_rows": len(review_rows), "review_statuses": dict(statuses), "trainable_reviewed_pseudo_images": len(reviewed_records), "trainable_reviewed_pseudo_boxes": sum(item["boxes"] for item in reviewed_records), "missing_or_invalid_records": len(errors)},
        "plot_groups": {group: serial_group(data) for group, data in sorted(groups.items())},
        "original_parents": [{**parents[cid], "plot_group": next(iter(groups_by_capture[cid])) if len(groups_by_capture[cid]) == 1 else "CONFLICT"} for cid in sorted(parents)],
        "sample_parent_counts": {cid: {"plot_group": d["plot_group"], "verified_tiles_by_existing_split": dict(d["verified"]), "reviewed_pseudo_tiles_by_status": dict(d["reviewed_pseudo"])} for cid, d in sorted(parent_tile_counts.items())},
        "parent_images_spanning_verified_splits": parent_split_spans,
        "original_parents_without_unique_plot_metadata": unmapped_parent_ids,
        "existing_plot_groups_by_split": plot_membership,
        "existing_plot_group_overlaps": overlaps,
        "proposed_group_assignment": proposed,
        "validation_class_distribution": {"existing_train": dict(Counter(k for d in groups.values() for k, n in d["verified"]["train"]["classes"].items() for _ in range(n))), "existing_val": dict(Counter(k for d in groups.values() for k, n in d["verified"]["val"]["classes"].items() for _ in range(n))), "existing_test": dict(Counter(k for d in groups.values() for k, n in d["verified"]["test"]["classes"].items() for _ in range(n)))},
        "issues": errors,
        "limitations": ["Only three independent source plot IDs are present.", "Existing validation is drawn from T1_T3, which is also represented in test.", "Existing test groups T1_T2 and T1_T3 consume two of three plots; the remaining T1_T1 cannot provide both training and independent validation.", "Reviewed pseudo-labels are reviewer-screened suggestions, not equivalent to independently verified ground truth."],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Dataset Group and Split Audit",
        "",
        "Audit is read-only with respect to images, labels, review CSV, model weights, and existing dataset assignments.",
        "",
        "## Summary",
        "",
        f"- Original parent photographs: **{len(parents)}**",
        f"- Unique source plot groups: **{len({group for values in groups_by_capture.values() for group in values})}**",
        f"- Verified dataset images: **{len(verified_image_info)}** (299 expected)",
        f"- Review manifest records: **{len(review_rows)}**; trainable reviewed records: **{len(reviewed_records)}**",
        f"- Reviewed pseudo boxes: **{sum(item['boxes'] for item in reviewed_records)}**",
        f"- Review statuses: `{dict(statuses)}`",
        f"- Missing/invalid reviewed records: **{len(errors)}**",
        "",
        "## Plot group totals",
        "",
        "Counts include unique original parent photographs and image/box counts for each current verified split plus reviewed pseudo-label statuses.",
        "",
        "| Plot group | Parent photos | Verified train images / boxes | Verified val images / boxes | Verified test images / boxes | Reviewed pseudo images / boxes | Classes in reviewed pseudo |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for group, data in sorted(groups.items()):
        ver = data["verified"]
        pseudo = data["reviewed_pseudo"]
        lines.append(f"| {group} | {data['original_parent_images']} | {ver['train']['images']} / {ver['train']['boxes']} | {ver['val']['images']} / {ver['val']['boxes']} | {ver['test']['images']} / {ver['test']['boxes']} | {pseudo['images']} / {pseudo['boxes']} | {dict(pseudo['classes'])} |")
    lines += ["", "## Existing plot assignment and overlap", "", f"- Train: `{plot_membership['train']}`", f"- Validation: `{plot_membership['val']}`", f"- Test: `{plot_membership['test']}`", f"- Train/validation overlap: `{overlaps['train-val']}`", f"- Train/test overlap: `{overlaps['train-test']}`", f"- Validation/test overlap: `{overlaps['val-test']}`", "", "## Proposed strict assignment", "", f"- Candidate training groups outside current test: `{proposed['train']}`", f"- Independent validation groups outside current test: `{proposed['validation']}`", f"- Preserve existing test groups and samples: `{proposed['test']}`", "- **A complete group-disjoint three-way split is not possible with the available groups.** The two existing test groups must remain untouched. Only T1_T1 remains outside test; current validation is T1_T3, a test group. Do not train or tune on that assignment.", "", "### Candidate split class distribution (not prepared)", "", "The only group-disjoint training candidate is T1_T1. There is no eligible validation group. Existing test images and labels remain untouched.", "", "| Candidate split | Plot groups | Images | Boxes | Vegetative | Tuber Bulking | Maturity |", "|---|---|---:|---:|---:|---:|---:|"]
    test_image_count = sum(data["verified"]["test"]["images"] for data in groups.values())
    test_box_count = sum(data["verified"]["test"]["boxes"] for data in groups.values())
    test_classes = Counter()
    for data in groups.values():
        test_classes.update(data["verified"]["test"]["classes"])
    lines.append(f"| Candidate train | {', '.join(proposed['train'])} | {candidate_train_images} | {candidate_train_boxes} | {candidate_train_classes.get('Vegetative', 0)} | {candidate_train_classes.get('Tuber Bulking', 0)} | {candidate_train_classes.get('Maturity', 0)} |")
    lines.append("| Validation | none available | 0 | 0 | 0 | 0 | 0 |")
    lines.append(f"| Preserved existing test | {', '.join(proposed['test'])} | {test_image_count} | {test_box_count} | {test_classes.get('Vegetative', 0)} | {test_classes.get('Tuber Bulking', 0)} | {test_classes.get('Maturity', 0)} |")
    lines += ["", "### Class distributions in existing verified splits", "", "| Existing split | Vegetative boxes | Tuber Bulking boxes | Maturity boxes |", "|---|---:|---:|---:|"]
    for split, values in result["validation_class_distribution"].items():
        lines.append(f"| {split} | {values.get('Vegetative', 0)} | {values.get('Tuber Bulking', 0)} | {values.get('Maturity', 0)} |")
    lines += ["", "## Parent photograph mapping and tile counts", "", "Each row maps one original parent capture ID to its source plot folder. Tile counts list existing verified split membership and reviewed pseudo-label statuses; they are not randomly reassigned.", "", "| Parent capture ID | Plot group | Verified train | Verified val | Verified test | Reviewed approved | Reviewed edited | Reviewed approved-empty |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cid, data in sorted(parent_tile_counts.items()):
        reviewed = data["reviewed_pseudo"]
        lines.append(f"| {cid} | {data['plot_group']} | {data['verified'].get('train', 0)} | {data['verified'].get('val', 0)} | {data['verified'].get('test', 0)} | {reviewed.get('approved', 0)} | {reviewed.get('edited', 0)} | {reviewed.get('approved_empty', 0)} |")
    lines += ["", "## Missing metadata and validation", "", f"Original parent images without a unique group mapping: **{len(unmapped_parent_ids)}**.", f"Original parent images with verified tiles spanning multiple existing splits: **{len(parent_split_spans)}**.", f"Reviewed-label/data issues: **{len(errors)}**."]
    if parent_split_spans:
        lines += ["", "| Parent capture ID | Existing verified splits |", "|---|---|", *[f"| {cid} | {', '.join(splits)} |" for cid, splits in sorted(parent_split_spans.items())]]
    if errors:
        lines += ["", "| Issue |", "|---|", *[f"| `{error}` |" for error in errors]]
    lines += ["", "## Recommendation", "", "Collect additional independently labeled field plots, with parent-image IDs and plot/group IDs recorded at source. Preserve all images from T1_T2 and T1_T3 in the untouched test group set. At minimum, retain T1_T1 for training and obtain a separate independent plot for validation; more plots per split are preferable for stable field-level estimates. Assign groups before tiling, and keep every tile from a parent image in that parent's plot split.", "", f"Machine-readable counts and parent mapping: `{OUTPUT_JSON.relative_to(ROOT).as_posix()}`.", ""]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    print(json.dumps(result["counts"], indent=2))
    print("plot_groups_by_split", json.dumps(plot_membership))
    print("overlaps", json.dumps(overlaps))
    if errors:
        raise SystemExit("Audit found invalid or missing reviewed records; inspect report.")


if __name__ == "__main__":
    main()
