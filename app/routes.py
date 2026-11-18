import base64
import json
import re
import shutil
import uuid
from datetime import date
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from .harvest_estimator import STAGE_NAMES, estimate_harvest
from .predictor import Predictor
from .review import ReviewDataError, ReviewStore
from .utils import allowed_image, now_iso, public_image_path, safe_name

bp = Blueprint("main", __name__)
_predictor = None
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def predictor():
    global _predictor
    model_path = Path(current_app.config["MODEL_PATH"]).resolve()
    if _predictor is None or _predictor.model_path != model_path:
        _predictor = Predictor(model_path)
    return _predictor


def assessment_file(assessment_id):
    if not re.fullmatch(r"[a-f0-9]{32}", assessment_id or ""):
        abort(404)
    return Path(current_app.config["OUTPUT_DIR"]) / assessment_id / "assessment.json"


def load_assessment(assessment_id):
    path = assessment_file(assessment_id)
    if not path.is_file():
        abort(404)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current_app.logger.exception("Assessment record could not be read")
        abort(404)
    if data.get("id") != assessment_id or not isinstance(data.get("images"), list):
        abort(404)
    # Older MVP records stored only URL basenames; recover files from their
    # established per-assessment folders without rewriting the saved record.
    for image in data["images"]:
        for kind, folder, file_key in (("original", "originals", "original_file"), ("annotated", "annotated", "annotated_file")):
            if image.get(file_key):
                continue
            old_name = PurePosixPath(str(image.get(kind, "")).replace("\\", "/")).name
            candidate = Path(current_app.config["OUTPUT_DIR"]) / assessment_id / folder / old_name
            if old_name and candidate.is_file():
                image[file_key] = f"{folder}/{old_name}"
                image[kind] = public_image_path(assessment_id, image[file_key])
    return data


def save_assessment(data):
    path = assessment_file(data["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def all_assessments():
    output = Path(current_app.config["OUTPUT_DIR"])
    records = []
    if not output.exists():
        return records
    for metadata in output.glob("*/assessment.json"):
        try:
            row = json.loads(metadata.read_text(encoding="utf-8"))
            if row.get("id") == metadata.parent.name and isinstance(row.get("images"), list):
                records.append(row)
        except (OSError, json.JSONDecodeError):
            current_app.logger.warning("Skipping unreadable assessment record %s", metadata.name)
    return sorted(records, key=lambda row: row.get("created_at", ""), reverse=True)


def assessment_summaries():
    records = all_assessments()
    totals = {name: 0 for name in STAGE_NAMES}
    for row in records:
        for name in STAGE_NAMES:
            try:
                totals[name] += max(0, int(row.get("stage_counts", {}).get(name, 0)))
            except (TypeError, ValueError):
                continue
    images = sum(len(row.get("images", [])) for row in records)
    detections = sum(totals.values())
    return {
        "records": records,
        "assessment_count": len(records),
        "image_count": images,
        "detection_count": detections,
        "stage_counts": totals,
        "distribution": {name: round(totals[name] / detections * 100, 1) if detections else 0 for name in STAGE_NAMES},
    }


def review_store():
    cache = current_app.extensions.setdefault("fieldwise_review_stores", {})
    key = (str(Path(current_app.config["PROJECT_ROOT"]).resolve()), str(Path(current_app.config["PREANNOTATION_DIR"]).resolve()))
    if key not in cache:
        cache[key] = ReviewStore(Path(current_app.config["PROJECT_ROOT"]), Path(current_app.config["PREANNOTATION_DIR"]))
    return cache[key]


def available_demo_images():
    root = Path(current_app.config["PROJECT_ROOT"]) / "potato_yolo_dataset" / "images" / "test"
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] if root.exists() else []
    return sorted(images)[:3]


def validate_assessment_fields(form):
    field_name = (form.get("field_name") or "").strip()
    planting_date = (form.get("planting_date") or "").strip()
    duration = (form.get("assumed_duration_days") or "").strip()
    if not planting_date:
        return None, "Enter the planting date to save this field assessment."
    try:
        date.fromisoformat(planting_date)
    except ValueError:
        return None, "Enter a valid planting date."
    if duration:
        try:
            days = int(duration)
        except ValueError:
            return None, "Assumed crop duration must be a whole number from 60 to 180 days."
        if not 60 <= days <= 180:
            return None, "Assumed crop duration must be between 60 and 180 days."
        duration = str(days)
    return {
        "field_name": field_name[:100],
        "planting_date": planting_date,
        "variety": (form.get("variety") or "").strip()[:100],
        "region": (form.get("region") or "").strip()[:100],
        "notes": (form.get("notes") or "").strip()[:1000],
        "assumed_duration_days": duration,
    }, None


def decode_upload(uploaded):
    if not uploaded or not uploaded.filename:
        raise ValueError("Choose at least one JPG or PNG image.")
    if not allowed_image(uploaded.filename):
        raise ValueError(f"Unsupported image type for {uploaded.filename}. Use JPG, JPEG, or PNG.")
    uploaded.stream.seek(0, 2)
    size = uploaded.stream.tell()
    uploaded.stream.seek(0)
    if size > current_app.config["MAX_IMAGE_BYTES"]:
        raise ValueError(f"{uploaded.filename} exceeds the {current_app.config['MAX_IMAGE_BYTES'] // (1024 * 1024)} MB per-image limit.")
    payload = uploaded.read()
    uploaded.stream.seek(0)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"Could not read {uploaded.filename}. Choose a valid JPG or PNG image.")
    return payload


def create_assessment(items, details, demo=False):
    model = predictor()
    if not model.ready:
        raise RuntimeError("The trained model is unavailable. Check the checkpoint and local ML environment.")
    assessment_id = uuid.uuid4().hex
    directory = Path(current_app.config["OUTPUT_DIR"]) / assessment_id
    original_dir, annotated_dir = directory / "originals", directory / "annotated"
    original_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "id": assessment_id,
        "created_at": now_iso(),
        **details,
        "images": [],
        "stage_counts": {name: 0 for name in STAGE_NAMES},
        "total_detections": 0,
        "average_confidence": None,
        "model_device": "CUDA" if model.device == "0" else "CPU",
        "model_checkpoint": Path(model.model_path).name,
        "demo": bool(demo),
    }
    confidences = []
    try:
        for source, display_name, payload in items:
            stored_name = safe_name(display_name)
            original_path = original_dir / stored_name
            if payload is None:
                shutil.copyfile(source, original_path)
            else:
                original_path.write_bytes(payload)
            annotated_name = f"{Path(stored_name).stem}.jpg"
            annotated_path = annotated_dir / annotated_name
            result = model.predict(original_path, annotated_path)
            for name in STAGE_NAMES:
                record["stage_counts"][name] += result["counts"][name]
            record["total_detections"] += result["total"]
            confidences.extend(item["confidence"] for item in result["detections"])
            original_relative = f"originals/{stored_name}"
            annotated_relative = f"annotated/{annotated_name}"
            record["images"].append({
                "name": display_name,
                "original_file": original_relative,
                "annotated_file": annotated_relative,
                "original": public_image_path(assessment_id, original_relative),
                "annotated": public_image_path(assessment_id, annotated_relative),
                "total": result["total"],
                "counts": result["counts"],
                "average_confidence": result["average_confidence"],
                "detections": result["detections"],
            })
        total = record["total_detections"]
        record["distribution"] = {name: round(count / total * 100, 1) if total else 0 for name, count in record["stage_counts"].items()}
        record["average_confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else None
        record["estimation"] = estimate_harvest(record["planting_date"], record["stage_counts"], record["assumed_duration_days"])
        save_assessment(record)
    except Exception:
        current_app.logger.exception("Assessment %s failed during inference", assessment_id)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return record


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return render_template("error.html", title="Upload too large", message="The request exceeds the 97 MB upload limit. Choose fewer or smaller images."), 413


@bp.app_errorhandler(404)
def not_found(_error):
    return render_template("error.html", title="Page not found", message="This page or assessment is no longer available."), 404


@bp.get("/")
def home():
    model = predictor()
    summary = assessment_summaries()
    sensor = historical_sensor_summary()
    return render_template("index.html", model_ready=model.ready, model_error=model.error, model_device="CUDA" if model.device == "0" else "CPU", demos=available_demo_images(), history=summary["records"][:6], summary=summary, sensor=sensor)


@bp.get("/history")
def history():
    summary = assessment_summaries()
    query = (request.args.get("q") or "").strip()[:100]
    rows = summary["records"]
    if query:
        needle = query.casefold()
        rows = [row for row in rows if any(needle in str(row.get(key, "")).casefold() for key in ("field_name", "region", "variety", "created_at", "id"))]
    return render_template("history.html", rows=rows, query=query, total=summary["assessment_count"])


@bp.get("/about")
def about():
    return render_template("about.html")


@bp.get("/review")
def review_dashboard():
    store = review_store()
    try:
        counts = store.dashboard()
        rows = store.list_items("pending")
    except ReviewDataError as exc:
        return render_template("error.html", title="Review data unavailable", message=str(exc)), 503
    initial_item = rows[0]["image_path"] if rows else ""
    return render_template("review.html", counts=counts, initial_item=initial_item)


@bp.get("/review/api/items")
def review_items_api():
    store = review_store()
    try:
        status = (request.args.get("status") or "pending").strip().lower()
        query = (request.args.get("q") or "").strip()[:200]
        items = store.list_items(status, query)
        return jsonify({"items": items, "counts": store.dashboard()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ReviewDataError as exc:
        return jsonify({"error": str(exc)}), 503


@bp.get("/review/api/item")
def review_item_api():
    image_id = request.args.get("image_path", "")
    try:
        item = review_store().get_item(image_id)
        item["original_url"] = url_for("main.review_image", image_path=item["image_path"])
        return jsonify(item)
    except KeyError as exc:
        abort(404, description=str(exc))
    except ReviewDataError as exc:
        return jsonify({"error": str(exc)}), 503


@bp.get("/review/image")
def review_image():
    image_id = request.args.get("image_path", "")
    try:
        item = review_store().get_item(image_id)
    except KeyError:
        abort(404)
    except ReviewDataError as exc:
        return jsonify({"error": str(exc)}), 503
    image_path = Path(current_app.config["TILES_DIR"]).resolve().joinpath(*PurePosixPath(item["image_path"]).parts)
    return send_from_directory(Path(current_app.config["TILES_DIR"]), image_path.relative_to(Path(current_app.config["TILES_DIR"]).resolve()).as_posix(), conditional=True)


@bp.post("/review/api/action")
def review_action_api():
    image_id = request.args.get("image_path", "")
    payload = request.get_json(silent=True) or {}
    try:
        result = review_store().update(
            image_id=image_id,
            action=str(payload.get("action", "")),
            reviewer=str(payload.get("reviewer", "")),
            boxes=payload.get("boxes"),
        )
        return jsonify({"item": result, "counts": review_store().dashboard()})
    except KeyError as exc:
        abort(404, description=str(exc))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ReviewDataError as exc:
        return jsonify({"error": str(exc)}), 503


@bp.post("/assessment")
def create_assessment_route():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        flash("Choose at least one JPG or PNG image.", "error")
        return redirect(url_for("main.home") + "#assessment")
    if len(files) > current_app.config["MAX_IMAGES"]:
        flash(f"Choose no more than {current_app.config['MAX_IMAGES']} images per assessment.", "error")
        return redirect(url_for("main.home") + "#assessment")
    details, field_error = validate_assessment_fields(request.form)
    if field_error:
        flash(field_error, "error")
        return redirect(url_for("main.home") + "#assessment")
    try:
        items = [(None, f.filename, decode_upload(f)) for f in files]
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.home") + "#assessment")
    try:
        record = create_assessment(items, details)
    except Exception:
        flash("The assessment could not be completed. Your images were not saved as a finished assessment; check model availability and try again.", "error")
        return redirect(url_for("main.home") + "#assessment")
    return redirect(url_for("main.results", assessment_id=record["id"]))


@bp.post("/demo")
def demo_assessment():
    demos = available_demo_images()
    if not demos:
        flash("No demo images were found in the project's YOLO test split.", "error")
        return redirect(url_for("main.home") + "#assessment")
    details = {"field_name": "Project demo samples", "planting_date": "", "variety": "", "region": "", "notes": "Existing images from the YOLO test split; demonstration inputs, not current field or live sensor data.", "assumed_duration_days": ""}
    try:
        record = create_assessment([(path, path.name, None) for path in demos], details, demo=True)
    except Exception:
        current_app.logger.exception("Project demo assessment failed")
        flash("The project demo could not be completed. Check model availability and try again.", "error")
        return redirect(url_for("main.home") + "#assessment")
    return redirect(url_for("main.results", assessment_id=record["id"]))


@bp.get("/results/<assessment_id>")
def results(assessment_id):
    return render_template("results.html", assessment=load_assessment(assessment_id))


@bp.get("/assessment-image/<assessment_id>/<path:relative_name>")
def assessment_image(assessment_id, relative_name):
    assessment = load_assessment(assessment_id)
    requested = PurePosixPath(relative_name.replace("\\", "/"))
    if requested.is_absolute() or any(part in {"", ".", ".."} for part in requested.parts):
        abort(404)
    allowed = {image.get(key) for image in assessment["images"] for key in ("original_file", "annotated_file")}
    normalized = requested.as_posix()
    if normalized not in allowed:
        abort(404)
    root = Path(current_app.config["OUTPUT_DIR"]) / assessment_id
    target = root.joinpath(*requested.parts)
    if not target.is_file() or root.resolve() not in target.resolve().parents:
        abort(404)
    return send_from_directory(root, normalized, conditional=True)


def report_images(assessment, embedded=False):
    root = Path(current_app.config["OUTPUT_DIR"]) / assessment["id"]
    output = []
    for image in assessment["images"]:
        row = dict(image)
        if embedded:
            for kind, key in (("original", "original_file"), ("annotated", "annotated_file")):
                path = root / Path(row[key])
                data = cv2.imread(str(path)) if path.is_file() else None
                if data is not None:
                    height, width = data.shape[:2]
                    scale = min(1.0, 1400 / max(height, width))
                    if scale < 1:
                        data = cv2.resize(data, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode(".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 84])
                    row[f"{kind}_report_src"] = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}" if ok else None
                else:
                    row[f"{kind}_report_src"] = None
        else:
            row["original_report_src"] = row["original"]
            row["annotated_report_src"] = row["annotated"]
        output.append(row)
    return output


@bp.get("/report/<assessment_id>")
def report(assessment_id):
    assessment = load_assessment(assessment_id)
    return render_template("report.html", assessment=assessment, report_images=report_images(assessment, embedded=True), printable=False, standalone=True), {"Content-Disposition": f"attachment; filename=fieldwise-assessment-{assessment_id[:8]}.html"}


@bp.get("/report/<assessment_id>/print")
def print_report(assessment_id):
    assessment = load_assessment(assessment_id)
    return render_template("report.html", assessment=assessment, report_images=report_images(assessment), printable=True, standalone=False)


def historical_sensor_summary():
    path = Path(current_app.config["PROJECT_ROOT"]) / "dataset" / "Agri-IoT_Data2023-25.xlsx"
    if not path.is_file():
        return None
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook["Year24-25"] if "Year24-25" in workbook.sheetnames else workbook.worksheets[-1]
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, ())
            numeric = {str(name): [] for name in headers[1:]}
            dates, row_count = [], 0
            for row in rows:
                if not row or row[0] is None:
                    continue
                row_count += 1
                dates.append(str(row[0]))
                for name, value in zip(headers[1:], row[1:]):
                    if isinstance(value, (int, float)):
                        numeric[str(name)].append(float(value))
            compact = []
            for name, values in list(numeric.items())[:4]:
                if values:
                    label = name.split(":")[0].replace("Year 2024-25", "").strip()
                    compact.append({"name": label, "average": round(sum(values) / len(values), 1)})
            return {"sheet": sheet.title, "row_count": row_count, "date_range": f"{dates[0]} – {dates[-1]}" if dates else "", "averages": compact}
        finally:
            workbook.close()
    except Exception:
        current_app.logger.exception("Historical sensor workbook could not be summarized")
        return {"sheet": "Workbook available", "row_count": None, "date_range": "", "averages": []}
