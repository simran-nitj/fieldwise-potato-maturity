import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath

from flask import url_for

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_name(filename):
    cleaned = secure_filename(filename) or "image"
    return f"{uuid.uuid4().hex[:10]}_{cleaned}"


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="minutes")


def public_image_path(assessment_id, filename):
    relative = PurePosixPath(str(filename).replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Invalid assessment image path")
    return url_for("main.assessment_image", assessment_id=assessment_id, relative_name=relative.as_posix())
