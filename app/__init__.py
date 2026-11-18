from pathlib import Path
import os

from flask import Flask


def create_app(test_config=None):
    project_root = Path(__file__).resolve().parent.parent
    instance = project_root / "instance"
    instance.mkdir(exist_ok=True)
    uploads = instance / "uploads"
    outputs = instance / "outputs"
    uploads.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    app = Flask(__name__, instance_path=str(instance), instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-demo-change-me"),
        MAX_CONTENT_LENGTH=97 * 1024 * 1024,
        MAX_IMAGES=12,
        MAX_IMAGE_BYTES=8 * 1024 * 1024,
        PROJECT_ROOT=project_root,
        UPLOAD_DIR=uploads,
        OUTPUT_DIR=outputs,
        PREANNOTATION_DIR=project_root / "preannotations",
        TILES_DIR=project_root / "tiles_640_final",
        MODEL_PATH=Path(os.environ.get("POTATO_MODEL_PATH", project_root / "runs/potato_yolov8n/weights/best.pt")),
    )
    if test_config:
        app.config.update(test_config)

    from .routes import bp

    app.register_blueprint(bp)
    return app
