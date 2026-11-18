import csv
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

# -------------------- SETTINGS --------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "tiles_640_final"
OUTPUT_CSV = BASE_DIR / "annotations.csv"
ANNOTATED_IMAGE_DIR = BASE_DIR / "annotated_images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {
    0: "Vegetative",
    1: "Tuber Bulking",
    2: "Maturity"
}

CLASS_COLORS = {
    0: "#00D26A",
    1: "#FFD43B",
    2: "#FF5C5C"
}

CSV_HEADERS = [
    "filename", "class_id", "class_name",
    "xmin", "ymin", "xmax", "ymax", "annotated_image"
]

CANVAS_WIDTH = 1050
CANVAS_HEIGHT = 680

# -------------------- FIND IMAGES --------------------

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

image_files = sorted(
    p for p in IMAGE_FOLDER.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_files:
    raise FileNotFoundError(f"No images found in: {IMAGE_FOLDER}")

ANNOTATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def relative_name(path):
    # Relative path prevents collisions when different folders contain
    # tiles with the same filename.
    return path.relative_to(IMAGE_FOLDER).as_posix()

# Create CSV if it does not exist.
if not OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADERS)

# A source image is considered completed only if it has at least one
# annotation row in the CSV. Skipped images can be revisited next run.
completed_images = set()
with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("filename"):
            completed_images.add(row["filename"])

remaining_images = [
    p for p in image_files if relative_name(p) not in completed_images
]

print("Total images:", len(image_files))
print("Already annotated:", len(completed_images))
print("Remaining:", len(remaining_images))

# -------------------- APPLICATION STATE --------------------

current_index = 0
current_image_path = None
original_image = None
tk_image = None

original_width = original_height = 0
display_width = display_height = 0
scale_x = scale_y = 1.0

start_x = start_y = None
current_rectangle = None
selected_class = None
pending_annotations = []

# -------------------- WINDOW --------------------

root = tk.Tk()
root.title("Potato Maturity Annotator")
root.geometry("1160x850")

canvas = tk.Canvas(
    root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
    bg="black", highlightthickness=0
)
canvas.pack(padx=10, pady=10)

status_label = tk.Label(root, text="", font=("Arial", 13, "bold"))
status_label.pack(pady=3)

class_label = tk.Label(
    root,
    text="Select class: 0 Vegetative | 1 Tuber Bulking | 2 Maturity",
    font=("Arial", 11)
)
class_label.pack(pady=3)

help_label = tk.Label(
    root,
    text="Draw box → choose class → A: Add | S: Save image | N: Skip | Q/Esc: Exit",
    font=("Arial", 10)
)
help_label.pack(pady=3)

# -------------------- LOAD IMAGE --------------------

def update_status():
    if current_image_path is not None:
        status_label.config(
            text=(
                f"Image {current_index + 1}/{len(remaining_images)}"
                f" | Saved: {len(completed_images)}"
                f" | Pending boxes: {len(pending_annotations)}"
                f" | {relative_name(current_image_path)}"
            )
        )


def load_image():
    global current_image_path, original_image, tk_image
    global original_width, original_height
    global display_width, display_height, scale_x, scale_y
    global current_rectangle, selected_class, pending_annotations

    canvas.delete("all")
    current_rectangle = None
    selected_class = None
    pending_annotations = []

    if current_index >= len(remaining_images):
        current_image_path = None
        status_label.config(text="All remaining images have been processed.")
        class_label.config(text="Close the window or press Q to finish.")
        messagebox.showinfo(
            "Completed",
            "All remaining images have been processed.\n"
            "Your CSV and annotated image copies are saved."
        )
        return

    current_image_path = remaining_images[current_index]
    original_image = Image.open(current_image_path).convert("RGB")
    original_width, original_height = original_image.size

    scale = min(
        CANVAS_WIDTH / original_width,
        CANVAS_HEIGHT / original_height,
        1.0
    )
    display_width = max(1, int(original_width * scale))
    display_height = max(1, int(original_height * scale))
    scale_x = original_width / display_width
    scale_y = original_height / display_height

    display_image = original_image.resize(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )
    tk_image = ImageTk.PhotoImage(display_image)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    update_status()
    class_label.config(text="Draw a box, select class 0/1/2, then press A to add it.")


# -------------------- CLASS SELECTION --------------------

def select_class(class_id):
    global selected_class
    selected_class = class_id
    class_label.config(
        text=f"Selected: {class_id} - {CLASS_NAMES[class_id]}"
    )


# -------------------- DRAW BOX --------------------

def mouse_down(event):
    global start_x, start_y, current_rectangle

    if current_image_path is None:
        return
    if not (0 <= event.x <= display_width and 0 <= event.y <= display_height):
        return

    start_x, start_y = event.x, event.y

    if current_rectangle is not None:
        canvas.delete(current_rectangle)

    current_rectangle = canvas.create_rectangle(
        start_x, start_y, start_x, start_y,
        outline="white", width=2
    )


def mouse_drag(event):
    if current_rectangle is None:
        return

    x = max(0, min(event.x, display_width))
    y = max(0, min(event.y, display_height))
    canvas.coords(current_rectangle, start_x, start_y, x, y)


# -------------------- ADD BOX --------------------

def add_box():
    global current_rectangle, selected_class

    if current_image_path is None:
        return
    if current_rectangle is None:
        messagebox.showwarning("No box", "Draw a bounding box first.")
        return
    if selected_class not in CLASS_NAMES:
        messagebox.showwarning("No class", "Select class 0, 1, or 2 first.")
        return

    x1, y1, x2, y2 = canvas.coords(current_rectangle)

    xmin = max(0, min(int(min(x1, x2) * scale_x), original_width))
    xmax = max(0, min(int(max(x1, x2) * scale_x), original_width))
    ymin = max(0, min(int(min(y1, y2) * scale_y), original_height))
    ymax = max(0, min(int(max(y1, y2) * scale_y), original_height))

    if xmax <= xmin or ymax <= ymin:
        messagebox.showwarning("Invalid box", "Draw a box with non-zero width and height.")
        return

    pending_annotations.append(
        (selected_class, xmin, ymin, xmax, ymax)
    )

    canvas.delete(current_rectangle)
    current_rectangle = None
    selected_class = None

    class_label.config(
        text=f"Added. {len(pending_annotations)} box(es) for this image. Draw another or save."
    )
    update_status()


def reset_current_box():
    global current_rectangle, selected_class

    if current_rectangle is not None:
        canvas.delete(current_rectangle)

    current_rectangle = None
    selected_class = None
    class_label.config(text="Unfinished box reset; added boxes are unchanged.")


# -------------------- SAVE IMAGE + CSV --------------------

def save_current_image():
    global current_index

    if current_image_path is None:
        return

    if not pending_annotations:
        messagebox.showwarning(
            "No annotations",
            "Add at least one box, or use Skip (N) for an image with no label."
        )
        return

    rel_name = relative_name(current_image_path)
    annotated_path = ANNOTATED_IMAGE_DIR / Path(rel_name)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw all saved boxes and class labels on a copy of the source image.
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for class_id, xmin, ymin, xmax, ymax in pending_annotations:
        color = CLASS_COLORS[class_id]
        label = CLASS_NAMES[class_id]
        line_width = max(3, original_width // 300)

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=color,
            width=line_width
        )

        label_y = max(0, ymin - 20)
        try:
            text_box = draw.textbbox((xmin, label_y), label)
            draw.rectangle(text_box, fill=color)
        except AttributeError:
            pass
        draw.text((xmin + 2, label_y), label, fill="black")

    annotated_image.save(annotated_path)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for class_id, xmin, ymin, xmax, ymax in pending_annotations:
            writer.writerow([
                rel_name, class_id, CLASS_NAMES[class_id],
                xmin, ymin, xmax, ymax,
                annotated_path.relative_to(BASE_DIR).as_posix()
            ])

    completed_images.add(rel_name)
    print(f"Saved {rel_name}: {len(pending_annotations)} box(es)")
    print("Annotated copy:", annotated_path)

    current_index += 1
    load_image()


# -------------------- SKIP IMAGE --------------------

def skip_image():
    global current_index

    if current_image_path is None:
        return

    if pending_annotations:
        answer = messagebox.askyesno(
            "Discard pending boxes?",
            "This image has unsaved boxes. Skip and discard them?"
        )
        if not answer:
            return

    current_index += 1
    load_image()


# -------------------- EXIT AT ANY TIME --------------------

def request_exit():
    if pending_annotations and current_image_path is not None:
        answer = messagebox.askyesnocancel(
            "Finish annotating?",
            "There are unsaved boxes on the current image.\n\n"
            "Yes = Save this image, then exit\n"
            "No = Exit and discard these pending boxes\n"
            "Cancel = Continue annotating"
        )

        if answer is None:
            return
        if answer is True:
            save_current_image()
            # save_current_image advances the index; now exit.
    root.destroy()


# -------------------- KEYBOARD SHORTCUTS --------------------

def key_pressed(event):
    key = event.keysym.lower()

    if key == "0":
        select_class(0)
    elif key == "1":
        select_class(1)
    elif key == "2":
        select_class(2)
    elif key == "a":
        add_box()
    elif key == "s":
        save_current_image()
    elif key == "n":
        skip_image()
    elif key == "r":
        reset_current_box()
    elif key in ("q", "escape"):
        request_exit()


# -------------------- BUTTONS --------------------

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(
    button_frame, text="0 - Vegetative", width=18,
    command=lambda: select_class(0)
).grid(row=0, column=0, padx=4, pady=4)

tk.Button(
    button_frame, text="1 - Tuber Bulking", width=18,
    command=lambda: select_class(1)
).grid(row=0, column=1, padx=4, pady=4)

tk.Button(
    button_frame, text="2 - Maturity", width=18,
    command=lambda: select_class(2)
).grid(row=0, column=2, padx=4, pady=4)

tk.Button(
    button_frame, text="ADD BOX (A)", width=16,
    command=add_box
).grid(row=1, column=0, padx=4, pady=4)

tk.Button(
    button_frame, text="SAVE IMAGE (S)", width=18,
    command=save_current_image
).grid(row=1, column=1, padx=4, pady=4)

tk.Button(
    button_frame, text="SKIP (N)", width=16,
    command=skip_image
).grid(row=1, column=2, padx=4, pady=4)

tk.Button(
    button_frame, text="RESET BOX (R)", width=16,
    command=reset_current_box
).grid(row=1, column=3, padx=4, pady=4)

tk.Button(
    button_frame, text="FINISH / EXIT (Q)", width=18,
    command=request_exit, bg="#8B1E1E", fg="white"
).grid(row=1, column=4, padx=4, pady=4)

canvas.bind("<ButtonPress-1>", mouse_down)
canvas.bind("<B1-Motion>", mouse_drag)
root.bind("<Key>", key_pressed)

# -------------------- START --------------------

root.protocol("WM_DELETE_WINDOW", request_exit)
load_image()
root.mainloop()

print("\nAnnotator closed.")
print("CSV:", OUTPUT_CSV)
print("Annotated images:", ANNOTATED_IMAGE_DIR)
print("Images with saved annotations:", len(completed_images))
