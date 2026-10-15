from pathlib import Path
import csv
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    ROOT
    / "data"
    / "video"
    / "ravdess"
    / "selected"
)

QA_DIR = (
    ROOT
    / "data"
    / "video"
    / "ravdess"
    / "qa"
)

OUT_MANIFEST = (
    ROOT
    / "experiments"
    / "manifests"
    / "ravdess_fixed_face_crops.csv"
)

QA_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

cascade_path = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(
    str(cascade_path)
)

if detector.empty():
    raise RuntimeError(
        f"Could not load cascade: {cascade_path}"
    )


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index)
    )

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not decode frame {frame_index}: "
            f"{video_path.name}"
        )

    return frame


def detect_largest_face(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
    )

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda b: int(b[2]) * int(b[3])
    )


def make_square_crop(
    face_box,
    frame_width,
    frame_height,
    scale=1.7,
):
    x, y, w, h = [
        float(v)
        for v in face_box
    ]

    cx = x + w / 2
    cy = y + h / 2

    side = int(
        math.ceil(
            max(w, h) * scale
        )
    )

    side = min(
        side,
        frame_width,
        frame_height,
    )

    x1 = int(
        round(
            cx - side / 2
        )
    )

    y1 = int(
        round(
            cy - side / 2
        )
    )

    x1 = max(
        0,
        min(
            x1,
            frame_width - side
        )
    )

    y1 = max(
        0,
        min(
            y1,
            frame_height - side
        )
    )

    return (
        x1,
        y1,
        x1 + side,
        y1 + side,
    )


def fully_inside(face_box, crop):
    x, y, w, h = [
        int(v)
        for v in face_box
    ]

    x1, y1, x2, y2 = crop

    return (
        x >= x1
        and y >= y1
        and x + w <= x2
        and y + h <= y2
    )


videos = sorted(
    VIDEO_DIR.glob("*.mp4")
)

if len(videos) != 14:
    raise RuntimeError(
        f"Expected 14 videos, found {len(videos)}"
    )

rows = []
preview_items = []

print(
    "\n========== FIXED FACE CROP AUDIT =========="
)

for video in videos:

    cap = cv2.VideoCapture(
        str(video)
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    reference_index = (
        frame_count // 2
    )

    reference_frame = read_frame(
        video,
        reference_index,
    )

    reference_face = (
        detect_largest_face(
            reference_frame
        )
    )

    if reference_face is None:
        raise RuntimeError(
            "No face detected in middle frame: "
            f"{video.name}"
        )

    crop = make_square_crop(
        reference_face,
        width,
        height,
        scale=1.7,
    )

    x1, y1, x2, y2 = crop

    checkpoints = [
        int(
            round(
                (frame_count - 1)
                * fraction
            )
        )
        for fraction in (
            0.10,
            0.30,
            0.50,
            0.70,
            0.90,
        )
    ]

    detected = 0
    contained = 0

    for frame_index in checkpoints:

        frame = read_frame(
            video,
            frame_index,
        )

        face = detect_largest_face(
            frame
        )

        if face is None:
            continue

        detected += 1

        if fully_inside(
            face,
            crop,
        ):
            contained += 1

    rx, ry, rw, rh = [
        int(v)
        for v in reference_face
    ]

    rows.append({
        "filename": video.name,
        "reference_frame": reference_index,
        "face_x": rx,
        "face_y": ry,
        "face_w": rw,
        "face_h": rh,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_side": x2 - x1,
        "audit_checkpoints": len(checkpoints),
        "audit_faces_detected": detected,
        "audit_faces_contained": contained,
    })

    crop_img = reference_frame[
        y1:y2,
        x1:x2,
    ].copy()

    crop_img = cv2.resize(
        crop_img,
        (224, 224),
    )

    cv2.putText(
        crop_img,
        video.stem,
        (5, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    preview_items.append(
        crop_img
    )

    print(
        f"{video.name} | "
        f"face={rw}x{rh} | "
        f"crop={x2-x1} | "
        f"audit={contained}/{detected}"
    )


with OUT_MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


cols = 4

row_count = math.ceil(
    len(preview_items) / cols
)

blank = np.zeros(
    (224, 224, 3),
    dtype=np.uint8,
)

while (
    len(preview_items)
    < row_count * cols
):
    preview_items.append(
        blank.copy()
    )

grid_rows = []

for i in range(row_count):

    grid_rows.append(
        np.hstack(
            preview_items[
                i * cols:
                (i + 1) * cols
            ]
        )
    )

preview = np.vstack(
    grid_rows
)

preview_path = (
    QA_DIR
    / "fixed_face_crop_preview.jpg"
)

cv2.imwrite(
    str(preview_path),
    preview,
)


total_checkpoints = sum(
    int(r["audit_checkpoints"])
    for r in rows
)

total_detected = sum(
    int(r["audit_faces_detected"])
    for r in rows
)

total_contained = sum(
    int(r["audit_faces_contained"])
    for r in rows
)

containment_rate = (
    total_contained
    / total_detected
    if total_detected
    else 0.0
)


print(
    "\n========== SUMMARY =========="
)

print(
    f"Videos                  : {len(rows)}"
)

print(
    f"Reference detections    : {len(rows)}/14"
)

print(
    f"Checkpoint detections   : "
    f"{total_detected}/{total_checkpoints}"
)

print(
    f"Faces inside fixed crop : "
    f"{total_contained}/{total_detected}"
)

print(
    f"Containment rate        : "
    f"{containment_rate:.4f}"
)

print(
    f"Manifest                : "
    f"{OUT_MANIFEST}"
)

print(
    f"Preview                 : "
    f"{preview_path}"
)

print(
    "\nFIXED FACE CROP AUDIT: OK"
)

