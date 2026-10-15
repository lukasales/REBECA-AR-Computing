from pathlib import Path
import math
import cv2


ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    ROOT
    / "data"
    / "video"
    / "ravdess"
    / "selected"
)

SCALES = [1.5, 1.6, 1.7, 1.8]

cascade_path = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(
    str(cascade_path)
)

if detector.empty():
    raise RuntimeError("Could not load Haar cascade.")


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
    scale,
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

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

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


total_frames = 0
total_detected = 0

contained = {
    scale: 0
    for scale in SCALES
}

per_video = []


print(
    "\n========== FULL-FRAME CROP SCALE AUDIT =========="
)


for video in videos:

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {video.name}"
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    reference_index = frame_count // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        reference_index
    )

    ok, reference_frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Could not read reference frame: {video.name}"
        )

    reference_face = detect_largest_face(
        reference_frame
    )

    if reference_face is None:
        raise RuntimeError(
            f"No reference face: {video.name}"
        )

    crops = {
        scale: make_square_crop(
            reference_face,
            width,
            height,
            scale,
        )
        for scale in SCALES
    }

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    detected_video = 0

    contained_video = {
        scale: 0
        for scale in SCALES
    }

    frames_read = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames_read += 1

        face = detect_largest_face(
            frame
        )

        if face is None:
            continue

        detected_video += 1

        for scale in SCALES:
            if fully_inside(
                face,
                crops[scale],
            ):
                contained_video[scale] += 1

    cap.release()

    total_frames += frames_read
    total_detected += detected_video

    for scale in SCALES:
        contained[scale] += (
            contained_video[scale]
        )

    per_video.append(
        (
            video.name,
            frames_read,
            detected_video,
            contained_video,
        )
    )

    rates = " | ".join(
        (
            f"{scale:.1f}="
            f"{contained_video[scale]}/"
            f"{detected_video}"
        )
        for scale in SCALES
    )

    print(
        f"{video.name} | "
        f"frames={frames_read} | "
        f"detected={detected_video} | "
        f"{rates}"
    )


print(
    "\n========== AGGREGATE =========="
)

print(
    f"Total frames       : {total_frames}"
)

print(
    f"Faces detected     : {total_detected}"
)

if total_frames:
    print(
        "Detection rate     : "
        f"{total_detected / total_frames:.4f}"
    )


for scale in SCALES:

    count = contained[scale]

    rate = (
        count / total_detected
        if total_detected
        else 0.0
    )

    print(
        f"Scale {scale:.1f} containment: "
        f"{count}/{total_detected} "
        f"({rate:.4f})"
    )


perfect_scales = [
    scale
    for scale in SCALES
    if contained[scale] == total_detected
]

print()

if perfect_scales:
    print(
        "Smallest perfect scale: "
        f"{min(perfect_scales):.1f}"
    )
else:
    best_scale = max(
        SCALES,
        key=lambda s: contained[s]
    )

    print(
        "No scale reached perfect containment."
    )

    print(
        "Best tested scale: "
        f"{best_scale:.1f}"
    )


print(
    "\nCROP SCALE SENSITIVITY AUDIT: OK"
)
