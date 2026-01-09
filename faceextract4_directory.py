import os
import cv2
import numpy as np
from PIL import Image
from mediapipe import solutions as mp_solutions


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh=0.35):
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


def verify_with_mediapipe(face_detector, crop_rgb, min_conf=0.45):
    res = face_detector.process(crop_rgb)
    if not res.detections:
        return None
    best = max(float(d.score[0]) for d in res.detections if d.score)
    return best if best >= min_conf else None


# --------------------------------------------------
# Core face extraction for ONE image
# --------------------------------------------------

def extract_faces_from_image(
    image_path,
    margin_ratio=0.30,
    jpeg_quality=90,
    expected_faces=None,
):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    haar_dir = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(
        haar_dir + "haarcascade_frontalface_default.xml"
    )
    profile = cv2.CascadeClassifier(
        haar_dir + "haarcascade_profileface.xml"
    )

    proposals = []
    proposals += list(frontal.detectMultiScale(gray, 1.05, 3, minSize=(40, 40)))
    proposals += list(profile.detectMultiScale(gray, 1.05, 3, minSize=(40, 40)))

    gray_flip = cv2.flip(gray, 1)
    prof_flip = profile.detectMultiScale(gray_flip, 1.05, 3, minSize=(40, 40))
    for (x, y, w, h) in prof_flip:
        proposals.append((W - x - w, y, w, h))

    # geometry filter
    filtered = []
    for (x, y, w, h) in proposals:
        ar = w / float(h)
        if 0.6 <= ar <= 1.6 and (w * h) > 0.0025 * (W * H):
            filtered.append((x, y, w, h))

    if not filtered:
        return

    detector = mp_solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.1
    )

    verified_boxes = []
    verified_scores = []

    for (x, y, w, h) in filtered:
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)

        x1 = clamp(x - mx, 0, W)
        y1 = clamp(y - my, 0, H)
        x2 = clamp(x + w + mx, 0, W)
        y2 = clamp(y + h + my, 0, H)

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        score = verify_with_mediapipe(detector, crop)
        if score is None:
            continue

        verified_boxes.append((x1, y1, x2 - x1, y2 - y1))
        verified_scores.append(score)

    detector.close()

    if not verified_boxes:
        return

    keep = nms(verified_boxes, verified_scores)
    boxes = [verified_boxes[i] for i in keep]
    scores = [verified_scores[i] for i in keep]

    if expected_faces and len(boxes) > expected_faces:
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        idx = idx[:expected_faces]
        boxes = [boxes[i] for i in idx]

    # save faces
    base = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.dirname(image_path)

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        face = img_rgb[y:y+h, x:x+w]
        out_name = f"{base}_face{i:02d}.jpg"
        out_path = os.path.join(folder, out_name)
        Image.fromarray(face).save(
            out_path,
            "JPEG",
            quality=jpeg_quality,
            optimize=True
        )


# --------------------------------------------------
# Batch over folders
# --------------------------------------------------

def process_all_folders(input_root="./input"):
    for sub in sorted(os.listdir(input_root)):
        subdir = os.path.join(input_root, sub)
        if not os.path.isdir(subdir):
            continue

        print(f"\nProcessing folder: {subdir}")

        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith(".jpeg"):
                img_path = os.path.join(subdir, fname)
                print(f"  Processing image: {fname}")
                extract_faces_from_image(img_path)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    process_all_folders("./input")