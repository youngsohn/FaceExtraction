import os
import cv2
import numpy as np
from PIL import Image
from mediapipe import solutions as mp_solutions


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou(a, b):
    # a, b are (x, y, w, h)
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh=0.35):
    """Non-maximum suppression. Keeps high-score boxes, removes near-duplicates."""
    if not boxes:
        return []

    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


def verify_with_mediapipe(face_detector, crop_rgb, min_conf=0.4):
    """
    Returns a score in [0,1] if a face is found in crop, else None.
    We accept detections that look like a face inside the crop.
    """
    res = face_detector.process(crop_rgb)
    if not res.detections:
        return None

    # take best detection score
    best = max(float(d.score[0]) for d in res.detections if d.score)
    return best if best >= min_conf else None


def extract_faces_refined(
    input_image="test.jpeg",
    output_dir="faces_out",
    jpeg_quality=90,
    margin_ratio=0.30,
    expected_faces=6,          # set None if you don't want a cap
    haar_scaleFactor=1.05,
    haar_minNeighbors=3,       # proposals: keep lower for recall, verification will clean
    haar_minSize=(40, 40),
    verify_min_conf=0.45,      # increase to reduce false positives; lower to catch occluded faces
    nms_iou_thresh=0.35,
):
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(input_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {input_image}")

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1) Haar proposals (frontal + profile + flipped profile) ---
    haar_dir = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(haar_dir + "haarcascade_profileface.xml")

    proposals = []
    proposals += list(frontal.detectMultiScale(
        gray, scaleFactor=haar_scaleFactor, minNeighbors=haar_minNeighbors, minSize=haar_minSize
    ))
    proposals += list(profile.detectMultiScale(
        gray, scaleFactor=haar_scaleFactor, minNeighbors=haar_minNeighbors, minSize=haar_minSize
    ))

    # also run profile on horizontally flipped image (catches the other side)
    gray_flip = cv2.flip(gray, 1)
    prof_flip = profile.detectMultiScale(
        gray_flip, scaleFactor=haar_scaleFactor, minNeighbors=haar_minNeighbors, minSize=haar_minSize
    )
    # map flipped coords back
    for (x, y, w, h) in prof_flip:
        x_unflip = (W - x - w)
        proposals.append((x_unflip, y, w, h))

    # quick geometry filter to reduce obvious junk
    filtered = []
    for (x, y, w, h) in proposals:
        if w < haar_minSize[0] or h < haar_minSize[1]:
            continue
        ar = w / float(h)
        if ar < 0.6 or ar > 1.6:   # faces are roughly near-square
            continue
        if (w * h) < 0.0025 * (W * H):  # too tiny => often noise
            continue
        filtered.append((int(x), int(y), int(w), int(h)))

    if not filtered:
        print("No candidate boxes found.")
        return

    # --- 2) Verify each candidate using MediaPipe FaceDetection ---
    face_detector = mp_solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.1  # internal; we do our own verify_min_conf threshold
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

        score = verify_with_mediapipe(face_detector, crop, min_conf=verify_min_conf)
        if score is None:
            continue

        verified_boxes.append((x1, y1, x2 - x1, y2 - y1))
        verified_scores.append(score)

    face_detector.close()

    if not verified_boxes:
        print("No faces verified. Try lowering verify_min_conf (e.g., 0.35).")
        return

    # --- 3) NMS to remove duplicates ---
    keep_idx = nms(verified_boxes, verified_scores, iou_thresh=nms_iou_thresh)
    boxes_nms = [verified_boxes[i] for i in keep_idx]
    scores_nms = [verified_scores[i] for i in keep_idx]

    # sort left-to-right (nicer numbering) but keep scores for top_k selection
    order_lr = sorted(range(len(boxes_nms)), key=lambda i: boxes_nms[i][0])
    boxes_nms = [boxes_nms[i] for i in order_lr]
    scores_nms = [scores_nms[i] for i in order_lr]

    # --- 4) Optional: keep top expected_faces by score (after NMS) ---
    if expected_faces is not None and len(boxes_nms) > expected_faces:
        idx_sorted = sorted(range(len(boxes_nms)), key=lambda i: scores_nms[i], reverse=True)[:expected_faces]
        idx_sorted = sorted(idx_sorted, key=lambda i: boxes_nms[i][0])  # re-sort left->right for naming
        boxes_nms = [boxes_nms[i] for i in idx_sorted]
        scores_nms = [scores_nms[i] for i in idx_sorted]

    print(f"Final faces: {len(boxes_nms)}")
    out_rgb = img_rgb

    # save
    for k, (x, y, w, h) in enumerate(boxes_nms, start=1):
        face = out_rgb[y:y+h, x:x+w]
        out_path = os.path.join(output_dir, f"{k:03d}.jpg")
        Image.fromarray(face).save(out_path, "JPEG", quality=jpeg_quality, optimize=True)
        print(f"Saved {out_path} (verify_conf={scores_nms[k-1]:.3f})")


if __name__ == "__main__":
    extract_faces_refined(
        input_image="test.jpeg",
        output_dir="faces_out",
        expected_faces=6,     # set None if you don't want to cap
        verify_min_conf=0.45  # if it misses faces, try 0.40 or 0.35
    )