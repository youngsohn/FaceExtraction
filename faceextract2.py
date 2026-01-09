import os
import cv2
import numpy as np
from PIL import Image


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0


def extract_faces_opencv(
    input_image="test.jpeg",
    output_dir="faces_out",
    margin_ratio=0.30,
    jpeg_quality=90,
):
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(input_image)
    if img_bgr is None:
        raise FileNotFoundError(input_image)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    haar_dir = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(
        haar_dir + "haarcascade_frontalface_default.xml"
    )
    profile = cv2.CascadeClassifier(
        haar_dir + "haarcascade_profileface.xml"
    )

    faces_f = frontal.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    faces_p = profile.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )

    all_faces = list(faces_f) + list(faces_p)

    # Deduplicate using IoU
    final_faces = []
    for f in all_faces:
        if all(iou(f, g) < 0.3 for g in final_faces):
            final_faces.append(f)

    print(f"Detected {len(final_faces)} faces")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    for idx, (x, y, bw, bh) in enumerate(final_faces, start=1):
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        face = img_rgb[y1:y2, x1:x2]
        Image.fromarray(face).save(
            os.path.join(output_dir, f"{idx:03d}.jpg"),
            "JPEG",
            quality=jpeg_quality,
            optimize=True
        )

        print(f"Saved faces_out/{idx:03d}.jpg")



if __name__ == "__main__":
    extract_faces_opencv()