import os
import cv2
from PIL import Image
from mediapipe import solutions as mp_solutions


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def extract_faces_mediapipe_detector(
    input_image="test.jpeg",
    output_dir="faces_out",
    margin_ratio=0.35,
    jpeg_quality=90,
    min_confidence=0.15,   # VERY IMPORTANT
):
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(input_image)
    if image_bgr is None:
        raise FileNotFoundError(input_image)

    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detector = mp_solutions.face_detection.FaceDetection(
        model_selection=1,                # long-range
        min_detection_confidence=min_confidence
    )

    results = detector.process(image_rgb)

    if not results.detections:
        print("No faces detected.")
        return

    print(f"Detected {len(results.detections)} faces")

    # Sort left → right (nice numbering)
    detections = sorted(
        results.detections,
        key=lambda d: d.location_data.relative_bounding_box.xmin
    )

    for idx, det in enumerate(detections, start=1):
        bbox = det.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x1 = clamp(x - mx, 0, w)
        y1 = clamp(y - my, 0, h)
        x2 = clamp(x + bw + mx, 0, w)
        y2 = clamp(y + bh + my, 0, h)

        face_rgb = image_rgb[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_rgb)

        out_path = os.path.join(output_dir, f"{idx:03d}.jpg")
        face_pil.save(out_path, "JPEG", quality=jpeg_quality, optimize=True)

        print(f"Saved {out_path} (conf={det.score[0]:.3f})")

    detector.close()


if __name__ == "__main__":
    extract_faces_mediapipe_detector()