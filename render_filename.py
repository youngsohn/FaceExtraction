import os
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_SIZE = 20
TEXT_COLOR = (255, 255, 255)
STROKE_COLOR = (0, 0, 0)
STROKE_WIDTH = 2
BOTTOM_MARGIN = 10

def process_image(path):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    name_only = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(
        os.path.dirname(path),
        f"{name_only}_named.jpeg"
    )

    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), name_only, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (img.width - w) // 2
    y = img.height - h - BOTTOM_MARGIN

    draw.text(
        (x, y),
        name_only,
        font=font,
        fill=TEXT_COLOR,
        stroke_width=STROKE_WIDTH,
        stroke_fill=STROKE_COLOR,
    )

    img.save(output_path, "JPEG", quality=95)
    print(f"[OK] {output_path}")

def main():
    print("Scanning recursively...")
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower().endswith(".jpeg"):
                full_path = os.path.join(root, f)
                print(f"Processing: {full_path}")
                process_image(full_path)

if __name__ == "__main__":
    main()