"""
Build the competition submission PDF from the Fusion 360 component renders.

    python tools/build_render_pdf.py

Reads every PNG/JPG in --src (plus any extra folders passed with --also),
normalises them all to one identical pixel size on a white canvas, and writes
a landscape-A4 PDF: title page, then one captioned image per page.

Captions come from reports/renders/manifest.csv when it is there; otherwise
they are derived from the filename.
"""

import argparse
import csv
import os
import re
import sys

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

VIEW_LABELS = {
    "iso": "Isometric View",
    "iso_l": "Isometric View (Left)",
    "top": "Top View",
    "bottom": "Bottom View",
    "front": "Front View",
    "back": "Back View",
    "left": "Left View",
    "right": "Side View",
}


def collect(folders):
    found = []
    for folder in folders:
        if not os.path.isdir(folder):
            print("skipping missing folder: {}".format(folder), file=sys.stderr)
            continue
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(folder, name))
    return found


def load_manifest(src):
    path = os.path.join(src, "manifest.csv")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["file"]: row for row in csv.DictReader(fh)}


def caption_for(path, manifest):
    name = os.path.basename(path)
    row = manifest.get(name)
    if row:
        view = VIEW_LABELS.get(row["view"], row["view"].title())
        return row["label"], view

    stem = os.path.splitext(name)[0]
    stem = re.sub(r"^\d+[_\-]", "", stem)
    parts = re.split(r"[_\-]+", stem)
    view = ""
    if parts and parts[-1].lower() in VIEW_LABELS:
        view = VIEW_LABELS[parts.pop().lower()]
    return " ".join(p for p in parts if p).title() or stem, view


def trim_border(img, tolerance=8):
    """Crop a flat near-uniform border so mixed-source images frame alike.

    No-ops on Fusion's grid background, which is not uniform.
    """
    from PIL import ImageChops

    background = Image.new("RGB", img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, background).convert("L")
    box = diff.point(lambda p: 255 if p > tolerance else 0).getbbox()
    if not box:
        return img
    pad = int(0.02 * max(img.width, img.height))
    left = max(0, box[0] - pad)
    top = max(0, box[1] - pad)
    right = min(img.width, box[2] + pad)
    bottom = min(img.height, box[3] + pad)
    if right - left < 32 or bottom - top < 32:
        return img
    return img.crop((left, top, right, bottom))


def normalise(path, size, trim=True):
    """Fit the image inside `size` on a white canvas, keeping aspect ratio."""
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, "white")
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")

    if trim:
        img = trim_border(img)

    # Scale to fit the canvas -- up as well as down, so images that came from
    # smaller screenshots still fill the frame like the native renders do.
    ratio = min(size[0] / img.width, size[1] / img.height)
    img = img.resize((max(1, round(img.width * ratio)),
                      max(1, round(img.height * ratio))), Image.LANCZOS)

    canvas_img = Image.new("RGB", size, "white")
    canvas_img.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.join("reports", "renders"),
                        help="folder of Fusion-exported renders")
    parser.add_argument("--also", nargs="*", default=[],
                        help="extra folders of images to append")
    parser.add_argument("--out", default=os.path.join("reports", "vehicle_components.pdf"))
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--no-trim", action="store_true",
                        help="keep the original framing instead of cropping flat borders")
    parser.add_argument("--title", default="Autonomous Sprayer Vehicle")
    parser.add_argument("--subtitle", default="Component Views — Design Submission")
    args = parser.parse_args()

    size = (args.width, args.height)
    manifest = load_manifest(args.src)
    images = collect([args.src] + list(args.also))
    if not images:
        print("No images found in {}. Run the Fusion script first.".format(args.src),
              file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    page_w, page_h = landscape(A4)
    pdf = pdfcanvas.Canvas(args.out, pagesize=(page_w, page_h))
    pdf.setTitle(args.title)

    # Title page.
    pdf.setFont("Helvetica-Bold", 30)
    pdf.drawCentredString(page_w / 2, page_h / 2 + 24, args.title)
    pdf.setFont("Helvetica", 15)
    pdf.drawCentredString(page_w / 2, page_h / 2 - 6, args.subtitle)
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(page_w / 2, page_h / 2 - 34,
                          "{} views · all images {} × {} px".format(
                              len(images), args.width, args.height))
    pdf.showPage()

    margin = 14 * mm
    caption_h = 16 * mm
    box_w = page_w - 2 * margin
    box_h = page_h - 2 * margin - caption_h

    for index, path in enumerate(images, start=1):
        label, view = caption_for(path, manifest)
        img = normalise(path, size, trim=not args.no_trim)

        scale = min(box_w / img.width, box_h / img.height)
        draw_w, draw_h = img.width * scale, img.height * scale
        x = (page_w - draw_w) / 2
        y = margin + caption_h + (box_h - draw_h) / 2

        pdf.drawImage(ImageReader(img), x, y, draw_w, draw_h)

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(margin, margin + 6 * mm, label)
        if view:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(margin, margin + 1.5 * mm, view)
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(page_w - margin, margin + 1.5 * mm,
                            "{} / {}".format(index, len(images)))
        pdf.showPage()

    pdf.save()
    print("Wrote {} with {} image pages -> {}".format(args.out, len(images), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
