"""
Fusion 360 script: batch component renders for the competition submission.

HOW TO RUN
    1. Open the vehicle assembly in Fusion 360.
    2. (Optional, recommended) Bottom toolbar -> Grid and Snaps -> untick "Layout Grid"
       so the background is clean white like the reference images.
    3. Utilities tab -> ADD-INS -> Scripts and Add-Ins -> Scripts -> "+"
       -> pick this folder -> select fusion_render_shots -> Run.

    Every image lands in OUT_DIR at exactly RENDER_W x RENDER_H pixels,
    plus a manifest.csv that tools/build_render_pdf.py turns into the PDF.

If a shot comes out empty, open OUT_DIR/components.txt -- it lists the real
top-level component names in your design -- and fix the keywords in SHOTS.
"""

import adsk.core
import adsk.fusion
import os
import re
import csv
import traceback

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

OUT_DIR = r"C:\Users\KESHAV\Desktop\GUJU\reports\renders"

RENDER_W = 2400
RENDER_H = 1800

# Hide origin planes / sketches / construction geometry before rendering.
CLEAN_VIEW = True

# Render only these shot numbers, e.g. ONLY = ["07"] for just the spray nozzle.
# Empty list = render every shot in SHOTS.
ONLY = []

# Each shot is: (order, label, keywords, exclude_keywords, views)
#   keywords        - show only top-level components whose name contains any of
#                     these (case-insensitive). Empty list = show everything.
#   exclude_keywords- of those, hide any whose name contains one of these.
#   views           - any of: iso, iso_l, top, bottom, front, back, left, right
SHOTS = [
    ("01", "Complete Vehicle",        [], [],
     ["iso", "iso_l", "top", "bottom", "front", "right"]),

    ("02", "Chassis without Tank",    [], ["tank", "enclosure"],
     ["iso", "top", "front", "right"]),

    ("03", "Chassis Frame",           ["chassis", "frame"], [],
     ["iso", "top", "front"]),

    ("04", "Tank",                    ["tank", "enclosure"], [],
     ["iso", "top", "front"]),

    ("05", "Wheel Assembly",          ["wheel", "tire", "tyre", "axel", "axle", "hub"], [],
     ["iso", "front", "right"]),

    ("06", "Tire",                    ["tire", "tyre"], ["adapter", "holder"],
     ["iso", "front", "right"]),

    ("07", "Spray Nozzle",            ["nozzle", "spray"], [],
     ["iso", "front", "top"]),

    ("08", "Spray Bar / Connectors",  ["nozzle", "spray", "pipe", "manifold", "connector", "boom"], [],
     ["iso", "front", "top"]),

    ("09", "Gearbox & Drive",         ["gearbox", "motor", "gear"], [],
     ["iso", "front", "right"]),

    ("10", "Camera Gimbal",           ["gimbal", "camera", "servo"], [],
     ["iso", "front", "right"]),

    ("11", "Raspberry Pi 4",          ["raspberry", "rpi", "pi 4"], [],
     ["iso", "top", "front"]),

    ("12", "Motor Driver (BTS7960)",  ["bts", "btn", "bridge", "driver"], [],
     ["iso", "top", "front"]),

    ("13", "Battery Pack",            ["battery"], [],
     ["iso", "front", "right"]),
]


# ----------------------------------------------------------------------------

def sanitize(name):
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name[:60] or "part"


def view_map():
    vo = adsk.core.ViewOrientations
    return {
        "iso":    vo.IsoTopRightViewOrientation,
        "iso_l":  vo.IsoTopLeftViewOrientation,
        "top":    vo.TopViewOrientation,
        "bottom": vo.BottomViewOrientation,
        "front":  vo.FrontViewOrientation,
        "back":   vo.BackViewOrientation,
        "left":   vo.LeftViewOrientation,
        "right":  vo.RightViewOrientation,
    }


def matches(name, keywords):
    low = name.lower()
    return any(k.lower() in low for k in keywords)


def apply_visibility(occurrences, keywords, exclude_keywords):
    """Return the number of occurrences left visible."""
    shown = 0
    for occ in occurrences:
        name = occ.name
        want = True if not keywords else matches(name, keywords)
        if want and exclude_keywords and matches(name, exclude_keywords):
            want = False
        try:
            occ.isLightBulbOn = want
        except Exception:
            continue
        if want:
            shown += 1
    return shown


def set_view(viewport, orientation):
    cam = viewport.camera
    cam.viewOrientation = orientation
    cam.isFitView = True
    viewport.camera = cam
    viewport.refresh()
    adsk.doEvents()
    viewport.fit()
    adsk.doEvents()


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("Open the vehicle design in the Design workspace, then run again.")
            return

        root = design.rootComponent
        viewport = app.activeViewport
        views = view_map()

        os.makedirs(OUT_DIR, exist_ok=True)

        occurrences = list(root.occurrences)
        if not occurrences:
            ui.messageBox("No top-level components found in the root component.")
            return

        # Dump the real component names so the keyword lists can be corrected.
        with open(os.path.join(OUT_DIR, "components.txt"), "w", encoding="utf-8") as fh:
            for occ in occurrences:
                fh.write(occ.name + "\n")

        # Remember what was visible so we can put the design back afterwards.
        original = [(occ, occ.isLightBulbOn) for occ in occurrences]
        original_camera = viewport.camera

        origin_state = root.isOriginFolderLightBulbOn
        sketch_state = root.isSketchFolderLightBulbOn
        constr_state = root.isConstructionFolderLightBulbOn
        if CLEAN_VIEW:
            root.isOriginFolderLightBulbOn = False
            root.isSketchFolderLightBulbOn = False
            root.isConstructionFolderLightBulbOn = False

        try:
            viewport.visualStyle = adsk.core.VisualStyles.ShadedVisualStyle
        except Exception:
            pass

        rows = []
        skipped = []

        for order, label, keywords, exclude_keywords, view_names in SHOTS:
            if ONLY and order not in ONLY:
                continue
            shown = apply_visibility(occurrences, keywords, exclude_keywords)
            if shown == 0:
                skipped.append(label)
                continue

            for view_name in view_names:
                orientation = views.get(view_name)
                if orientation is None:
                    continue
                set_view(viewport, orientation)

                filename = "{}_{}_{}.png".format(order, sanitize(label), view_name)
                path = os.path.join(OUT_DIR, filename)
                if viewport.saveAsImageFile(path, RENDER_W, RENDER_H):
                    rows.append({
                        "file": filename,
                        "label": label,
                        "view": view_name,
                        "components_visible": shown,
                    })

        # Restore the design.
        for occ, state in original:
            try:
                occ.isLightBulbOn = state
            except Exception:
                pass
        if CLEAN_VIEW:
            root.isOriginFolderLightBulbOn = origin_state
            root.isSketchFolderLightBulbOn = sketch_state
            root.isConstructionFolderLightBulbOn = constr_state
        viewport.camera = original_camera
        viewport.refresh()

        # Merge into any existing manifest so a single-shot re-run (ONLY = [...])
        # does not wipe the captions for the shots it skipped.
        manifest = os.path.join(OUT_DIR, "manifest.csv")
        fields = ["file", "label", "view", "components_visible"]
        merged = {}
        if os.path.isfile(manifest):
            with open(manifest, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    merged[row["file"]] = row
        for row in rows:
            merged[row["file"]] = row

        with open(manifest, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for key in sorted(merged):
                writer.writerow({f: merged[key].get(f, "") for f in fields})

        message = "Exported {} images at {}x{} to:\n{}".format(
            len(rows), RENDER_W, RENDER_H, OUT_DIR)
        if skipped:
            message += ("\n\nNo components matched these shots:\n  - "
                        + "\n  - ".join(skipped)
                        + "\n\nCheck components.txt in the same folder and fix "
                          "the keywords in SHOTS.")
        ui.messageBox(message)

    except Exception:
        if ui:
            ui.messageBox("Script failed:\n{}".format(traceback.format_exc()))
