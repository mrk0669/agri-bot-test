# Component renders → submission PDF

Two steps. Step 1 runs inside Fusion 360 (it needs the 3D viewport); step 2 runs here.

## 1. Export the images from Fusion 360

1. Open the vehicle assembly in Fusion 360.
2. Bottom toolbar → **Grid and Snaps** → untick **Layout Grid** (clean white background).
3. **Utilities** tab → **Scripts and Add-Ins** → **Scripts** → **GUJU_RenderShots** → **Run**.

The script is already installed in Fusion's Scripts folder, so it appears in that
list without adding anything. It writes to `reports/renders/`:

- 34 PNGs, every one exactly **2400 × 1800 px**
- `manifest.csv` — captions and view names
- `components.txt` — the real top-level component names in your design

It hides the origin planes, sketches and construction geometry before rendering,
and puts every visibility setting back the way it was when it finishes.

### If a shot is missing

The dialog at the end lists any shot where no component matched. Open
`reports/renders/components.txt`, find the real names, and edit the `SHOTS` list
in `tools/fusion_render_shots.py` — each entry is:

```
(order, label, keywords_to_show, keywords_to_hide, views)
```

Then copy the edited file over
`%APPDATA%\Autodesk\Autodesk Fusion 360\API\Scripts\GUJU_RenderShots\GUJU_RenderShots.py`
and run again.

## 2. Build the PDF

```bash
python tools/build_render_pdf.py
```

Writes `reports/vehicle_components.pdf` — title page, then one captioned image
per page on landscape A4.

To fold in the screenshots you already took, drop them in a folder and add:

```bash
python tools/build_render_pdf.py --also reports/manual_shots
```

Every image is normalised to the same 2400 × 1800 canvas, so mixed sources still
come out at one uniform resolution.

## Shot list (34 images)

| # | Subject | Views |
|---|---------|-------|
| 01 | Complete Vehicle | iso, iso-left, top, bottom, front, side |
| 02 | Chassis without Tank | iso, top, front, side |
| 03 | Chassis Frame | iso, top, front |
| 04 | Tank | iso, top, front |
| 05 | Wheel Assembly | iso, front, side |
| 06 | Tire | iso, front, side |
| 07 | Spray Nozzle | iso, front, top |
| 08 | Spray Bar / Connectors | iso, front, top |
| 09 | Gearbox & Drive | iso, front, side |
| 10 | Camera Gimbal | iso, front, side |
| 11 | Raspberry Pi 4 | iso, top, front |
| 12 | Motor Driver (BTS7960) | iso, top, front |
| 13 | Battery Pack | iso, front, side |
