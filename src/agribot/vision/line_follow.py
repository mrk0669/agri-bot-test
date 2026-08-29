"""Vision-based guidance-line extraction (Section 5.2).

The competition guidelines require navigation through a computer-vision
technique, so the guidance line is tracked with the camera rather than the
infra-red array (which is retained only as a silent fail-safe).

Pipeline, per frame:

1. Crop to a region of interest near the bottom of the frame, so the controller
   reacts to the section of line immediately ahead of the wheels rather than to
   distant curvature.
2. Convert BGR -> HSV and threshold on the line colour. HSV is far more robust
   to the uneven illumination of a real field than a fixed intensity threshold,
   because hue and saturation are largely invariant to brightness.
3. Morphological opening then closing, which removes soil speckle and fills
   glare holes in the line.
4. Compute the centroid from the image moments, ``Cx = M10 / M00``.
5. Normalise the horizontal offset from the frame centre into ``[-1, +1]``.

The normalised error is what the PID consumes, which keeps the gains
independent of capture resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import LineObservation
from ..utils.geometry import normalise_error

__all__ = ["LineFollower", "extract_roi", "hsv_mask"]


def extract_roi(
    frame: np.ndarray,
    top: float,
    bottom: float,
    left: float = 0.0,
    right: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop a fractional region of interest.

    Returns the cropped view and the ``(x_offset, y_offset)`` needed to map
    ROI-local pixel coordinates back into full-frame coordinates.
    """
    if frame.ndim < 2:
        raise ValueError("frame must be at least 2-D")
    h, w = frame.shape[:2]
    # Guarantee a non-empty slice for any fractions, including top == bottom
    # and 1.0. Clamping the start to the last valid index before widening the
    # end is what keeps a degenerate request inside the array: widening first
    # produces frame[h:h+1], which is empty, and every downstream op then
    # raises on a zero-size image.
    y1 = min(int(np.clip(top, 0.0, 1.0) * h), max(h - 1, 0))
    x1 = min(int(np.clip(left, 0.0, 1.0) * w), max(w - 1, 0))
    y2 = min(max(int(np.clip(bottom, 0.0, 1.0) * h), y1 + 1), h)
    x2 = min(max(int(np.clip(right, 0.0, 1.0) * w), x1 + 1), w)
    return frame[y1:y2, x1:x2], (x1, y1)


def hsv_mask(
    bgr_roi: np.ndarray,
    lower: Sequence[int],
    upper: Sequence[int],
    morph_kernel: int = 5,
    open_iters: int = 1,
    close_iters: int = 2,
    blur_kernel: int = 0,
) -> np.ndarray:
    """Threshold a BGR image in HSV and clean the result morphologically.

    Opening first (erode-then-dilate) deletes isolated soil speckle; closing
    second (dilate-then-erode) fills glare holes without re-growing the speckle
    that opening just removed. Doing them in the other order re-inflates noise.
    """
    if bgr_roi.size == 0:
        raise ValueError("empty ROI passed to hsv_mask")

    work = bgr_roi
    if blur_kernel and blur_kernel >= 3:
        k = blur_kernel | 1     # OpenCV requires an odd kernel
        work = cv2.GaussianBlur(work, (k, k), 0)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))

    if morph_kernel and morph_kernel >= 3:
        k = morph_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        if close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    return mask


@dataclass
class LineFollower:
    """Stateless-per-frame guidance-line extractor.

    ``process`` returns a :class:`~agribot.types.LineObservation`; the caller
    decides what to do when ``found`` is False (the mission state machine
    handles the grace period and the fail-safe).
    """

    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    roi_top: float = 0.62
    roi_bottom: float = 1.0
    roi_left: float = 0.0
    roi_right: float = 1.0
    morph_kernel: int = 5
    morph_open_iters: int = 1
    morph_close_iters: int = 2
    min_mask_area_px: float = 400.0
    largest_component_only: bool = True
    # -- line-shape gates (see _select_component) ---------------------------
    min_height_fraction: float = 0.60
    blob_reject_aspect: float = 0.75
    blob_reject_fill: float = 0.60

    @classmethod
    def from_config(cls, cfg) -> "LineFollower":
        """Build from the ``navigation.line`` config section."""
        return cls(
            hsv_lower=tuple(cfg.hsv_lower),
            hsv_upper=tuple(cfg.hsv_upper),
            roi_top=cfg.get("roi_top", 0.62),
            roi_bottom=cfg.get("roi_bottom", 1.0),
            roi_left=cfg.get("roi_left", 0.0),
            roi_right=cfg.get("roi_right", 1.0),
            morph_kernel=cfg.get("morph_kernel", 5),
            morph_open_iters=cfg.get("morph_open_iters", 1),
            morph_close_iters=cfg.get("morph_close_iters", 2),
            min_mask_area_px=cfg.get("min_mask_area_px", 400),
            min_height_fraction=cfg.get("min_height_fraction", 0.60),
            blob_reject_aspect=cfg.get("blob_reject_aspect", 0.75),
            blob_reject_fill=cfg.get("blob_reject_fill", 0.60),
        )

    def build_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return the cleaned binary mask of the ROI and its frame offset."""
        roi, offset = extract_roi(
            frame, self.roi_top, self.roi_bottom, self.roi_left, self.roi_right
        )
        mask = hsv_mask(
            roi,
            self.hsv_lower,
            self.hsv_upper,
            self.morph_kernel,
            self.morph_open_iters,
            self.morph_close_iters,
        )
        return mask, offset

    def process(self, frame: np.ndarray) -> LineObservation:
        """Extract the normalised horizontal line error from one BGR frame."""
        if frame is None or frame.size == 0:
            return LineObservation(found=False)

        frame_w = frame.shape[1]
        mask, (x_off, y_off) = self.build_mask(frame)
        roi_h, roi_w = mask.shape[:2]

        if self.largest_component_only:
            selected = self._select_component(mask)
            if selected is None:
                return LineObservation(
                    found=False,
                    mask_area_px=float(cv2.countNonZero(mask)),
                    roi_shape=(roi_h, roi_w),
                )
            mask = selected

        moments = cv2.moments(mask, binaryImage=True)
        area = float(moments["m00"])

        if area < self.min_mask_area_px:
            return LineObservation(
                found=False,
                mask_area_px=area,
                roi_shape=(roi_h, roi_w),
            )

        # Cx = M10 / M00 - the first-order moment over the zeroth.
        cx_roi = moments["m10"] / area
        cy_roi = moments["m01"] / area
        cx_frame = cx_roi + x_off
        cy_frame = cy_roi + y_off

        error = normalise_error(cx_frame, frame_w)
        roi_area = float(roi_h * roi_w)
        confidence = float(np.clip(area / roi_area, 0.0, 1.0)) if roi_area else 0.0

        return LineObservation(
            found=True,
            error=error,
            centroid_px=(cx_frame, cy_frame),
            mask_area_px=area,
            roi_shape=(roi_h, roi_w),
            confidence=confidence,
        )

    def _select_component(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Pick the connected component that is actually the guidance line.

        Taking simply the largest blob is not enough. Specular glare on wet
        soil saturates towards white, and adding a constant to all three
        channels *lowers* saturation - so a glare patch lands squarely inside
        a "bright, unsaturated" line threshold. Left unchecked the robot
        steers towards a reflection of the sun.

        Two shape gates separate them, both measured against the ROI rather
        than in absolute pixels so they survive a resolution change:

        * **Vertical span.** The guidance line runs through the ROI, so its
          component spans essentially the full ROI height. A glare patch is a
          disc and spans a fraction of it.
        * **Broad *and* filled.** A line tilted far enough to be as wide as it
          is tall is a diagonal band, which fills only about half of its
          bounding box. A disc that wide fills three quarters of it. Rejecting
          only on the *conjunction* keeps a steeply tilted line - which is
          exactly the frame a recovering robot needs - while dropping the disc.

        Components are tried largest first, so a real line is still found when
        a larger glare patch shares the frame. Returns None if none qualifies.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:                      # background only
            return None

        roi_h, roi_w = mask.shape[:2]
        candidates = sorted(
            range(1, count),
            key=lambda i: int(stats[i, cv2.CC_STAT_AREA]),
            reverse=True,
        )

        for idx in candidates:
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < self.min_mask_area_px:
                break                       # sorted by area: the rest are smaller

            w = float(stats[idx, cv2.CC_STAT_WIDTH])
            h = float(stats[idx, cv2.CC_STAT_HEIGHT])
            if h <= 0 or w <= 0:
                continue

            if (h / roi_h) < self.min_height_fraction:
                continue

            aspect = w / h
            fill = area / (w * h)
            if aspect > self.blob_reject_aspect and fill > self.blob_reject_fill:
                continue

            return np.where(labels == idx, 255, 0).astype(np.uint8)

        return None

    def debug_overlay(self, frame: np.ndarray, obs: LineObservation) -> np.ndarray:
        """Annotate a frame with the ROI, centroid and error - for the tuning tool."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        y1, y2 = int(self.roi_top * h), int(self.roi_bottom * h)
        x1, x2 = int(self.roi_left * w), int(self.roi_right * w)
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), 2)
        cv2.line(vis, (w // 2, y1), (w // 2, y2), (200, 200, 200), 1)

        if obs.found and obs.centroid_px:
            cx, cy = int(obs.centroid_px[0]), int(obs.centroid_px[1])
            cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, cy), (cx, cy), (0, 0, 255), 2)
            label = f"err {obs.error:+.3f}  area {obs.mask_area_px:.0f}"
            colour = (0, 255, 0)
        else:
            label = "LINE LOST"
            colour = (0, 0, 255)

        cv2.putText(vis, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return vis
