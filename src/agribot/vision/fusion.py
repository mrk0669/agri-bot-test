"""Late fusion of the perception tiers, and target tracking (Section 5.6).

The fusion rule is deliberately **asymmetric** (Novelty 4). An action is
triggered when the colour tier reports a red weed marker, or when the learned
detector reports a weed with high confidence. But any region classified as crop
suppresses action entirely, regardless of what the other detectors report.

The reasoning is a cost asymmetry, not a tuning preference: a false positive on
a weed costs a little fluid, while a false positive on a crop damages the plant
the robot exists to protect. Encoding that asymmetry into the decision rule
rather than into threshold tuning makes crop protection a structural property
of the system - it cannot be undone by someone lowering a confidence threshold
in the field.

On top of the per-frame rule sits a small multi-object tracker. Acting on a
single frame is how a system sprays a glint of sunlight; a target must persist
for ``confirm_frames`` consecutive frames before the mission state machine will
interrupt travel for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import BBox, Detection, DetectionSource, TargetClass, Track
from ..utils.logging_setup import get_logger

__all__ = ["FusionDecision", "PerceptionFusion", "TargetTracker"]

log = get_logger("vision.fusion")


@dataclass
class FusionDecision:
    """Per-frame fusion output.

    ``actionable`` holds weed detections that survived the crop veto;
    ``vetoed`` holds those that did not, with the reason recorded - which is
    what makes a "why did it not spray?" question answerable from the log.
    """

    actionable: List[Detection] = field(default_factory=list)
    vetoed: List[Tuple[Detection, str]] = field(default_factory=list)
    crops: List[Detection] = field(default_factory=list)
    all_weeds: List[Detection] = field(default_factory=list)

    @property
    def n_actionable(self) -> int:
        return len(self.actionable)

    def to_dict(self) -> dict:
        return {
            "actionable": [d.to_dict() for d in self.actionable],
            "vetoed": [{"det": d.to_dict(), "reason": r} for d, r in self.vetoed],
            "n_crops": len(self.crops),
            "n_weeds_raw": len(self.all_weeds),
        }


@dataclass
class PerceptionFusion:
    """Combines detections from the colour, learned and zero-shot tiers."""

    yolo_trigger_conf: float = 0.55
    crop_veto_conf: float = 0.35
    crop_veto_iou: float = 0.05
    crop_veto_radius_px: float = 60.0
    merge_iou: float = 0.55

    @classmethod
    def from_config(cls, cfg) -> "PerceptionFusion":
        """Build from the ``perception.fusion`` config section."""
        return cls(
            yolo_trigger_conf=cfg.get("yolo_trigger_conf", 0.55),
            crop_veto_conf=cfg.get("crop_veto_conf", 0.35),
            crop_veto_iou=cfg.get("crop_veto_iou", 0.05),
            crop_veto_radius_px=cfg.get("crop_veto_radius_px", 60.0),
        )

    def fuse(self, *detection_sets: Sequence[Detection]) -> FusionDecision:
        """Apply the late-fusion rule to detections from any number of tiers."""
        detections: List[Detection] = [d for group in detection_sets for d in (group or [])]

        crops = [d for d in detections if d.cls is TargetClass.CROP]
        weeds = [d for d in detections if d.cls is TargetClass.WEED]

        # Crop evidence below the veto confidence is not strong enough to
        # protect, but is still recorded so the log explains the decision.
        veto_crops = [c for c in crops if c.confidence >= self.crop_veto_conf]

        triggered = [w for w in weeds if self._triggers(w)]
        merged = self._merge_overlapping(triggered)

        decision = FusionDecision(crops=crops, all_weeds=weeds)
        for weed in merged:
            reason = self._veto_reason(weed, veto_crops)
            if reason is None:
                decision.actionable.append(weed)
            else:
                decision.vetoed.append((weed, reason))

        if decision.vetoed:
            log.debug("crop veto suppressed %d weed detection(s)", len(decision.vetoed))
        return decision

    # -- rule components ----------------------------------------------------
    def _triggers(self, weed: Detection) -> bool:
        """Does this weed detection, on its own, justify an action?

        The colour tier is trusted unconditionally: it has already passed the
        HSV and geometry gates, which is a harder test than a confidence
        threshold. The learned tiers must clear ``yolo_trigger_conf``.
        """
        if weed.source is DetectionSource.COLOR:
            return True
        return weed.confidence >= self.yolo_trigger_conf

    def _veto_reason(
        self, weed: Detection, crops: Sequence[Detection]
    ) -> Optional[str]:
        """Return why this weed must not be sprayed, or None if it may be."""
        for crop in crops:
            iou = weed.bbox.iou(crop.bbox)
            if iou > self.crop_veto_iou:
                return f"crop_overlap(iou={iou:.3f},src={crop.source.value})"
            distance = weed.bbox.distance_to(crop.bbox)
            if distance < self.crop_veto_radius_px:
                return f"crop_proximity(d={distance:.1f}px,src={crop.source.value})"
        return None

    def _merge_overlapping(self, weeds: Sequence[Detection]) -> List[Detection]:
        """Collapse detections of the same physical weed from different tiers.

        Without this the robot sprays twice when the colour tier and the
        learned tier both fire on one marker. The surviving detection keeps the
        higher confidence and records which tiers agreed, so that agreement
        between tiers remains visible in the log.
        """
        ordered = sorted(weeds, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for candidate in ordered:
            duplicate_of = None
            for existing in kept:
                if candidate.bbox.iou(existing.bbox) >= self.merge_iou:
                    duplicate_of = existing
                    break
            if duplicate_of is None:
                merged = Detection(
                    cls=candidate.cls,
                    bbox=candidate.bbox,
                    confidence=candidate.confidence,
                    source=candidate.source,
                    area_px=candidate.area_px,
                    meta=dict(candidate.meta),
                )
                merged.meta["sources"] = [candidate.source.value]
                kept.append(merged)
            else:
                sources = duplicate_of.meta.setdefault("sources", [duplicate_of.source.value])
                if candidate.source.value not in sources:
                    sources.append(candidate.source.value)
                # Agreement between independent tiers is stronger evidence than
                # either alone, so mark the surviving detection as fused.
                duplicate_of.source = DetectionSource.FUSED
        return kept


class TargetTracker:
    """Greedy nearest-centroid tracker over fused weed detections.

    Deliberately simple: the arena presents a handful of well-separated markers
    at walking pace, so a Hungarian assignment or a Kalman-per-track would add
    failure modes without adding capability. What it must get right is (a) not
    confirming a one-frame flicker and (b) not re-confirming a target that has
    already been sprayed.
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        max_age: int = 5,
        match_dist_px: float = 90.0,
    ):
        self.confirm_frames = int(confirm_frames)
        self.max_age = int(max_age)
        self.match_dist_px = float(match_dist_px)
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    @classmethod
    def from_config(cls, cfg) -> "TargetTracker":
        """Build from the ``perception.fusion`` config section."""
        return cls(
            confirm_frames=cfg.get("confirm_frames", 3),
            max_age=cfg.get("track_max_age", 5),
            match_dist_px=cfg.get("track_match_dist_px", 90.0),
        )

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Advance all tracks by one frame and return the live ones."""
        unmatched = list(detections)

        # Age every track first; matching resets the age to zero.
        for track in self._tracks.values():
            track.age += 1

        # Greedy match: closest pair first, so a detection cannot be stolen by
        # a track that merely happened to be iterated earlier.
        pairs: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(unmatched):
            for track_id, track in self._tracks.items():
                distance = math.hypot(
                    det.bbox.cx - track.bbox.cx, det.bbox.cy - track.bbox.cy
                )
                if distance <= self.match_dist_px:
                    pairs.append((distance, det_idx, track_id))
        pairs.sort()

        used_dets: set = set()
        used_tracks: set = set()
        for _distance, det_idx, track_id in pairs:
            if det_idx in used_dets or track_id in used_tracks:
                continue
            det = unmatched[det_idx]
            track = self._tracks[track_id]
            track.bbox = det.bbox
            track.confidence = max(track.confidence, det.confidence)
            track.source = det.source
            track.hits += 1
            track.age = 0
            used_dets.add(det_idx)
            used_tracks.add(track_id)

        for det_idx, det in enumerate(unmatched):
            if det_idx in used_dets:
                continue
            track = Track(
                track_id=self._next_id,
                cls=det.cls,
                bbox=det.bbox,
                confidence=det.confidence,
                source=det.source,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        # Drop tracks that have been missing too long. A sprayed track is kept
        # for twice as long, so that a marker briefly lost and re-acquired is
        # not treated as a fresh target and sprayed a second time.
        for track_id in [
            tid for tid, tr in self._tracks.items()
            if tr.age > (self.max_age * 2 if tr.sprayed else self.max_age)
        ]:
            del self._tracks[track_id]

        return self.tracks

    def confirmed_targets(self, include_sprayed: bool = False) -> List[Track]:
        """Live tracks that have persisted long enough to act on.

        Sorted by image row descending, so the nearest target (lowest in the
        frame, since the camera is pitched down) is engaged first.
        """
        out = [
            t for t in self._tracks.values()
            if t.age == 0
            and t.is_confirmed(self.confirm_frames)
            and (include_sprayed or not t.sprayed)
        ]
        out.sort(key=lambda t: t.bbox.cy, reverse=True)
        return out

    def mark_sprayed(self, track_id: int) -> None:
        """Record that a target has been treated, so it is never re-engaged."""
        track = self._tracks.get(track_id)
        if track is not None:
            track.sprayed = True

    def reset(self) -> None:
        """Clear all tracks - called at a row change, where continuity ends."""
        self._tracks.clear()
