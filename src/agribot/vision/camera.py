"""Camera capture abstraction.

One interface, three backends, so that every layer above perception is
identical whether the frames come from the arena camera, a recorded video used
for offline tuning, or the synthetic field generator used by the tests.

The live backend deliberately requests a global-shutter-friendly configuration
and does not silently accept a different resolution than the one asked for:
the pixel-to-actuator calibration is resolution-dependent, so a driver quietly
falling back to 320x240 would misaim every spray.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.logging_setup import get_logger

__all__ = ["CameraBase", "Camera", "VideoFileCamera", "FrameListCamera", "open_camera"]

log = get_logger("vision.camera")


class CameraBase(ABC):
    """Common camera interface: ``open`` / ``read`` / ``release``."""

    width: int = 0
    height: int = 0

    @abstractmethod
    def open(self) -> bool:
        """Acquire the device. Returns True on success."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab one BGR frame. Returns ``(ok, frame)``."""

    @abstractmethod
    def release(self) -> None:
        """Release the device."""

    @property
    def is_open(self) -> bool:
        return True

    def __enter__(self) -> "CameraBase":
        if not self.open():
            raise RuntimeError(f"{type(self).__name__} failed to open")
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def frames(self, limit: Optional[int] = None) -> Iterator[np.ndarray]:
        """Iterate frames until exhausted or ``limit`` reached."""
        count = 0
        while limit is None or count < limit:
            ok, frame = self.read()
            if not ok or frame is None:
                return
            count += 1
            yield frame


class Camera(CameraBase):
    """Live V4L2 / DirectShow camera."""

    def __init__(
        self,
        source: Any = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        fourcc: Optional[str] = "MJPG",
        warmup_frames: int = 5,
    ):
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.flip_horizontal = bool(flip_horizontal)
        self.flip_vertical = bool(flip_vertical)
        self.fourcc = fourcc
        self.warmup_frames = int(warmup_frames)
        self._cap: Optional[cv2.VideoCapture] = None
        self.actual_size: Tuple[int, int] = (0, 0)

    @classmethod
    def from_config(cls, cfg) -> "CameraBase":
        """Build from the ``camera`` config section.

        A string source that names an existing file yields a video-file camera,
        so the same config drives a live run and an offline replay.
        """
        source = cfg.get("source", 0)
        if isinstance(source, str) and Path(source).is_file():
            return VideoFileCamera(
                source,
                loop=False,
                flip_horizontal=cfg.get("flip_horizontal", False),
                flip_vertical=cfg.get("flip_vertical", False),
            )
        return cls(
            source=source,
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
            flip_horizontal=cfg.get("flip_horizontal", False),
            flip_vertical=cfg.get("flip_vertical", False),
        )

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            log.error("Camera source %r did not open", self.source)
            self._cap = None
            return False

        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # A small buffer keeps the frame the controller acts on close to now.
        # A backed-up buffer means steering on a frame from half a second ago.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - not supported on every backend
            pass

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_size = (actual_w, actual_h)
        if (actual_w, actual_h) != (self.width, self.height):
            log.warning(
                "Camera returned %dx%d, requested %dx%d - spray calibration "
                "assumes the requested size and will be wrong until re-run",
                actual_w, actual_h, self.width, self.height,
            )

        for _ in range(self.warmup_frames):
            self._cap.read()
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, self._apply_flips(frame)

    def _apply_flips(self, frame: np.ndarray) -> np.ndarray:
        if self.flip_horizontal and self.flip_vertical:
            return cv2.flip(frame, -1)
        if self.flip_horizontal:
            return cv2.flip(frame, 1)
        if self.flip_vertical:
            return cv2.flip(frame, 0)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileCamera(CameraBase):
    """Replay a recorded video - used for offline HSV and PID tuning."""

    def __init__(
        self,
        path: Any,
        loop: bool = False,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        realtime: bool = False,
    ):
        self.path = str(path)
        self.loop = bool(loop)
        self.flip_horizontal = bool(flip_horizontal)
        self.flip_vertical = bool(flip_vertical)
        self.realtime = bool(realtime)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_interval = 0.0
        self._last_read = 0.0

    def open(self) -> bool:
        if not Path(self.path).is_file():
            log.error("Video file not found: %s", self.path)
            return False
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / fps if fps > 0 else 0.0
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        if self.realtime and self._frame_interval > 0:
            wait = self._frame_interval - (time.monotonic() - self._last_read)
            if wait > 0:
                time.sleep(wait)
            self._last_read = time.monotonic()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            if not self.loop:
                return False, None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return False, None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.flip_vertical:
            frame = cv2.flip(frame, 0)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FrameListCamera(CameraBase):
    """In-memory frame sequence - the backend the test suite runs against."""

    def __init__(self, frames: Sequence[np.ndarray], loop: bool = False):
        self._frames = list(frames)
        self.loop = bool(loop)
        self._index = 0
        if self._frames:
            self.height, self.width = self._frames[0].shape[:2]

    def open(self) -> bool:
        self._index = 0
        return len(self._frames) > 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._frames:
            return False, None
        if self._index >= len(self._frames):
            if not self.loop:
                return False, None
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def release(self) -> None:
        self._index = 0

    def __len__(self) -> int:
        return len(self._frames)


def open_camera(cfg) -> CameraBase:
    """Build and open the camera described by the ``camera`` config section.

    Raises RuntimeError if the device cannot be opened - there is no safe way
    to run a vision-guided mission without a camera, so this fails loudly.
    """
    cam = Camera.from_config(cfg)
    if not cam.open():
        raise RuntimeError(f"Unable to open camera source {cfg.get('source', 0)!r}")
    return cam
