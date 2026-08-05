"""
Wagon segmenter — splits a train video into per-wagon frame buffers.

Uses a YOLO coupler detector to locate inter-wagon junctions and a background
subtractor to decide when the train is actually moving. Each wagon is yielded
as a :class:`WagonResult` containing:

* ``frames``        – evenly-spaced frames for OCR,
* ``all_frames``    – every "clean" frame (no coupler in view),
* ``raw_frames``    – every frame while the train was moving (for wheel counting),
* ``total_buffered`` – size of ``all_frames`` before sampling.
"""
import os
from dataclasses import dataclass
from typing import Iterator

import cv2
_imshow = cv2.imshow  # ultralytics monkey-patches cv2.imshow on import — save the original first
import supervision as sv
from ultralytics import YOLO
cv2.imshow = _imshow  # restore the real imshow so debug windows work as expected
import numpy as np

from src.debug import get_logger, is_debug

log = get_logger(__name__)


@dataclass
class WagonResult:
    """Output of :meth:`WagonSegmenter.process_video` for one wagon.

    Attributes:
        wagon_id: 1-based index within the current video.
        frames: ``target_frames_per_wagon`` evenly-spaced clean frames,
            suitable for OCR.
        all_frames: every clean frame captured for this wagon (no coupler /
            cooldown frames).
        raw_frames: every frame while the train was moving, including coupler
            frames — used by the wheel counter for tracking continuity.
        total_buffered: ``len(all_frames)`` before sampling — useful for
            diagnostics.
    """
    wagon_id: int
    frames: list
    all_frames: list
    raw_frames: list
    total_buffered: int


class WagonSegmenter:
    """Streams wagons out of a train video one at a time.

    Args:
        model_path: Path to the YOLO coupler-detector weights. Defaults to
            ``best.pt`` in this module's directory.
        roi_top, roi_bottom, roi_left, roi_right: ROI bounds (fractions of
            frame size) — only pixels inside this rectangle are used for both
            motion detection and coupler detection.
        target_frames_per_wagon: Number of OCR frames to sample per wagon.
        min_edge_padding, max_edge_padding: Minimum/maximum number of frames
            to skip after a coupler leaves the view, adjusted dynamically
            based on train speed.
        motion_threshold: Fraction of ROI pixels that must change between
            frames for the train to be considered "moving".
        trigger_zone_left, trigger_zone_right: Fractions of frame width
            defining the central zone; a coupler entering this zone flushes
            the current wagon buffer and yields a :class:`WagonResult`.
        debug_display: If ``True`` (or when global debug mode is on), opens
            an OpenCV window with annotated overlays.
    """

    def __init__(
        self,
        model_path: str = None,
        roi_top: float = 0.5,
        roi_bottom: float = 0.9,
        roi_left: float = 0.05,
        roi_right: float = 0.95,
        target_frames_per_wagon: int = 20,
        min_edge_padding: int = 3,
        max_edge_padding: int = 20,
        motion_threshold: float = 0.05,
        trigger_zone_left: float = 0.35,
        trigger_zone_right: float = 0.65,
        debug_display: bool = False,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")

        import torch
        self.model = YOLO(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right
        self.target_frames = target_frames_per_wagon
        self.min_edge_padding = min_edge_padding
        self.max_edge_padding = max_edge_padding
        self.motion_threshold = motion_threshold
        self.trigger_zone_left = trigger_zone_left
        self.trigger_zone_right = trigger_zone_right
        self.debug_display = debug_display

    def process_video(self, video_path: str) -> Iterator[WagonResult]:
        """Iterate over :class:`WagonResult` objects, one per detected wagon.

        Args:
            video_path: Path to the train-pass video.

        Yields:
            :class:`WagonResult` for every wagon with enough clean frames to
            be considered valid.
        """
        log.info("[Segmenter] Opening %s", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate_needed = h > w
        if rotate_needed:
            w, h = h, w

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        wait_ms = max(1, int(1000 / fps))

        tracker = sv.ByteTrack()
        zone_x1 = int(w * self.trigger_zone_left)
        zone_x2 = int(w * self.trigger_zone_right)
        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=20, detectShadows=False
        )

        roi_y1, roi_y2 = int(h * self.roi_top), int(h * self.roi_bottom)
        roi_x1, roi_x2 = int(w * self.roi_left), int(w * self.roi_right)

        current_buffer = []
        raw_buffer = []
        wagon_count = 0
        cooldown = 0
        train_detected = False
        frame_count = 0
        coupler_in_zone = False
        zone_triggered = False
        smoothed_motion = self.motion_threshold  # warm-start at threshold
        debug_window = self.debug_display or is_debug()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if rotate_needed:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            fg_mask = back_sub.apply(roi)
            motion_ratio = cv2.countNonZero(fg_mask) / (roi.shape[0] * roi.shape[1])
            smoothed_motion = 0.9 * smoothed_motion + 0.1 * motion_ratio
            train_moving = motion_ratio > self.motion_threshold

            if train_moving and not train_detected:
                train_detected = True
                log.info("[Segmenter] Train motion detected at frame %d (%.2fs)",
                         frame_count, frame_count / fps)

            results = self.model(roi, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            for i in range(len(detections.xyxy)):
                detections.xyxy[i] += [roi_x1, roi_y1, roi_x1, roi_y1]
            detections = tracker.update_with_detections(detections)

            # Padding scales inversely with train speed — slower trains need more clearance.
            if len(detections) > 0:
                speed = max(smoothed_motion, 1e-3)
                dynamic_padding = int(np.clip(self.min_edge_padding / speed, self.min_edge_padding, self.max_edge_padding))
                cooldown = dynamic_padding
                is_clean = False
            else:
                if cooldown > 0:
                    cooldown -= 1
                is_clean = train_moving and (cooldown == 0)

            if train_moving:
                raw_buffer.append(frame.copy())

            if is_clean:
                current_buffer.append(frame.copy())

            # Transition trigger — a coupler entering the centre zone ends the current wagon.
            boxes = detections.xyxy if len(detections) > 0 else []
            coupler_now_in_zone = any(
                zone_x1 <= (x1 + x2) / 2 <= zone_x2
                for x1, _, x2, _ in boxes
            )

            if coupler_now_in_zone and not coupler_in_zone and not zone_triggered:
                zone_triggered = True
                if len(current_buffer) > 0:
                    wagon_count += 1
                    log.info("[Segmenter] Wagon %d boundary detected (%d clean frames buffered)",
                             wagon_count, len(current_buffer))
                    wagon = self._finalize(current_buffer, raw_buffer.copy(), wagon_count)
                    if wagon is not None:
                        yield wagon
                    current_buffer = []
                    raw_buffer = []

            if not coupler_now_in_zone:
                zone_triggered = False  # reset so the next coupler can trigger again

            coupler_in_zone = coupler_now_in_zone

            if debug_window:
                annotated = frame.copy()
                color = (0, 255, 0) if is_clean else (0, 0, 255)
                text = "RECORDING BODY" if is_clean else "JUNCTION/PADDING"
                cv2.putText(annotated, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
                cv2.rectangle(annotated, (zone_x1, 0), (zone_x2, h), (0, 165, 255), 1)
                display = cv2.resize(annotated, (1280, 720))
                cv2.imshow("Wagon Segmenter", display)
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break

        # Flush the trailing wagon if the video ends mid-wagon.
        if len(current_buffer) > 0:
            wagon_count += 1
            wagon = self._finalize(current_buffer, raw_buffer.copy(), wagon_count)
            if wagon is not None:
                yield wagon

        cap.release()
        if debug_window:
            cv2.destroyAllWindows()

        log.info("[Segmenter] Segmentation complete — %d wagon(s) emitted.", wagon_count)

    def _finalize(self, buffer: list, raw_buffer: list, wagon_idx: int) -> WagonResult | None:
        """Build a :class:`WagonResult` from a clean-frame buffer.

        Returns ``None`` and logs a warning if the buffer is shorter than
        ``target_frames`` (not enough frames to sample meaningfully).
        """
        if len(buffer) < self.target_frames:
            log.warning("Skipping wagon %d: not enough frames (%d)", wagon_idx, len(buffer))
            return None

        idx = np.linspace(0, len(buffer) - 1, self.target_frames, dtype=int)
        selected = [buffer[i] for i in idx]

        return WagonResult(
            wagon_id=wagon_idx,
            frames=selected,
            all_frames=buffer,
            raw_frames=raw_buffer,
            total_buffered=len(buffer),
        )
