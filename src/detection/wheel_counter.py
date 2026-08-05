"""
Wheel counter using DeepSORT tracking + line-crossing detection.

Accepts frames directly (no video-file dependency) so the pipeline can hand
in the raw buffer produced by :class:`src.detection.wagon_segmenter.WagonSegmenter`.
"""
import os

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.debug import get_logger, is_debug

log = get_logger(__name__)


class WheelCounter:
    """Counts wheels in a sequence of frames via line-crossing tracking.

    The counter crops a horizontal band at the centre of each frame (the
    "zone"), runs a YOLO wheel detector on the crop, feeds the detections
    into DeepSORT, and increments the counter every time a confirmed track
    crosses the vertical midline of the zone.

    Args:
        model_path: Path to the wheel-detector YOLO weights. Defaults to
            ``wheel_detection_best.pt`` next to this module.
        conf: YOLO confidence threshold.
        iou: YOLO IoU threshold.
        zone_width: Width of the central crop (in pixels). The zone is always
            centred horizontally and covers the bottom 55% of the frame.
        max_age: Frames a DeepSORT track survives without being re-detected.
        min_box_size: Reject any detection whose width or height is below
            this many pixels (filters out distant/partial wheels).
        debug_display: If ``True`` (or when global debug mode is on), opens
            an OpenCV window with annotated overlays.
    """

    def __init__(
        self,
        model_path: str = None,
        conf: float = 0.7,
        iou: float = 0.4,
        zone_width: int = 1000,
        max_age: int = 15,
        min_box_size: int = 50,
        debug_display: bool = False,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "wheel_detection_best.pt")

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.zone_width = zone_width
        self.max_age = max_age
        self.min_box_size = min_box_size
        self.debug_display = debug_display

    def _setup_zone(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Return ``(x1, x2, y1, y2)`` of the central counting zone."""
        cx = frame_w // 2
        x1 = max(0, cx - self.zone_width // 2)
        x2 = min(frame_w, cx + self.zone_width // 2)
        y1 = int(frame_h * 0.45)
        y2 = frame_h
        return x1, x2, y1, y2

    def _detect(self, zone_crop: np.ndarray) -> list[tuple]:
        """Run YOLO on the zone crop and convert the output to DeepSORT tuples.

        Returns:
            List of ``([x1, y1, x2, y2], confidence, "wheels")`` tuples, already
            filtered by confidence and minimum box size.
        """
        results = self.model(zone_crop, verbose=False, iou=self.iou)[0]
        detections = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if cls_name != "wheels":
                continue
            conf = float(box.conf[0])
            if conf < self.conf:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                continue
            detections.append(([float(x1), float(y1), float(x2), float(y2)], conf, "wheels"))
        return detections

    @staticmethod
    def _annotate(frame, zone_x1, zone_x2, zone_y1, zone_y2,
                  line_abs_x, tracks, crossed_ids, wheel_count):
        """Return a copy of ``frame`` with the zone, counting line, tracks and counter drawn on."""
        vis = frame.copy()
        cv2.rectangle(vis, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
        cv2.line(vis, (line_abs_x, zone_y1), (line_abs_x, zone_y2), (0, 0, 255), 2)
        for t in tracks:
            if not t.is_confirmed():
                continue
            bx1, by1, bx2, by2 = [int(v) for v in t.to_ltrb()]
            bx1 += zone_x1; bx2 += zone_x1
            by1 += zone_y1; by2 += zone_y1
            tid = t.track_id
            color = (128, 128, 128) if tid in crossed_ids else (0, 255, 0)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(vis, f"ID:{tid}", (bx1, max(by1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(vis, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(vis, f"Wheels counted: {wheel_count}", (18, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return vis

    def count_wheels(self, frames: list[np.ndarray]) -> int:
        """Count wheels via line-crossing tracking across a frame sequence.

        Args:
            frames: BGR numpy arrays with temporal continuity. The sequence
                must be in capture order — DeepSORT relies on that.

        Returns:
            Total number of unique track IDs that crossed the midline.
        """
        if not frames:
            log.warning("[Wheels] No frames supplied; skipping wheel counting.")
            return 0

        log.debug("[Wheels] Processing %d frames...", len(frames))
        frame_h, frame_w = frames[0].shape[:2]
        zone_x1, zone_x2, zone_y1, zone_y2 = self._setup_zone(frame_w, frame_h)
        zone_mid_x = (zone_x2 - zone_x1) // 2
        line_abs_x = zone_x1 + zone_mid_x

        tracker = DeepSort(max_age=self.max_age, n_init=3)
        prev_positions: dict[int, float] = {}
        crossed_ids: set[int] = set()
        wheel_count = 0
        debug_window = self.debug_display or is_debug()

        for frame in frames:
            zone_crop = frame[zone_y1:zone_y2, zone_x1:zone_x2]
            detections = self._detect(zone_crop)
            tracks = tracker.update_tracks(detections, frame=zone_crop)

            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                bx1, _, bx2, _ = t.to_ltrb()
                cx = (bx1 + bx2) / 2

                if tid not in crossed_ids and tid in prev_positions:
                    prev_cx = prev_positions[tid]
                    if prev_cx < zone_mid_x <= cx or prev_cx > zone_mid_x >= cx:
                        wheel_count += 1
                        crossed_ids.add(tid)
                        log.debug("Wheel ID=%s crossed → total=%d", tid, wheel_count)

                prev_positions[tid] = cx

            if debug_window:
                vis = self._annotate(frame, zone_x1, zone_x2, zone_y1, zone_y2,
                                     line_abs_x, tracks, crossed_ids, wheel_count)
                display = cv2.resize(vis, (1280, 720))
                cv2.imshow("Wheel Counter Debug", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if debug_window:
            cv2.destroyWindow("Wheel Counter Debug")

        return wheel_count
