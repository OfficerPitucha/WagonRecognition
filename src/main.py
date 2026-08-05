"""
Main entry point for the train detection pipeline.

Two threads run concurrently:

* :class:`TrainMonitor` watches a video feed (file or simulated live stream),
  uses motion detection to decide when a train is passing, and records each
  pass as an ``.mp4`` segment that it pushes onto a queue.
* :class:`ProcessingWorker` consumes those segments and runs the full
  :class:`src.pipeline.TrainPipeline` on each one, saving a per-train JSON
  report.

Run ``python -m src.main <video> [--output-dir DIR] [--debug]``.
"""
import json
import queue
import threading
import argparse
from pathlib import Path

import cv2

from src.pipeline import TrainPipeline
from src.debug import configure, get_logger, is_debug
from src.cloud_api import CloudAPI

log = get_logger(__name__)

# --- Configuration -----------------------------------------------------------
MOTION_THRESHOLD = 0.05       # fraction of moving pixels to consider "train present"
MOTION_TIMEOUT_SEC = 4.0      # seconds without motion before train is considered gone
BG_SUB_HISTORY = 500
BG_SUB_VAR_THRESHOLD = 20
ROI_TOP, ROI_BOTTOM = 0.5, 0.9
ROI_LEFT, ROI_RIGHT = 0.05, 0.95
PRE_MOTION_SEC = 2.0          # seconds of background frames prepended to each segment


class TrainMonitor:
    """Watches a video source, records each train pass, enqueues the segment.

    The monitor uses an MOG2 background subtractor on a central ROI to decide
    when a train is present. A segment starts on the first motion frame and
    ends after :data:`MOTION_TIMEOUT_SEC` seconds without motion; the finished
    file path is pushed to ``segment_queue``. A terminal ``None`` is enqueued
    when the video source is exhausted, signalling downstream consumers to
    exit.

    Args:
        video_path: Path to the input video file (treated as a live feed).
        segment_queue: Thread-safe queue used to hand segments to the worker.
        output_dir: Directory where recorded ``segment_N.mp4`` files are
            written. Created if missing.
    """

    def __init__(self, video_path: str, segment_queue: queue.Queue, output_dir: str = "outputs"):
        self.video_path = video_path
        self.segment_queue = segment_queue
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()

    def stop(self) -> None:
        """Request the monitor loop to exit at the next iteration."""
        self._stop.set()

    def run(self) -> None:
        """Main monitor loop — blocks until the video ends or :meth:`stop` is called."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rotate_needed = h > w
        if rotate_needed:
            w, h = h, w

        timeout_frames = int(MOTION_TIMEOUT_SEC * fps)

        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=BG_SUB_HISTORY, varThreshold=BG_SUB_VAR_THRESHOLD, detectShadows=False
        )

        roi_y1, roi_y2 = int(h * ROI_TOP), int(h * ROI_BOTTOM)
        roi_x1, roi_x2 = int(w * ROI_LEFT), int(w * ROI_RIGHT)

        train_active = False
        frames_since_motion = timeout_frames
        segment_path = None
        segment_writer = None
        segment_count = 0
        pre_motion_buffer = []
        pre_motion_maxlen = int(PRE_MOTION_SEC * fps)

        log.info("[Monitor] Opened video feed (%dx%d @ %.1f fps). Watching for trains...",
                 w, h, fps)

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            if rotate_needed:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            fg_mask = back_sub.apply(roi)
            motion_ratio = cv2.countNonZero(fg_mask) / (roi.shape[0] * roi.shape[1])

            if motion_ratio > MOTION_THRESHOLD:
                frames_since_motion = 0

                if not train_active:
                    # First motion frame — flush pre-motion buffer then start recording.
                    train_active = True
                    segment_count += 1
                    segment_path = str(self.output_dir / f"segment_{segment_count}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    segment_writer = cv2.VideoWriter(segment_path, fourcc, fps, (w, h))
                    for pre_frame in pre_motion_buffer:
                        segment_writer.write(pre_frame)
                    pre_motion_buffer = []
                    log.info("[Monitor] Train detected — recording segment %d", segment_count)

            else:
                frames_since_motion += 1

                if not train_active:
                    # Keep a rolling window of background frames to prepend when a train arrives.
                    pre_motion_buffer.append(frame.copy())
                    if len(pre_motion_buffer) > pre_motion_maxlen:
                        pre_motion_buffer.pop(0)

                if train_active and frames_since_motion >= timeout_frames:
                    # Train gone — finalise the segment and hand it off.
                    train_active = False
                    if segment_writer is not None:
                        segment_writer.release()
                        segment_writer = None
                    log.info("[Monitor] Train gone — enqueuing segment %d", segment_count)
                    self.segment_queue.put(segment_path)
                    segment_path = None

            if train_active and segment_writer is not None:
                segment_writer.write(frame)

        # If the video ended mid-train, still hand off the partial segment.
        if train_active and segment_writer is not None:
            segment_writer.release()
            log.info("[Monitor] Video ended — enqueuing final segment %d", segment_count)
            self.segment_queue.put(segment_path)

        cap.release()
        self.segment_queue.put(None)  # poison pill — signals ProcessingWorker to exit
        log.info("[Monitor] Finished.")


class ProcessingWorker:
    """Consumes segments from the queue and runs the full pipeline on each.

    Each processed segment produces a result dict (see
    :class:`src.pipeline.TrainPipeline`). Per-train results are also written to
    ``<output_dir>/train_<N>_result.json``.

    Args:
        segment_queue: Queue fed by :class:`TrainMonitor`. A ``None`` item is
            treated as a poison pill.
        output_dir: Directory for per-train JSON output and any pipeline
            by-products.
    """

    def __init__(self, segment_queue: queue.Queue, output_dir: str = "outputs"):
        self.segment_queue = segment_queue
        self.output_dir = output_dir
        self.pipeline = TrainPipeline(output_dir=output_dir, debug=is_debug())
        self.cloud_api = CloudAPI()
        self.results: list[dict] = []

    def run(self) -> None:
        """Main worker loop — pulls segments until a ``None`` item arrives."""
        log.info("[Processor] Waiting for segments...")
        train_number = 0

        while True:
            segment_path = self.segment_queue.get()
            if segment_path is None:
                break

            train_number += 1
            log.info("[Processor] Processing train %d: %s", train_number, segment_path)
            self.cloud_api.send_start_message()
            result = self.pipeline.process_train(segment_path)
            result["train_number"] = train_number
            self.results.append(result)

            self.cloud_api.send_end_message(result)
            out_path = Path(self.output_dir) / f"train_{train_number}_result.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info("[Processor] Train %d result saved to %s", train_number, out_path)

        log.info("[Processor] Finished. Processed %d train(s).", train_number)


def main() -> None:
    """CLI entry point — parses args and runs monitor + worker threads."""
    parser = argparse.ArgumentParser(description="Train detection pipeline with threaded monitoring")
    parser.add_argument("video", help="Path to video file (simulated live feed)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging and visual debug windows")
    args = parser.parse_args()

    configure(debug=args.debug)

    segment_queue: queue.Queue = queue.Queue()

    monitor = TrainMonitor(args.video, segment_queue, output_dir=args.output_dir)
    processor = ProcessingWorker(segment_queue, output_dir=args.output_dir)

    monitor_thread = threading.Thread(target=monitor.run, name="MonitorThread")
    processor_thread = threading.Thread(target=processor.run, name="ProcessorThread")

    monitor_thread.start()
    processor_thread.start()

    monitor_thread.join()
    processor_thread.join()

    log.info("All done. %d train(s) processed.", len(processor.results))
    for r in processor.results:
        log.info("  Train %s: %s wagons", r['train_number'], r['total_wagons'])


if __name__ == "__main__":
    main()
