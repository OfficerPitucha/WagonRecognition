"""
Train processing pipeline.

Orchestrates the per-train stages:

1. Wagon segmentation — splits a train video into individual wagon frame
   buffers using a coupler detector + motion cues.
2. Wheel counting — DeepSORT line-crossing on the full frame buffer.
3. GHS dangerous-goods detection — GroundingDINO + CLIP on sampled frames.
4. OCR extraction — UIC, Kemler plate, container ID and approximate wagon
   length via PaddleOCR + majority voting.
5. Result assembly — a single JSON document describing the whole train.
"""
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant")

# PaddleOCR must be imported before PyTorch to avoid cuDNN DLL conflicts on Windows.
from src.postprocessing.ocr_extractor import OCRExtractor
from src.postprocessing.json_forming import validate_uic, validate_container_id

from src.detection.wagon_segmenter import WagonSegmenter
from src.detection.wheel_counter import WheelCounter
from src.detection.frame_sampler import FrameSampler
from src.ghs import GHSDetector
from src.debug import configure, get_logger

log = get_logger(__name__)


class TrainPipeline:
    """Runs every per-train stage in sequence and emits a single result dict.

    Args:
        wagon_model_path: Override for the wagon-coupler YOLO weights.
        wheel_model_path: Override for the wheel-detector YOLO weights.
        ocr_frames: Maximum evenly-spaced frames sampled from each wagon for
            OCR (more = better voting, slower).
        output_dir: Directory for the aggregated ``train_result.json``.
        debug: Propagates to every stage that has optional visual debug
            overlays (see :mod:`src.debug`).
    """

    def __init__(
        self,
        wagon_model_path: str = None,
        wheel_model_path: str = None,
        ocr_frames: int = 20,
        output_dir: str = "outputs",
        debug: bool = False,
    ):
        self.debug = debug
        log.info("[Init] Loading pipeline models...")
        log.info("[Init] Loading wagon segmenter...")
        self.segmenter = WagonSegmenter(model_path=wagon_model_path, debug_display=debug)
        log.info("[Init] Loading wheel counter...")
        self.wheel_counter = WheelCounter(model_path=wheel_model_path, debug_display=debug)
        self.frame_sampler = FrameSampler(n_frames=ocr_frames)
        log.info("[Init] Loading OCR extractor...")
        self.ocr = OCRExtractor()
        log.info("[Init] Loading GHS dangerous-goods detector...")
        try:
            self.ghs_detector = GHSDetector(svm_path="")
        except Exception as e:
            log.warning("[Init] GHS detector failed to load, dangerous goods detection disabled: %s", e)
            self.ghs_detector = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("[Init] Pipeline ready.")

    def process_train(self, video_path: str) -> dict:
        """Run the full pipeline on one video.

        Args:
            video_path: Path to an ``.mp4`` showing a single train pass.

        Returns:
            A dict with keys ``timestamp``, ``video``, ``total_wagons`` and
            ``wagons`` (a list of per-wagon result dicts).
        """
        log.info("=" * 60)
        log.info("[Pipeline] Starting: %s", video_path)
        log.info("=" * 60)
        log.info("[Stage 1/3] Segmenting wagons from video...")

        wagon_results = []
        for wagon in self.segmenter.process_video(video_path):
            log.info("-" * 60)
            log.info("[Stage 2/3] Wagon %d — running per-wagon analysis", wagon.wagon_id)

            # Wheel counting needs the full (raw) buffer so DeepSORT tracking stays continuous.
            log.info("  [Wheels] Counting wheels (%d raw frames)...", len(wagon.raw_frames))
            wheel_count = self.wheel_counter.count_wheels(wagon.raw_frames)
            log.info("  [Wheels] Result: %d wheels", wheel_count)

            ocr_frames = wagon.frames
            log.info("  [Frames] %d OCR frames sampled (from %d buffered)",
                     len(ocr_frames), wagon.total_buffered)

            log.info("  [GHS] Scanning for dangerous-goods placards...")
            dangerous_goods = self.ghs_detector.detect(wagon.frames) if self.ghs_detector else []
            log.info("  [GHS] Result: %s",
                     ", ".join(dangerous_goods) if dangerous_goods else "none detected")

            log.info("[Stage 3/3] Wagon %d — OCR extraction (UIC / Kemler / container / length)", wagon.wagon_id)
            general_ocr_extraction = self.ocr.extract_info(ocr_frames)

            uic, uic_confidence, _ = general_ocr_extraction["uic"]
            uic_str = str(uic) if uic is not None else ""
            uic_valid = validate_uic(uic_str) if uic_str else False

            kemler, kemler_confidence, _ = general_ocr_extraction["kemler"]
            gevi_code, un_code = kemler if kemler is not None else (None, None)
            kemler_valid = bool(kemler)  # already validated upstream

            container, container_confidence, _ = general_ocr_extraction["container"]
            container_valid = validate_container_id(container) if container is not None else False

            length, length_confidence, _ = general_ocr_extraction["length"]

            log.info("  [OCR] UIC=%s (conf=%.2f, valid=%s) | Kemler=%s (valid=%s) | Container=%s (valid=%s) | Length=%s",
                     uic_str or "-", uic_confidence, uic_valid,
                     f"{gevi_code}/{un_code}" if kemler else "-", kemler_valid,
                     container or "-", container_valid,
                     length if length is not None else "-")

            wagon_results.append({
                "wagon_id": wagon.wagon_id,
                "wheels": wheel_count,
                "dangerous_goods": dangerous_goods,
                "wagon_approximate_length": {
                    "value": length,
                    "confidence": length_confidence,
                },
                "uic": {
                    "detected_text": uic_str,
                    "confidence": uic_confidence,
                    "is_valid": uic_valid,
                },
                "kemler_plate": {
                    "gevi_code": gevi_code,
                    "un_code": un_code,
                    "confidence": kemler_confidence,
                    "is_valid": kemler_valid,
                },
                "container_id": {
                    "detected_text": container,
                    "confidence": container_confidence,
                    "is_valid": container_valid,
                },
                "total_frames_captured": wagon.total_buffered,
            })

        if not wagon_results:
            log.warning("[Pipeline] No wagons detected in %s", video_path)
            return self._build_result(video_path, [])

        log.info("-" * 60)
        log.info("[Pipeline] Assembling final result (%d wagons)", len(wagon_results))
        result = self._build_result(video_path, wagon_results)

        output_path = self.output_dir / "train_result.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("[Pipeline] Complete. Result saved to %s", output_path)
        log.info("=" * 60)

        return result

    def _build_result(self, video_path: str, wagon_results: list) -> dict:
        """Assemble the top-level result document for one train."""
        return {
            "timestamp": datetime.now().isoformat(),
            "video": os.path.basename(video_path),
            "total_wagons": len(wagon_results),
            "wagons": wagon_results,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run train processing pipeline")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--ocr-frames", type=int, default=20, help="Max frames sampled per wagon for OCR")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging and visual debug windows")
    args = parser.parse_args()

    configure(debug=args.debug)

    pipeline = TrainPipeline(output_dir=args.output_dir, ocr_frames=args.ocr_frames, debug=args.debug)
    result = pipeline.process_train(args.video)
    print(json.dumps(result, indent=2))
