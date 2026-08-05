"""
OCR extractor that works on in-memory frames (numpy arrays).

Wraps :class:`src.postprocessing.ocr.OCR_2` and performs majority voting
across frames so transient mis-reads cannot win.
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from collections import Counter

import numpy as np

from src.postprocessing.ocr import OCR_2
from src.debug import get_logger

log = get_logger(__name__)


class OCRExtractor:
    """One-shot extractor that loads PaddleOCR once and votes across frames.

    After construction the model is cached on this instance — call
    :meth:`extract_info` once per wagon.
    """

    def __init__(self):
        log.info("[OCR] Loading PaddleOCR model...")
        self._model = OCR_2(None, None, text_only=False).get_model()
        self._elements = ["uic", "kemler", "container", "length"]

    def extract_info(self, frames: list[np.ndarray]) -> dict:
        """Run OCR on every frame and return the majority-vote winner per field.

        Args:
            frames: List of BGR numpy arrays from one wagon.

        Returns:
            A dict keyed by field name (``"uic"``, ``"kemler"``, ``"container"``,
            ``"length"``). Each value is a ``(winner, confidence, details)``
            triple:

            * ``winner`` — best candidate (int / tuple / str / float),
              or ``None`` if no candidate was found.
            * ``confidence`` — ``winner_votes / total_votes`` in ``[0.0, 1.0]``.
            * ``details`` — ``{"per_frame": {...}, "vote_tally": {...}}``
              for debugging or downstream inspection.

            If ``frames`` is empty an empty dict is returned.
        """
        if not frames:
            log.warning("[OCR] No frames supplied; skipping extraction.")
            return {}

        log.info("[OCR] Running OCR on %d frame(s)...", len(frames))

        all_candidates = {element: [] for element in self._elements}
        per_frame_results = {element: {} for element in self._elements}

        for i, frame in enumerate(frames):
            log.debug("[OCR] Frame %d/%d", i + 1, len(frames))
            frame_ocr = OCR_2(frame, self._model, text_only=False)

            candidates_uic = frame_ocr.extract_uic()
            candidates_kemler = frame_ocr.extract_kemler_plate()
            candidates_container = frame_ocr.extract_container_code()
            candidates_length = frame_ocr.extract_length()

            per_frame_results["uic"][f"frame_{i:02d}"] = candidates_uic
            per_frame_results["kemler"][f"frame_{i:02d}"] = candidates_kemler
            per_frame_results["container"][f"frame_{i:02d}"] = candidates_container
            per_frame_results["length"][f"frame_{i:02d}"] = candidates_length

            all_candidates["uic"].extend(candidates_uic)
            all_candidates["kemler"].extend(candidates_kemler)
            all_candidates["container"].extend(candidates_container)
            all_candidates["length"].extend(candidates_length)

        log.info("[OCR] Tallying votes across %d frame(s)...", len(frames))
        result: dict = {element: None for element in self._elements}
        for element in result:
            if not all_candidates[element]:
                result[element] = (None, 0.0, {"per_frame": per_frame_results[element], "vote_tally": {}})
                log.debug("[OCR] %s: no candidates.", element)
                continue

            vote_tally = Counter(all_candidates[element])
            winner, winner_count = vote_tally.most_common(1)[0]
            confidence = winner_count / len(all_candidates[element])

            details = {
                "per_frame": per_frame_results[element],
                "vote_tally": {str(k): v for k, v in vote_tally.most_common()},
            }
            result[element] = (winner, confidence, details)
            log.debug("[OCR] %s: winner=%s (conf=%.2f)", element, winner, confidence)

        return result
