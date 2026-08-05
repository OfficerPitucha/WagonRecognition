"""
Low-level OCR wrapper around PaddleOCR.

Exposes :class:`OCR_2`, which:

* initialises PaddleOCR once and keeps the model alive,
* runs detection + recognition on a single image,
* extracts four structured fields from the OCR output using regex + spatial
  reasoning: UIC wagon number, Kemler plate (GEVI/UN), container ID, and
  approximate wagon length.

The ``if __name__ == "__main__"`` block at the bottom is a standalone test
harness that walks a folder of cropped frames and prints what was extracted;
it is not used by the pipeline.
"""
import os
import time

# Pip-installed CUDA wheels put their DLLs under site-packages/nvidia/<pkg>/bin,
# which PaddlePaddle does not add to PATH automatically on Windows. Add them
# manually before importing paddle.
def _add_nvidia_dlls_to_path() -> None:
    """Prepend every ``site-packages/nvidia/*/bin`` directory to PATH."""
    import site
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)  # Windows-specific, Python 3.8+
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

_add_nvidia_dlls_to_path()
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
from itertools import permutations

import cv2
import numpy as np
from paddleocr import TextRecognition, PaddleOCR

from src.debug import get_logger

log = get_logger(__name__)

# Paths used only by the standalone __main__ harness below.
FRAME_PATH = r'../../data/testing_data/detection_extraction/'
FRAMES_OCR = r'../../data/testing_data/visualization/'


def draw_ocr(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw OCR polygons + labels on a frame (used by the test harness).

    Args:
        frame: BGR image to draw on (modified in place and also returned).
        result: A PaddleOCR result dict (keys ``rec_texts``, ``rec_polys``,
            ``rec_scores``).

    Returns:
        The annotated frame.
    """
    GREEN = (0, 255, 0)
    for det in range(len(result["rec_texts"])):
        box = result["rec_polys"][det]
        text = result["rec_texts"][det]
        score = result["rec_scores"][det]
        pts = np.array(box, dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)

        cv2.polylines(frame, [pts], isClosed=True, color=GREEN, thickness=2)
        label = f"{text} ({score:.2f})"
        x, y = pts[0][0]
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    return frame


class OCR_2:
    """PaddleOCR wrapper with per-image field extractors.

    There are two ways to construct this class:

    * ``OCR_2(None, None, text_only=False).get_model()`` — pass ``image_path=None``
      to skip any OCR work and just instantiate the model for later reuse.
    * ``OCR_2(image, shared_model, text_only=False)`` — pass the BGR numpy
      array for ``image_path`` plus a model returned by ``get_model()``; the
      extractor methods will run on that image.

    Args:
        image_path: A BGR numpy array, or ``None`` to skip OCR (model-only
            construction). Named ``image_path`` for historical reasons.
        model: A pre-built PaddleOCR / TextRecognition model to reuse. If
            ``None``, a fresh model is constructed.
        text_only: If ``True``, load only the recognition model (no detector)
            — useful when input images are already cropped to text regions.
    """

    def __init__(self, image_path, model=None, text_only: bool = False):
        self.base_result = None
        self.image_path = image_path
        self.text_only = text_only

        if model is not None:
            self.model = model
            return

        if text_only:
            self.model = TextRecognition(
                model_name="PP-OCRv5_server_rec",
                device="gpu",
            )
        else:
            self.model = PaddleOCR(
                lang='en',
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_textline_orientation=False,
            )

    def base_extraction(self) -> dict:
        """Run PaddleOCR once and return the raw result dict.

        The result contains parallel lists: ``rec_texts`` (strings),
        ``rec_polys`` (quadrilateral corners) and ``rec_scores`` (floats).
        """
        result = self.model.predict(
            self.image_path,
            text_det_limit_side_len=1980,
            text_det_limit_type="max",
            text_det_thresh=0.70,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.2,
            text_rec_score_thresh=0.70,
            return_word_box=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        return result[0]

    def extract_uic(self) -> list[int]:
        """Return every UIC number candidate found in the image.

        Uses a 12-digit pattern (optionally split by space/dash). If the full
        12-digit number is present in a single token it is returned directly;
        otherwise the method looks for partial matches (7 or 8 digits) and
        tries to stitch them together with 2- or 4-digit tokens found in the
        neighbourhood of the main box.

        Returns:
            A list of candidate integers. May contain duplicates — the
            caller is expected to apply majority voting.
        """
        PATTERN = re.compile(r'(\d\d ?\d\d ?)?(\d{4} ?\d{3}[- ]?\d)')
        if self.base_result is None:
            self.base_result = self.base_extraction()

        potential_candidates: list[int] = []
        for i in range(len(self.base_result["rec_texts"])):
            box_main = self.base_result["rec_polys"][i]
            text_main = self.base_result["rec_texts"][i]

            if PATTERN.fullmatch(text_main) is None:
                continue
            log.debug("UIC raw match: %s", text_main)
            text_main = re.sub(r"[ -]", "", text_main)

            # CASE 1 — full 12-digit UIC in a single token.
            if len(text_main) == 12:
                return [int(text_main)]

            nearest_texts, _ = self._find_nearest(box_main, 3, 1)
            log.debug("UIC neighbours: %s", nearest_texts)

            nearest_texts = [re.fullmatch(r"\d\d ?(\d\d)?\D*", t) for t in nearest_texts]
            nearest_texts = [re.sub(r"\D", "", t[0]) for t in nearest_texts if t is not None]
            log.debug("UIC neighbours cleaned: %s", nearest_texts)

            # CASE 2 — a 4-digit neighbour prefixed to text_main.
            for text in nearest_texts:
                if re.fullmatch(r"\d\d\d\d", text) is not None:
                    potential_candidates.append(int(text + text_main))

            # CASE 3 — two 2-digit neighbours prefixed in every permutation.
            nearest_2digits = [t for t in nearest_texts if len(t) == 2]
            if len(nearest_2digits) > 1:
                potential_candidates.extend(
                    int(''.join(p) + text_main) for p in permutations(nearest_2digits, 2)
                )
            # CASE 4 — some UIC digits missing: nothing to do.

        return potential_candidates

    def extract_kemler_plate(self) -> list[tuple]:
        """Return every ``(GEVI, UN)`` Kemler plate candidate in the image.

        A Kemler plate has a two/three-digit GEVI code on top and a four-digit
        UN number below. This method finds UN candidates, then looks upward
        for a spatially overlapping GEVI.
        """
        GEVI_PATTERN = re.compile(r'X?[2-9]\d{1,2}')
        UN_PATTERN = re.compile(r'\d{4}')

        self.base_result = self.base_extraction()
        potential_candidates: list[tuple] = []

        for i in range(len(self.base_result["rec_texts"])):
            box_main = self.base_result["rec_polys"][i]
            text_main = re.sub(r'\s', '', self.base_result["rec_texts"][i])

            is_gevi = GEVI_PATTERN.fullmatch(text_main) is not None
            is_un = UN_PATTERN.fullmatch(text_main) is not None and (4 <= int(text_main) <= 3999)
            if not is_gevi and not is_un:
                continue

            log.debug("Kemler candidate token: %s", text_main)

            ul, _, _, bl = box_main
            x_left = min(p[0] for p in box_main)
            x_right = max(p[0] for p in box_main)
            box_width = x_right - x_left
            box_height = abs(bl[1] - ul[1])

            for j in range(len(self.base_result["rec_texts"])):
                if j == i:
                    continue
                text_j = re.sub(r'\s', '', self.base_result["rec_texts"][j])
                box_j = self.base_result["rec_polys"][j]

                x_left_j = min(p[0] for p in box_j)
                x_right_j = max(p[0] for p in box_j)
                overlap_x = min(x_right, x_right_j) - max(x_left, x_left_j)
                min_width = min(box_width, x_right_j - x_left_j)
                if overlap_x < 0.5 * min_width:
                    continue

                if is_un:
                    # GEVI must sit directly above the UN number.
                    y_min_search = ul[1] - 2 * box_height
                    y_max_search = ul[1]
                    if GEVI_PATTERN.fullmatch(text_j):
                        if any(y_min_search < y < y_max_search for _, y in box_j):
                            log.debug("GEVI above UN: %s / %s", text_j, text_main)
                            potential_candidates.append((text_j, text_main))

        return potential_candidates

    def extract_length(self) -> list[float]:
        """Return every wagon-length candidate (metres) found in the image.

        Accepts tokens of the form ``NN.NNN?m`` with the numeric value between
        ``MIN_LENGTH`` and ``MAX_LENGTH`` metres.
        """
        MIN_LENGTH = 8
        MAX_LENGTH = 30
        PATTERN = re.compile(r'[123]?\d\.\d\d\d?m')

        if self.base_result is None:
            self.base_result = self.base_extraction()

        potential_candidates: list[float] = []
        for text in self.base_result['rec_texts']:
            length = PATTERN.search(text)
            if length is not None and (MIN_LENGTH < float(length[0][:-1]) < MAX_LENGTH):
                potential_candidates.append(float(length[0][:-1]))
        return potential_candidates

    def extract_container_code(self) -> list[str]:
        """Return every ISO 6346 container ID candidate in the image."""
        PATTERN = re.compile(r'([A-Z]{4})? ?(\d{3} ?\d{3}) ?\d?')
        if self.base_result is None:
            self.base_result = self.base_extraction()

        potential_candidates: list[str] = []
        for i in range(len(self.base_result['rec_texts'])):
            text = self.base_result['rec_texts'][i]
            box_main = self.base_result["rec_polys"][i]
            if PATTERN.fullmatch(text) is None:
                continue
            container_code = text.replace(' ', '')
            if len(re.search(r"\d+", container_code)[0]) == 6:
                container_code = container_code + "?"

            if len(container_code) == 11:
                potential_candidates.append(container_code)
                continue

            nearest_texts, _ = self._find_nearest(box_main, 2, 1.5)
            potential_candidates.extend(
                f"{t}{container_code}" for t in nearest_texts if re.fullmatch(r"[A-Z]{4}", t)
            )

        return potential_candidates

    def get_model(self):
        """Return the underlying PaddleOCR / TextRecognition model."""
        return self.model

    def _find_nearest(
        self,
        main_box,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
    ) -> tuple[list, list]:
        """Find OCR boxes inside a rectangle placed above-and-right of ``main_box``.

        The search rectangle is sized relative to ``main_box`` (``width_scale``
        times the max side for width, ``height_scale`` for height) so the
        method scales with text size.

        Returns:
            ``(texts, box_strings)`` — parallel lists of matched texts and
            string-serialised polygons.
        """
        ul, ur, br, bl = main_box
        box_width = br[0] - bl[0]
        box_height = ul[1] - bl[1]
        scale_unit = max(box_width, box_height)
        x_min, y_max = br[0] - width_scale * scale_unit, bl[1] - height_scale * scale_unit

        nearest_boxes = []
        nearest_texts = []
        for j in range(len(self.base_result["rec_texts"])):
            text_j = self.base_result["rec_texts"][j]
            box_j = self.base_result["rec_polys"][j]
            if np.array_equal(box_j, main_box):
                continue
            if any((x_min < x < br[0]) and (y_max < y < br[1]) for x, y in box_j):
                nearest_boxes.append(np.array2string(box_j, threshold=np.inf))
                nearest_texts.append(text_j)

        return nearest_texts, nearest_boxes


if __name__ == '__main__':
    from src.debug import configure
    configure(debug=False)

    model = OCR_2(None, None, text_only=False).get_model()
    for i in range(255):
        image_path = f"{FRAME_PATH}frame_{i}.png"
        image = cv2.imread(image_path)
        screen = OCR_2(image, model, text_only=False)

        t = time.time()
        uic = screen.extract_uic()
        kemler = screen.extract_kemler_plate()
        container = screen.extract_container_code()
        length = screen.extract_length()

        if len(uic + kemler + container + length) == 0:
            continue

        log.info("%d. UIC: %s", i, uic)
        log.info("%d. KEMLER: %s", i, kemler)
        log.info("%d. CONTAINER ID: %s", i, container)
        log.info("%d. LENGTH: %s", i, length)
        log.info("elapsed: %.2fs", time.time() - t)
        log.info("-" * 60)
