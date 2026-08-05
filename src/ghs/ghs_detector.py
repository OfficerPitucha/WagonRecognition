"""
GHS Dangerous Goods Detector
==============================
Wraps GroundingDINO + CLIP (+ optional SVM) into a single class that accepts
a list of numpy frames (BGR, from OpenCV) and returns a deduplicated list of
human-readable dangerous goods labels per wagon.

Default model paths point to C:/local_design/design-project/models/ where the
weights were found on this machine.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from src.debug import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# GHS CLASS → HUMAN-READABLE MAPPING
# ---------------------------------------------------------------------------

GHS_LABEL_MAP = {
    "flammable":        "Flammable Liquid",
    "oxidizer":         "Oxidizer",
    "toxic":            "Toxic",
    "corrosive":        "Corrosive",
    "health_hazard":    "Health Hazard",
    "environmental":    "Environmental Hazard",
    "gas_cylinder":     "Compressed Gas",
}

# ---------------------------------------------------------------------------
# GHS TEXT PROMPTS  (CLIP zero-shot)
# ---------------------------------------------------------------------------

GHS_PROMPTS = {
    "flammable": [
        "red diamond-shaped hazard placard with a bold black flame pictogram in the center",
        "ADR flammable liquid sign red diamond with black fire symbol and white inner border",
        "close-up of a small red tilted square sign showing a black flame on a white background inside",
        "GHS flammable hazard label red rhombus shape with distinct black fire pictogram",
    ],
    "oxidizer": [
        "close-up of a yellow diamond sign with flame over circle in center",
        "yellow hazard diamond with black flame above a circle symbol",
        "yellow diamond-shaped sign with flame over circle",
    ],
    "toxic": [
        "a white diamond hazard label with skull and crossbones",
        "ADR toxic substance white diamond placard close-up",
        "toxic warning sign skull crossbones on white diamond",
        "white diamond-shaped sign with skull and crossbones",
    ],
    "corrosive": [
        "white diamond GHS hazard label with black pictogram of acid dripping onto a hand and a flat surface causing damage",
        "ADR corrosive placard white diamond showing liquid eating through metal and burning a hand",
        "GHS corrosive sign white background black image of test tube pouring liquid onto hand and table",
        "hazard diamond with black symbol of two surfaces being corroded by dripping liquid from above",
    ],
    "health_hazard": [
        "a white diamond hazard label with exclamation mark",
        "health hazard white diamond placard close-up",
        "irritant warning sign white diamond exclamation point",
        "white diamond-shaped sign with black vertical stripes",
    ],
    "environmental": [
        "a white diamond-shaped hazard label with a dead tree and fish symbol inside a white border",
        "environmental hazard ADR placard white diamond close-up with tree and fish pictogram",
        "aquatic hazard warning sign white diamond with black dead tree and dead fish on white background",
        "white diamond-shaped GHS sign with black silhouette of dead tree and fish inside white border",
    ],
    "gas_cylinder": [
        "a white diamond hazard label with gas cylinder bottle symbol",
        "compressed gas white diamond placard close-up",
        "gas pressure warning sign white diamond bottle icon",
    ],
    "no_ghs_label": [
        "a blank section of a dark freight wagon with no labels",
        "empty dark metal surface with no warning signs",
        "black tank wagon wall with no hazard placards",
        "yellow equilateral triangle sign on a freight wagon not a diamond shape",
        "rectangular plate with text and numbers on a wagon not a diamond shape",
        "circular logo or brand marking on a tank wagon not a diamond shape",
        "square sticker with handling instructions on a container not a diamond shape",
        "green recycling symbol with arrows and globe on a freight wagon",
        "company environmental logo with recycling arrows on a train wagon",
        "solid red painted surface of a freight wagon with no symbols or markings",
        "red metal panel of a train wagon without any hazard pictogram",
        "red wagon body paint with no warning signs attached",
    ],
}

# ---------------------------------------------------------------------------
# DINO SETTINGS
# ---------------------------------------------------------------------------

BOX_THRESHOLD  = 0.45
TEXT_THRESHOLD = 0.1

MIN_BOX_SIZE = 0.03
MAX_BOX_SIZE = 0.15

MIN_ASPECT = 0.55
MAX_ASPECT = 1.4

HAZMAT_COLORS = {"red", "blue", "orange", "green","white"}

ALL_SIGNS = {
    "hazmat_diamond": (
        "diamond-shaped hazard sign on the wagon . "
        "red diamond-shaped sign with flame . "
        "yellow diamond-shaped sign with flame over circle . "
        "white diamond-shaped sign with skull and crossbones . "
        "white diamond-shaped sign with black vertical stripes . "
        "blue diamond-shaped sign with flame . "
        "orange diamond-shaped sign with exploding bomb . "
        "green diamond-shaped sign with flame . "
        "white diamond-shaped sign with dead tree and fish . "
        "white diamond-shaped sign with dripping liquid on hand . "
        "white diamond-shaped sign with gas cylinder"
    ),
    "kemler": "orange rectangle with numbers",
}

# ---------------------------------------------------------------------------
# DINO HELPERS
# ---------------------------------------------------------------------------

def _build_caption() -> str:
    return " . ".join(ALL_SIGNS.values())


def _clean_label(label: str) -> set:
    return set(label.lower().split())


def _classify_sign(label_words: set) -> str | None:
    if "diamond" in label_words and (label_words & HAZMAT_COLORS):
        return "hazmat_diamond"
    if "diamond" in label_words and ("fish" in label_words or "tree" in label_words):
        return "hazmat_diamond"
    if "rectangle" in label_words and "orange" in label_words:
        return "kemler"
    return None


def _box_iou_like(box1, box2, img_w: int, img_h: int, tol: float = 0.03) -> bool:
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    ca_x = (x1a + x2a) / 2 / img_w
    ca_y = (y1a + y2a) / 2 / img_h
    cb_x = (x1b + x2b) / 2 / img_w
    cb_y = (y1b + y2b) / 2 / img_h
    return abs(ca_x - cb_x) < tol and abs(ca_y - cb_y) < tol


def _valid_box(bw: float, bh: float, label_words: set) -> bool:
    if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
        return False
    if bw > MAX_BOX_SIZE or bh > MAX_BOX_SIZE:
        return False
    if "diamond" in label_words:
        aspect = bw / bh
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            return False
    return True


def _crop_detection(frame_bgr: np.ndarray, box: list, pad: int = 10) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(frame_bgr.shape[1], x_max + pad)
    y_max = min(frame_bgr.shape[0], y_max + pad)
    return frame_bgr[y_min:y_max, x_min:x_max]


# HSV ranges for colored diamond pre-checks
_COLOR_RANGES = {
    "flammable":  [((0,   100, 80), (10,  255, 255)), ((170, 100, 80), (180, 255, 255))],  # red (wraps hue)
    "oxidizer":   [((20,  100, 80), (35,  255, 255))],                                      # yellow
    "explosive":  [((8,   150, 80), (20,  255, 255))],                                      # orange
}
_COLOR_MIN_RATIO = 0.10   # at least 10% of crop pixels must match the expected color


def _passes_color_check(crop_bgr: np.ndarray, label: str) -> bool:
    """Return False if the crop lacks the expected dominant color for colored diamond classes."""
    ranges = _COLOR_RANGES.get(label)
    if ranges is None:
        return True  # no check for white-diamond classes
    if crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    matched = sum(
        cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi)))
        for lo, hi in ranges
    )
    return (matched / total) >= _COLOR_MIN_RATIO


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class GHSDetector:
    """
    Detects GHS/ADR dangerous goods signs in wagon frames.

    Usage::

        detector = GHSDetector()
        labels = detector.detect(wagon.frames)   # list[np.ndarray] BGR frames
        # labels → e.g. ["Flammable Liquid", "Toxic"]
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _DEFAULT_DINO_CFG = str(Path(__file__).resolve().parents[2] / "models" / "GroundingDINO_SwinT_OGC.py")
    _DEFAULT_DINO_WEIGHTS = str(Path(__file__).resolve().parents[2] / "models" / "groundingdino_swint_ogc.pth")
    _DEFAULT_SVM = str(Path(__file__).resolve().parents[2] / "ghs_svm.pkl")

    def __init__(
        self,
        dino_cfg: str = None,
        dino_weights: str = None,
        svm_path: str = None,
        clip_threshold: float = 0.85,
        min_frames: int = 1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_threshold = clip_threshold
        self.min_frames = min_frames

        dino_cfg     = dino_cfg     or self._DEFAULT_DINO_CFG
        dino_weights = dino_weights or self._DEFAULT_DINO_WEIGHTS

        # --- Load GroundingDINO ---
        log.info("[GHS] Loading GroundingDINO on %s...", self.device)
        from groundingdino.util.inference import load_model, predict
        self._predict = predict
        self._dino = load_model(dino_cfg, dino_weights)

        # --- Load CLIP ---
        log.info("[GHS] Loading CLIP (ViT-B/32) on %s...", self.device)
        import clip as openai_clip
        self._clip_model, self._clip_preprocess = openai_clip.load("ViT-B/32", device=self.device)
        self._clip_model.eval()
        self._class_names, self._text_emb = self._build_text_embeddings()

        # --- Load SVM (optional) ---
        if svm_path is None:
            svm_path = self._DEFAULT_SVM
        self._svm_bundle = self._load_svm(svm_path) if svm_path else None

        self._caption = _build_caption()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def detect(self, frames: list) -> list[str]:
        """
        Run DINO + CLIP on a list of BGR numpy frames.

        Returns a deduplicated list of human-readable dangerous goods labels,
        e.g. ["Flammable Liquid", "Toxic"]. Returns [] if nothing found.
        """
        frame_votes: dict[str, set[int]] = {}  # label → set of frame indices that detected it

        for frame_idx, frame in enumerate(frames):
            crops = self._detect_signs_in_frame(frame)
            for crop_bgr, sign_key in crops:
                if sign_key != "hazmat_diamond":
                    continue  # kemler is handled by OCR pipeline
                label = self._classify_crop(crop_bgr)
                log.debug("[GHS] frame=%d sign=%s -> %s", frame_idx, sign_key, label)
                if label and label != "no_ghs_label":
                    frame_votes.setdefault(label, set()).add(frame_idx)

        accepted = [
            GHS_LABEL_MAP[cls]
            for cls, frames_seen in frame_votes.items()
            if len(frames_seen) >= self.min_frames and cls in GHS_LABEL_MAP
        ]
        if frame_votes:
            log.debug("[GHS] votes: %s (min_frames=%d)",
                      {k: len(v) for k, v in frame_votes.items()}, self.min_frames)
        return sorted(accepted)

    # ------------------------------------------------------------------
    # DINO DETECTION
    # ------------------------------------------------------------------

    def _detect_signs_in_frame(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Returns list of (crop_bgr, sign_key) for each accepted detection."""
        from groundingdino.util.inference import predict

        pil_img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB PIL
        h, w = frame_bgr.shape[:2]

        from groundingdino.datasets import transforms as GDT
        transform = GDT.Compose([
            GDT.RandomResize([800], max_size=1333),
            GDT.ToTensor(),
            GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(pil_img, None)

        boxes, scores, labels = self._predict(
            model=self._dino,
            image=image_tensor,
            caption=self._caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.device,
        )

        accepted: list[dict] = []   # within-frame accepted detections for dedup

        results = []
        for box, score, label in zip(boxes, scores, labels):
            cx, cy, bw, bh = box
            label_words = _clean_label(label)

            if not _valid_box(bw, bh, label_words):
                continue

            sign_key = _classify_sign(label_words)
            if sign_key is None:
                continue

            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
            pixel_box = [x_min, y_min, x_max, y_max]

            # Duplicate check (within frame)
            is_dup = any(
                prev["sign_key"] == sign_key and _box_iou_like(prev["box"], pixel_box, w, h)
                for prev in accepted
            )
            if is_dup:
                continue

            crop = _crop_detection(frame_bgr, pixel_box)
            accepted.append({"sign_key": sign_key, "box": pixel_box})
            results.append((crop, sign_key))

        return results

    # ------------------------------------------------------------------
    # CLIP CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify_crop(self, crop_bgr: np.ndarray) -> str | None:
        """Return top-1 GHS class name, or None if crop is empty."""
        if crop_bgr.size == 0:
            return None

        pil_img = Image.fromarray(crop_bgr[:, :, ::-1]).convert("RGB")

        if self._svm_bundle is not None:
            results = self._svm_classify(pil_img)
        else:
            results = self._zeroshot_classify(pil_img)

        top_label, top_score = results[0]
        log.debug("[CLIP] top=%s (%.2f), 2nd=%s (%.2f), 3rd=%s (%.2f)",
                  top_label, top_score, results[1][0], results[1][1], results[2][0], results[2][1])
        if self.clip_threshold > 0 and top_score < self.clip_threshold:
            log.debug("[CLIP] rejected (below threshold %s)", self.clip_threshold)
            return None
        if not _passes_color_check(crop_bgr, top_label):
            log.debug("[CLIP] rejected (color check failed for %s)", top_label)
            return None
        return top_label

    def _zeroshot_classify(self, pil_img: Image.Image) -> list[tuple]:
        tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._clip_model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        sims = (100.0 * emb @ self._text_emb.T).softmax(dim=-1)
        scores = sims.squeeze().cpu().numpy()
        ranked = np.argsort(scores)[::-1]
        return [(self._class_names[i], float(scores[i])) for i in ranked[:3]]

    def _svm_classify(self, pil_img: Image.Image) -> list[tuple]:
        tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._clip_model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        feat = emb.squeeze().cpu().numpy().reshape(1, -1)

        svm = self._svm_bundle["svm"]
        classes = self._svm_bundle["classes"]
        probs = svm.predict_proba(feat)[0]
        ranked = np.argsort(probs)[::-1]
        return [(classes[i], float(probs[i])) for i in ranked[:3]]

    # ------------------------------------------------------------------
    # SETUP HELPERS
    # ------------------------------------------------------------------

    def _build_text_embeddings(self):
        import clip as openai_clip
        class_names = list(GHS_PROMPTS.keys())
        text_embeddings = []
        with torch.no_grad():
            for cls in class_names:
                tokens = openai_clip.tokenize(GHS_PROMPTS[cls]).to(self.device)
                embs = self._clip_model.encode_text(tokens)
                embs /= embs.norm(dim=-1, keepdim=True)
                avg = embs.mean(dim=0)
                avg /= avg.norm()
                text_embeddings.append(avg)
        return class_names, torch.stack(text_embeddings)

    def _load_svm(self, path: str):
        """Load an optional SVM classifier bundle. Returns ``None`` on any failure."""
        if not os.path.exists(path):
            log.info("[GHS] SVM not found at %s, using zero-shot CLIP.", path)
            return None
        try:
            import joblib
            bundle = joblib.load(path)
            log.info("[GHS] Loaded SVM from %s", path)
            return bundle
        except Exception as e:
            log.warning("[GHS] Could not load SVM (%s), using zero-shot CLIP.", e)
            return None
