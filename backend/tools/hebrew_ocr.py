"""
Hebrew OCR engine exposing the same interface as PaddleOCR.

OcrRecogniser consumes `engine.predict_iter(image)`, which yields dicts with
`dt_polys`, `rec_texts` and `rec_scores`. HebrewOCRPaddleAdapter produces that
exact shape from paddleocr-hebrew, so OcrRecogniser needs no Hebrew-specific
code beyond choosing this engine.

paddleocr-hebrew is imported only when an engine is constructed, so this module
is safe to import for every language.
"""
import os

from backend.config import BASE_DIR

HEBREW_LANGUAGES = frozenset({"he", "hebrew", "עברית"})
MODELS_DIR_ENV_VAR = "VSE_HEBREW_MODELS_DIR"
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models", "hebrew")


def is_hebrew_language(language):
    return str(language).strip().lower() in HEBREW_LANGUAGES


def resolve_models_dir():
    models_dir = os.environ.get(MODELS_DIR_ENV_VAR, DEFAULT_MODELS_DIR)
    if not os.path.isdir(models_dir):
        raise RuntimeError(
            "Hebrew OCR models were not found. "
            f"Expected them at: {models_dir}. "
            f"Set {MODELS_DIR_ENV_VAR} to the downloaded model directory."
        )
    return models_dir


class HebrewOCRPaddleAdapter:
    """Adapts paddleocr-hebrew to the subset of PaddleOCR's interface used here."""

    def __init__(self, models_dir, providers):
        try:
            from paddleocr_hebrew import HebrewOCR
            from backend.tools.hebrew_confidence import (
                ScoredPlanECascade,
                read_lines_with_confidence,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hebrew OCR is selected, but paddleocr-hebrew is not installed."
            ) from exc

        self._read_lines = read_lines_with_confidence
        self.ocr = HebrewOCR.word(models_dir=models_dir, providers=providers)
        # The stock recogniser discards its confidence; swap in one that keeps it.
        self.ocr.rec = ScoredPlanECascade.from_cascade(self.ocr.rec)

    @classmethod
    def create(cls, hardware_accelerator):
        """Build an engine from the app's hardware configuration."""
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if hardware_accelerator.has_cuda()
            else ["CPUExecutionProvider"]
        )
        return cls(models_dir=resolve_models_dir(), providers=providers)

    def predict_iter(self, image):
        # Whole lines rather than words: the package already assembled each
        # line in logical RTL Unicode order.
        dt_polys = []
        rec_texts = []
        rec_scores = []

        for line in self._read_lines(self.ocr, image):
            text = str(line.get("text", "")).strip()
            bbox = line.get("bbox")
            if not text or not bbox or len(bbox) != 4:
                continue

            xmin, ymin, xmax, ymax = map(int, bbox)
            dt_polys.append([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ])
            rec_texts.append(text)
            rec_scores.append(float(line["conf"]))

        yield {
            "dt_polys": dt_polys,
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        }
