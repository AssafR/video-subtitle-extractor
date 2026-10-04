import unittest
from unittest.mock import Mock, patch

import numpy as np

from backend.tools import hebrew_confidence
from backend.tools.hebrew_confidence import ScoredPlanECascade, line_confidence
from backend.tools.hebrew_ocr import (
    HebrewOCRPaddleAdapter,
    is_hebrew_language,
    resolve_models_dir,
)


def make_cascade(ctc_results, nrtr_result=("nrtr", 0.4)):
    cascade = ScoredPlanECascade.__new__(ScoredPlanECascade)
    cascade.max_w = 1280
    cascade.batch_size = 16
    cascade.conf_threshold = 0.5
    cascade.last_n_fallback = 0
    cascade.cin = "x"
    cascade.ctc = Mock()
    cascade.ctc.run.return_value = [np.zeros((len(ctc_results), 4, 4))]
    cascade._ctc_decode_conf = Mock(side_effect=ctc_results)
    cascade._nrtr_infer_scored = Mock(return_value=nrtr_result)
    return cascade


def crop():
    return np.zeros((48, 96, 3), dtype=np.uint8)


class HebrewConfidenceTest(unittest.TestCase):
    def test_line_confidence_is_weighted_by_word_length(self) -> None:
        words = [
            {"text": "אבגדה", "conf": 0.9},
            {"text": "א", "conf": 0.1},
        ]

        self.assertAlmostEqual(line_confidence(words), (0.9 * 5 + 0.1) / 6)

    def test_line_confidence_of_empty_text_is_zero(self) -> None:
        self.assertEqual(line_confidence([{"text": "", "conf": 0.9}]), 0.0)

    def test_ctc_confidence_is_kept_when_crop_is_accepted(self) -> None:
        cascade = make_cascade([("שלום", 0.93)])

        texts, confs = cascade.rec_crops_scored([crop()])

        self.assertEqual(texts, ["שלום"])
        self.assertEqual(confs, [0.93])
        cascade._nrtr_infer_scored.assert_not_called()

    def test_low_confidence_crop_uses_nrtr_confidence(self) -> None:
        cascade = make_cascade([("שלום", 0.2)], nrtr_result=("שלום2", 0.61))

        texts, confs = cascade.rec_crops_scored([crop()])

        self.assertEqual(texts, ["שלום2"])
        self.assertEqual(confs, [0.61])
        self.assertEqual(cascade.last_n_fallback, 1)

    def test_latin_crop_falls_back_even_when_ctc_is_confident(self) -> None:
        cascade = make_cascade([("Zoom", 0.99)], nrtr_result=("Zoom", 0.7))

        _, confs = cascade.rec_crops_scored([crop()])

        self.assertEqual(confs, [0.7])

    def test_invalid_crop_gets_zero_confidence(self) -> None:
        cascade = make_cascade([])

        texts, confs = cascade.rec_crops_scored([None])

        self.assertEqual((texts, confs), ([""], [0.0]))

    def test_adapter_reports_real_line_confidence(self) -> None:
        lines = [
            {"text": "שלום", "bbox": [0, 0, 10, 5], "conf": 0.42},
            {"text": "עולם", "bbox": [0, 10, 10, 15], "conf": 0.88},
        ]
        fake_ocr = Mock()
        with patch("paddleocr_hebrew.HebrewOCR") as hebrew_ocr, patch(
            "backend.tools.hebrew_confidence.ScoredPlanECascade"
        ), patch(
            "backend.tools.hebrew_confidence.read_lines_with_confidence",
            return_value=lines,
        ):
            hebrew_ocr.word.return_value = fake_ocr
            adapter = HebrewOCRPaddleAdapter("models", ["CPUExecutionProvider"])
            result = next(adapter.predict_iter(np.zeros((20, 20, 3), np.uint8)))

        self.assertEqual(result["rec_scores"], [0.42, 0.88])
        self.assertEqual(result["rec_texts"], ["שלום", "עולם"])
        self.assertEqual(result["dt_polys"][0], [[0, 0], [10, 0], [10, 5], [0, 5]])

    def test_language_detection(self) -> None:
        for language in ("he", "Hebrew", " HE ", "עברית"):
            self.assertTrue(is_hebrew_language(language))
        for language in ("en", "ch", "japan"):
            self.assertFalse(is_hebrew_language(language))

    def test_missing_models_dir_raises_helpful_error(self) -> None:
        with patch.dict("os.environ", {"VSE_HEBREW_MODELS_DIR": "Z:\\nope"}):
            with self.assertRaisesRegex(RuntimeError, "VSE_HEBREW_MODELS_DIR"):
                resolve_models_dir()

    def test_create_selects_cuda_provider_when_available(self) -> None:
        accelerator = Mock()
        accelerator.has_cuda.return_value = True
        with patch(
            "backend.tools.hebrew_ocr.resolve_models_dir", return_value="m"
        ), patch.object(HebrewOCRPaddleAdapter, "__init__", return_value=None) as init:
            HebrewOCRPaddleAdapter.create(accelerator)

        init.assert_called_once_with(
            models_dir="m",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_module_exposes_read_function(self) -> None:
        self.assertTrue(callable(hebrew_confidence.read_lines_with_confidence))


if __name__ == "__main__":
    unittest.main()
