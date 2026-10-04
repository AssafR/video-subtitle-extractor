"""
Line-level confidence for paddleocr-hebrew.

paddleocr-hebrew computes a CTC confidence internally (only to gate its NRTR
fallback) and discards it, and its NRTR decoder never keeps token
probabilities. ScoredPlanECascade reuses the library's own decoding but keeps a
per-crop confidence, and read_lines_with_confidence() aggregates the word
scores into one score per line.
"""
from collections import defaultdict

import numpy as np
from paddleocr_hebrew import PlanECascade
from paddleocr_hebrew.detect import crop_bbox, detect_words
from paddleocr_hebrew.heb_postprocess import (
    cluster_rows, fix_geresh_yod_text, order_row, post_process,
)
from paddleocr_hebrew.plan_e_rec import _LTR, _softmax
from paddleocr_hebrew.rec_decode import prep_dynamic_target_w, quantize_width

NRTR_BOS = 2
NRTR_EOS = 3


class ScoredPlanECascade(PlanECascade):
    """PlanECascade that also returns a confidence (0..1) for every crop.

    CTC crops: mean max-softmax over emitted characters (the library's metric).
    NRTR crops: mean probability of the chosen tokens.
    """

    @classmethod
    def from_cascade(cls, cascade):
        """Share the already-loaded ONNX sessions instead of reloading them."""
        scored = cls.__new__(cls)
        scored.__dict__.update(cascade.__dict__)
        return scored

    def _nrtr_infer_scored(self, tensor):
        memory = self.enc.run(None, {self.ein: tensor})[0]
        seq = np.array([[NRTR_BOS]], dtype=np.int64)
        probs_chosen = []
        for _ in range(self.maxlen):
            wp = self.dec.run(None, {self.dn[0]: memory, self.dn[1]: seq})[0][0]
            if not (wp.min() >= 0.0 and abs(float(wp.sum()) - 1.0) < 0.05):
                wp = _softmax(wp)
            nxt = int(wp.argmax())
            if nxt == NRTR_EOS:
                break
            probs_chosen.append(float(wp[nxt]))
            seq = np.concatenate([seq, [[nxt]]], axis=1)
        out, confs = [], []
        for idx, p in zip(list(seq[0])[1:], probs_chosen):
            if idx < 4 or idx >= len(self.FULL):
                continue
            out.append(self.FULL[idx])
            confs.append(p)
        return "".join(out), (float(np.mean(confs)) if confs else 0.0)

    def rec_crops_scored(self, crops_bgr, mode="cascade"):
        """Return (texts, confidences), both aligned to crops_bgr."""
        self.last_n_fallback = 0
        n = len(crops_bgr)
        preds = [""] * n
        confs = [0.0] * n
        tens = [None] * n
        widths = [None] * n
        for i, c in enumerate(crops_bgr):
            if c is None or c.shape[0] < 2 or c.shape[1] < 2:
                continue
            widths[i] = quantize_width(c, 48, 16, max_w=self.max_w)
            tens[i] = prep_dynamic_target_w(c, 48, widths[i])[None].astype(np.float32)

        if mode == "nrtr":
            for i, t in enumerate(tens):
                if t is not None:
                    preds[i], confs[i] = self._nrtr_infer_scored(t)
                    self.last_n_fallback += 1
            return preds, confs

        buckets = defaultdict(list)
        for i, w in enumerate(widths):
            if w is not None:
                buckets[w].append(i)
        fallback = []
        for w, idxs in buckets.items():
            for j in range(0, len(idxs), self.batch_size):
                bidx = idxs[j:j + self.batch_size]
                batch = np.concatenate([tens[k] for k in bidx], axis=0)
                out = self.ctc.run(None, {self.cin: batch})[0]
                for bi, k in enumerate(bidx):
                    text, conf = self._ctc_decode_conf(out[bi])
                    if mode == "cascade" and (
                        conf < self.conf_threshold or bool(_LTR.search(text))
                    ):
                        fallback.append(k)
                    else:
                        preds[k], confs[k] = text, conf

        for k in fallback:
            preds[k], confs[k] = self._nrtr_infer_scored(tens[k])
            self.last_n_fallback += 1
        return preds, confs


def line_confidence(words):
    """Character-weighted mean of the word confidences in one line."""
    total = sum(len(w["text"]) for w in words)
    if total == 0:
        return 0.0
    return sum(w["conf"] * len(w["text"]) for w in words) / total


def read_lines_with_confidence(ocr, img_bgr):
    """
    Equivalent of HebrewOCR.read_array(), but each returned line carries
    "conf" (0..1). `ocr` is a HebrewOCR whose `rec` is a ScoredPlanECascade.
    """
    bboxes = detect_words(
        img_bgr, ocr.det_sess, ocr.det_inp,
        thresh=ocr.det_thresh, unclip_ratio=ocr.det_unclip,
        max_side=ocr.det_max_side, dilate_w=ocr.dilate_w, dilate_h=ocr.dilate_h,
    )
    crops, keep = [], []
    for bb in bboxes:
        crop = crop_bbox(img_bgr, bb)
        if crop is not None:
            crops.append(crop)
            keep.append(bb)
    texts, confs = ocr.rec.rec_crops_scored(crops)

    words = []
    for bb, txt, conf in zip(keep, texts, confs):
        if ocr.gershayim:
            txt = post_process(txt)
        if txt.strip():
            words.append({"bbox": list(bb), "text": txt, "conf": conf})

    lines = []
    for row in cluster_rows(words):
        ordered = order_row(row)
        text = " ".join(w["text"] for w in ordered).strip()
        if ocr.geresh_yod:
            text = fix_geresh_yod_text(text)
        if not text:
            continue
        lines.append({
            "text": text,
            "bbox": [
                min(w["bbox"][0] for w in ordered),
                min(w["bbox"][1] for w in ordered),
                max(w["bbox"][2] for w in ordered),
                max(w["bbox"][3] for w in ordered),
            ],
            "conf": line_confidence(ordered),
        })
    return lines
