"""Runs in THOMAS_PYTHON: CPU inference for a thomas encoder artifact.

    python predict_runner.py <model_dir> <cases.jsonl> <out.jsonl>

Writes one {id, text, expected, output, confidence} per case. Confidence is the
thomas contract definition — max(softmax(logits / T)) over the full label
distribution — via thomas.encoder_train.scaled_softmax, the same function the
training run used to fit T.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BATCH = 64
MAX_LENGTH = 128


def main(model_dir: str, cases_path: str, out_path: str) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from thomas.encoder_train import scaled_softmax

    d = Path(model_dir)
    label2id = json.loads((d / "label2id.json").read_text(encoding="utf-8"))
    id2label = {int(i): l for l, i in label2id.items()}
    temperature = float(json.loads((d / "temperature.json").read_text(encoding="utf-8"))["temperature"])
    if not temperature > 0:
        raise SystemExit(f"non-positive temperature {temperature} in {d}")

    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    model.eval()

    cases = [json.loads(l) for l in Path(cases_path).read_text(encoding="utf-8").splitlines() if l]
    out = []
    for i in range(0, len(cases), BATCH):
        chunk = cases[i:i + BATCH]
        enc = tok([c["text"] for c in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LENGTH)
        with torch.inference_mode():
            logits = model(**enc).logits
        probs = scaled_softmax(logits, temperature)
        conf, idx = probs.max(dim=-1)
        for c, p, k in zip(chunk, conf.tolist(), idx.tolist()):
            out.append({"id": c["id"], "text": c["text"], "expected": c["label"],
                        "output": id2label[int(k)], "confidence": round(float(p), 6)})
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
