"""Runs in THOMAS_PYTHON, detached: train on Modal, pull the artifact, record status.

    python train_runner.py <run_dir>

Reads <run_dir>/config.json (written by the plugin), writes <run_dir>/status.json
at every state change. Uses thomas' public Modal entry points, so this is the
same code path as `modal run -m thomas.encoder_train`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def write_status(run_dir: Path, **fields) -> None:
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "status.json")


def main(run_dir: Path) -> int:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_status(run_dir, state="running", started=started)
    try:
        from thomas.encoder_train import (EncoderTrainConfig, pull_encoder_artifact,
                                          run_encoder_train_modal)

        rows = []
        for line in Path(cfg["data_path"]).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                d = json.loads(line)
                rows.append((d["text"], d["label"]))
        config = EncoderTrainConfig(
            model_name=cfg["model"], num_labels=len({l for _, l in rows}),
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            calib_size=cfg["calib_size"], seed=cfg["seed"])
        print(f"[thomas] training {cfg['run_name']}: {config}", flush=True)
        result = run_encoder_train_modal(rows, config, cfg["run_name"])
        print(f"[thomas] trained; T={result.temperature:.4f}; pulling artifact", flush=True)

        artifact = run_dir / "artifact"
        last = None
        for attempt in range(4):  # a fresh read container can see a stale volume snapshot
            try:
                pull_encoder_artifact(cfg["run_name"], str(artifact))
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[thomas] pull attempt {attempt + 1} failed: {exc!r}", flush=True)
                time.sleep(10 * (attempt + 1))
        if last is not None:
            raise last

        m = result.metrics
        write_status(run_dir, state="done", started=started,
                     finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                     artifact_dir=str(artifact), temperature=result.temperature,
                     calib_accuracy=m.get("calib_accuracy"), ece_before=m.get("ece_before"),
                     ece_after=m.get("ece_after"), train_size=m.get("train_size"),
                     calib_size=m.get("calib_size"), num_labels=m.get("num_labels"),
                     final_train_loss=m.get("final_train_loss"))
        print("[thomas] done", flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - status must be written on any exit
        traceback.print_exc()
        write_status(run_dir, state="failed", started=started,
                     finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                     error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
