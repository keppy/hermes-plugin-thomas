"""Tool handlers for the thomas plugin.

Every handler returns a JSON string and never raises: a missing file, a bad
row, or a missing training environment comes back as ``{"error": ...}`` for
the model to read and act on.

Two environments, on purpose:

- The Hermes venv runs this module and gonogo (declared in pyproject.toml).
  Data checks, run bookkeeping and the gonogo verdict happen here.
- ``THOMAS_PYTHON`` points at an interpreter with ``thomas-train[encoder]``
  installed (torch, transformers, modal, with ``modal token`` configured).
  Training and inference run there as subprocesses, so a GPU toolchain never
  has to live inside Hermes.

Runs live under ``$HERMES_HOME/thomas/runs/<run_name>/``: ``config.json``,
``status.json`` (written by the runner), ``log.txt`` and ``artifact/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "thomas_check_data", "thomas_encoder_train", "thomas_run_status",
    "thomas_encoder_eval", "credit_gate",
]

DEFAULT_MODEL = "johnnyboycurtis/ModernBERT-small-v2"
DEFAULTS = {"epochs": 3, "batch_size": 32, "lr": 2e-5, "calib_size": 500, "seed": 7}
_HERE = Path(__file__).resolve().parent
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MIN_PER_LABEL = 5
# thomas docs/CONTRACT.md. Readers accept artifacts at or below this version;
# an artifact without the field is version 1.
CONTRACT_VERSION = 1  # below this a label can neither be learned nor show up in the calib split

# --------------------------------------------------------------------------- #
# helpers


def _fail(message: str, **extra: Any) -> str:
    return json.dumps({"error": message, **extra}, ensure_ascii=False)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return int(f) if f.is_integer() else default


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        env = os.environ.get("HERMES_HOME")
        return Path(env).expanduser() if env else Path.home() / ".hermes"


def runs_root() -> Path:
    override = os.environ.get("THOMAS_RUNS_DIR")
    root = Path(override).expanduser() if override else _hermes_home() / "thomas" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _thomas_python() -> str:
    exe = os.environ.get("THOMAS_PYTHON", "").strip()
    if not exe:
        raise RuntimeError(
            "THOMAS_PYTHON is not set. Point it at a Python with thomas-train[encoder] "
            "installed and `modal token` configured, e.g. <thomas repo>/.venv/Scripts/python.exe")
    if not Path(exe).expanduser().is_file():
        raise RuntimeError(f"THOMAS_PYTHON={exe} is not a file")
    return str(Path(exe).expanduser())


def load_rows(raw_path: Any) -> tuple[list[dict], list[str]]:
    """Read a {text, label, id?} JSONL file. Returns (valid rows, problems)."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("no data path given")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    rows, problems = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {lineno}: not a JSON object")
            continue
        text, label = row.get("text"), row.get("label")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"line {lineno}: missing or empty 'text'")
            continue
        if not isinstance(label, str) or not label.strip():
            problems.append(f"line {lineno}: 'label' must be a non-empty string")
            continue
        rows.append({"id": str(row.get("id") or f"row-{lineno}"), "text": text, "label": label})
    return rows, problems


def check_rows(rows: list[dict], problems: list[str], calib_size: int) -> dict:
    counts = Counter(r["label"] for r in rows)
    texts = Counter(r["text"].strip().lower() for r in rows)
    dup_texts = sum(c - 1 for c in texts.values() if c > 1)
    conflicting = 0
    by_text: dict[str, set] = {}
    for r in rows:
        by_text.setdefault(r["text"].strip().lower(), set()).add(r["label"])
    conflicting = sum(1 for labels in by_text.values() if len(labels) > 1)
    thin = sorted(l for l, c in counts.items() if c < MIN_PER_LABEL)

    blockers, warnings = [], []
    if len(rows) == 0:
        blockers.append("no valid rows")
    if len(counts) < 2:
        blockers.append(f"need at least 2 labels, found {len(counts)}")
    if problems:
        blockers.append(f"{len(problems)} row(s) fail the schema — fix or drop them first")
    if calib_size >= len(rows) and rows:
        blockers.append(f"calib_size {calib_size} >= {len(rows)} rows: nothing left to train on")
    elif rows and calib_size > 0.3 * len(rows):
        warnings.append(f"calib_size {calib_size} is {calib_size / len(rows):.0%} of the data; "
                        "consider a smaller holdout")
    if thin:
        warnings.append(f"{len(thin)} label(s) have fewer than {MIN_PER_LABEL} rows: "
                        + ", ".join(thin[:10]) + (" ..." if len(thin) > 10 else ""))
    if conflicting:
        warnings.append(f"{conflicting} text(s) appear with more than one label — "
                        "label noise caps accuracy")
    if dup_texts:
        warnings.append(f"{dup_texts} duplicate text(s); if duplicates straddle train and "
                        "eval, the eval is optimistic")
    if counts:
        top, top_n = counts.most_common(1)[0]
        if top_n / len(rows) > 0.5:
            warnings.append(f"label '{top}' is {top_n / len(rows):.0%} of rows; a constant "
                            "predictor already scores that")

    return {
        "ok": not blockers,
        "n_rows": len(rows),
        "n_labels": len(counts),
        "label_counts": dict(counts.most_common()),
        "min_per_label": min(counts.values()) if counts else 0,
        "duplicate_texts": dup_texts,
        "conflicting_texts": conflicting,
        "schema_problems": problems[:20],
        "n_schema_problems": len(problems),
        "blockers": blockers,
        "warnings": warnings,
    }


def train_config(args: dict) -> dict:
    """The resolved config a train call would launch — also what the gate shows."""
    cfg = {
        "data_path": str(Path(str(args.get("data_path", ""))).expanduser()),
        "model": str(args.get("model") or DEFAULT_MODEL),
        "epochs": _as_int(args.get("epochs"), DEFAULTS["epochs"]),
        "batch_size": _as_int(args.get("batch_size"), DEFAULTS["batch_size"]),
        "lr": _as_float(args.get("lr"), DEFAULTS["lr"]),
        "calib_size": _as_int(args.get("calib_size"), DEFAULTS["calib_size"]),
        "seed": _as_int(args.get("seed"), DEFAULTS["seed"]),
        "gpu": "L4 (Modal)",
    }
    name = args.get("run_name")
    cfg["run_name"] = str(name).strip() if isinstance(name, str) and name.strip() else ""
    return cfg


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail(path: Path, n: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:] if n > 0 else []


def _spawn(argv: list[str], log_path: Path) -> int:
    """Start a detached child that outlives this tool call; return its pid."""
    log = open(log_path, "ab")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT,
                              "stdin": subprocess.DEVNULL, "cwd": str(log_path.parent)}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(argv, env=env, **kwargs)
    log.close()
    return proc.pid


def run_state(run_dir: Path) -> dict:
    status = _read_json(run_dir / "status.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    state = status.get("state", "unknown")
    if state in ("launched", "running") and not _pid_alive(int(config.get("pid", 0) or 0)):
        state = "failed"
        status.setdefault("error", "runner exited without writing a final status — see log")
    out = {"run_name": run_dir.name, "state": state, "config": {k: v for k, v in config.items()
                                                               if k != "pid"}}
    for key in ("started", "finished", "error", "artifact_dir", "temperature",
                "calib_accuracy", "ece_before", "ece_after", "train_size", "calib_size",
                "num_labels", "final_train_loss"):
        if key in status:
            out[key] = _clean(status[key])
    return out


# --------------------------------------------------------------------------- #
# the credit gate


def credit_gate(tool_name: str = "", args: dict | None = None, **_: Any) -> dict | None:
    """``pre_tool_call`` hook: escalate every training launch to the human gate.

    The rule key hashes the resolved config, so an ``[a]lways`` answer covers
    exactly this config and never a different model, dataset or epoch count.
    """
    if tool_name != "thomas_encoder_train":
        return None
    cfg = train_config(args or {})
    try:
        n_rows = len(load_rows(cfg["data_path"])[0])
    except Exception:
        n_rows = "?"
    summary = (f"thomas: launch a PAID GPU fine-tune — {cfg['model']} on {cfg['data_path']} "
               f"({n_rows} rows), {cfg['epochs']} epochs, batch {cfg['batch_size']}, "
               f"lr {cfg['lr']}, calib {cfg['calib_size']}, seed {cfg['seed']}, {cfg['gpu']}")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return {"action": "approve", "message": summary, "rule_key": f"thomas_encoder_train:{digest}"}


# --------------------------------------------------------------------------- #
# tools


def thomas_check_data(args: dict, **_: Any) -> str:
    try:
        rows, problems = load_rows(args.get("data_path"))
        out = check_rows(rows, problems, _as_int(args.get("calib_size"), DEFAULTS["calib_size"]))
        out["data_path"] = str(Path(str(args.get("data_path"))).expanduser())
        return json.dumps(out, ensure_ascii=False)
    except ValueError as exc:
        return _fail(str(exc))
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


def thomas_encoder_train(args: dict, **_: Any) -> str:
    try:
        cfg = train_config(args)
        rows, problems = load_rows(cfg["data_path"])
        check = check_rows(rows, problems, cfg["calib_size"])
        if not check["ok"]:
            return _fail("data check failed; nothing launched", blockers=check["blockers"],
                         warnings=check["warnings"])
        py = _thomas_python()

        name = cfg["run_name"] or f"enc-{int(time.time())}"
        if not _RUN_NAME.match(name):
            return _fail(f"run_name {name!r} must be letters, digits, '.', '_' or '-' (max 64)")
        run_dir = runs_root() / name
        if run_dir.exists():
            return _fail(f"run {name!r} already exists; pick another run_name",
                         run=run_state(run_dir))
        run_dir.mkdir(parents=True)
        cfg["run_name"] = name
        cfg["num_labels"] = check["n_labels"]
        cfg["n_rows"] = check["n_rows"]
        cfg["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (run_dir / "status.json").write_text(json.dumps({"state": "launched"}), encoding="utf-8")
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        pid = _spawn([py, str(_HERE / "runners" / "train_runner.py"), str(run_dir)],
                     run_dir / "log.txt")
        cfg["pid"] = pid
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return json.dumps({
            "run_name": name, "state": "launched", "run_dir": str(run_dir),
            "warnings": check["warnings"],
            "next": "poll thomas_run_status with this run_name; do not launch a second run",
        }, ensure_ascii=False)
    except (ValueError, RuntimeError) as exc:
        return _fail(str(exc))
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


def thomas_run_status(args: dict, **_: Any) -> str:
    try:
        root = runs_root()
        name = args.get("run_name")
        if not isinstance(name, str) or not name.strip():
            runs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")),
                          key=lambda p: p.stat().st_mtime, reverse=True)[:20]
            return json.dumps({"runs_dir": str(root),
                               "runs": [{k: v for k, v in run_state(p).items() if k != "config"}
                                        for p in runs]}, ensure_ascii=False)
        name = name.strip()
        if not _RUN_NAME.match(name):
            return _fail(f"run_name {name!r} must be letters, digits, '.', '_' or '-' (max 64)")
        run_dir = root / name
        if not run_dir.is_dir():
            return _fail(f"no run named {name!r} under {root}")
        out = run_state(run_dir)
        out["log_tail"] = _tail(run_dir / "log.txt", _as_int(args.get("log_lines"), 20))
        return json.dumps(out, ensure_ascii=False)
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


def _gonogo_report(pred_rows: list[dict], *, task: str, target: float):
    """Rebuild predictions into gonogo objects and take gonogo's verdict."""
    import gonogo as g

    results = []
    for p in pred_rows:
        case = g.Case(input=p["text"], expected=p["expected"], id=p["id"])
        ok = p["output"] == p["expected"]
        results.append(g.CaseResult(
            case, g.Prediction(p["output"], confidence=float(p["confidence"])),
            passed=ok, score=1.0 if ok else 0.0,
            detail="" if ok else f"expected {p['expected']!r}, got {p['output']!r}"))
    trials = [(r.confidence, r.passed) for r in results]
    decision = g.decide(trials, target=target, level=0.95, unit="cases")
    report = g.Report(task=task, results=results, decision=decision,
                      metadata={"source": "hermes-plugin-thomas"})
    return decision, report


def thomas_encoder_eval(args: dict, **_: Any) -> str:
    try:
        model_dir = Path(str(args.get("model_dir") or "")).expanduser()
        for need in ("config.json", "label2id.json", "temperature.json", "metrics.json"):
            if not (model_dir / need).is_file():
                return _fail(f"{model_dir} is not a thomas artifact dir (missing {need})")
        metrics = _read_json(model_dir / "metrics.json") or {}
        version = _as_int(metrics.get("contract_version"), 1)
        if version > CONTRACT_VERSION:
            return _fail(f"artifact has thomas contract_version {version}; this plugin reads "
                         f"<= {CONTRACT_VERSION} — update hermes-plugin-thomas")
        rows, problems = load_rows(args.get("cases_path"))
        if problems:
            return _fail(f"{len(problems)} case row(s) fail the schema", problems=problems[:20])
        if not rows:
            return _fail("no cases")
        target = _as_float(args.get("target"), 0.95)
        if not 0.0 < target < 1.0:
            return _fail(f"target must be in (0, 1), got {target}")
        label2id = json.loads((model_dir / "label2id.json").read_text(encoding="utf-8"))
        unknown = sorted({r["label"] for r in rows} - set(label2id))
        py = _thomas_python()

        work = runs_root() / "_evals"
        work.mkdir(exist_ok=True)
        stamp = f"{int(time.time() * 1000)}"
        cases_file, preds_file = work / f"{stamp}.cases.jsonl", work / f"{stamp}.preds.jsonl"
        cases_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                              encoding="utf-8")
        proc = subprocess.run(
            [py, str(_HERE / "runners" / "predict_runner.py"), str(model_dir),
             str(cases_file), str(preds_file)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if proc.returncode != 0:
            return _fail("inference failed", stderr_tail=proc.stderr.splitlines()[-15:])
        preds = [json.loads(l) for l in preds_file.read_text(encoding="utf-8").splitlines() if l]
        for f in (cases_file, preds_file):
            f.unlink(missing_ok=True)

        task = str(args.get("task") or f"encoder: {model_dir.name}")
        decision, report = _gonogo_report(preds, task=task, target=target)
        op = decision.operating_point
        out = {
            "task": task,
            "verdict": decision.verdict.value,
            "reason": decision.reason,
            "n": len(preds),
            "n_passed": sum(1 for _c, ok in ((0, p["output"] == p["expected"]) for p in preds) if ok),
            "pass_rate": str(decision.pass_rate),
            "target": target,
            "operating_point": None if op is None else {
                "threshold": op.threshold, "coverage": op.coverage,
                "n_deferred": op.n_deferred, "precision": str(op.precision)},
            "calibration_error": _clean(decision.calibration_error),
            "needed_n": decision.needed_n,
            "notes": list(decision.notes),
            "labels_unseen_in_training": unknown,
            "contract_version": version,
            "markdown": report.markdown(show_failures=5),
        }
        save = args.get("save_report")
        if isinstance(save, str) and save.strip():
            dest = Path(save).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                            encoding="utf-8")
            out["saved_report"] = str(dest)
        html = args.get("write_html")
        if isinstance(html, str) and html.strip():
            dest = Path(html).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(report.html(show_failures=5), encoding="utf-8")
            out["html_path"] = str(dest)
        return json.dumps(out, ensure_ascii=False)
    except (ValueError, RuntimeError) as exc:
        return _fail(str(exc))
    except subprocess.TimeoutExpired:
        return _fail("inference timed out after 30 minutes")
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
