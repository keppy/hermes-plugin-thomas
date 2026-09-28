"""Handler tests. No GPU, no Modal, no thomas install: the training subprocess
is replaced by a fake runner, and eval runs gonogo on canned predictions."""

from __future__ import annotations

import json
import importlib.util
import hashlib
import sys
import time
from pathlib import Path

import pytest

import tools


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def rows(n_per_label=20, labels=("a", "b", "c")):
    return [{"id": f"{l}-{i}", "text": f"text {l} {i}", "label": l}
            for l in labels for i in range(n_per_label)]


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setenv("THOMAS_RUNS_DIR", str(d))
    return d


# --- check_data -------------------------------------------------------------

def test_check_data_clean(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    out = json.loads(tools.thomas_check_data({"data_path": str(p), "calib_size": 10}))
    assert out["ok"] and out["n_rows"] == 60 and out["n_labels"] == 3
    assert out["blockers"] == []


def test_check_data_flags_schema_thin_and_conflicts(tmp_path):
    data = rows(10, ("a", "b")) + [{"text": "x", "label": "rare"},
                                    {"text": "text a 0", "label": "b"},
                                    {"text": "", "label": "a"}, {"text": "y", "label": 3}]
    p = write_jsonl(tmp_path / "d.jsonl", data)
    out = json.loads(tools.thomas_check_data({"data_path": str(p), "calib_size": 2}))
    assert not out["ok"]
    assert out["n_schema_problems"] == 2
    assert out["conflicting_texts"] == 1
    assert any("rare" in w for w in out["warnings"])


def test_check_data_blocks_calib_bigger_than_data(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", rows(5))
    out = json.loads(tools.thomas_check_data({"data_path": str(p)}))  # default calib 500
    assert not out["ok"] and any("nothing left to train on" in b for b in out["blockers"])


def test_check_data_missing_file():
    assert "error" in json.loads(tools.thomas_check_data({"data_path": "nope.jsonl"}))


# --- credit gate ------------------------------------------------------------

def test_gate_ignores_other_tools():
    assert tools.credit_gate(tool_name="thomas_check_data", args={}) is None
    assert tools.credit_gate(tool_name="terminal", args={"command": "ls"}) is None


def test_gate_escalates_train_with_config(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    d = tools.credit_gate(tool_name="thomas_encoder_train",
                          args={"data_path": str(p), "calib_size": 10, "epochs": 5, "run_name": "gate"}, task_id="t")
    assert d["action"] == "approve"
    assert "PAID" in d["message"] and "60 rows" in d["message"] and "5 epochs" in d["message"]
    assert "sha256" in d["message"]


def test_gate_rule_key_changes_with_data_and_nonce(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    first_digest = hashlib.sha256(p.read_bytes()).hexdigest()
    def key(name):
        return tools.credit_gate(tool_name="thomas_encoder_train",
                                 args={"data_path": str(p), "calib_size": 10, "run_name": name})["rule_key"]
    k1 = key("first")
    p.write_text(p.read_text(encoding="utf-8").replace("text a 0", "changed a 0"), encoding="utf-8")
    second_digest = hashlib.sha256(p.read_bytes()).hexdigest()
    k2 = key("second")
    manifest_root = tools.runs_root() / "_approvals"
    assert json.loads((manifest_root / "first" / "approval.json").read_text())["data_sha256"] == first_digest
    assert json.loads((manifest_root / "second" / "approval.json").read_text())["data_sha256"] == second_digest
    assert first_digest != second_digest and k1 != k2
    assert tools.credit_gate(tool_name="thomas_encoder_train",
                             args={"data_path": str(p), "run_name": "second"})["action"] == "block"
    assert key("third") != k2


# --- train + status (fake runner) --------------------------------------------

FAKE_RUNNER = """
import hashlib, json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
cfg = json.loads((run / "config.json").read_text())
bound = {k: v for k, v in cfg.items() if k != "pid"}
expected = hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert sys.argv[2] == expected
json.dump({"state": "running"}, open(run / "status.json", "w"))
print("fake training", flush=True)
time.sleep(0.3)
json.dump({"state": "done", "artifact_dir": str(run / "artifact"), "temperature": 0.8,
           "ece_before": 0.06, "ece_after": 0.01}, open(run / "status.json", "w"))
"""


@pytest.fixture
def fake_thomas(tmp_path, monkeypatch):
    runners = tmp_path / "plugin" / "runners"
    runners.mkdir(parents=True)
    (runners / "train_runner.py").write_text(FAKE_RUNNER, encoding="utf-8")
    monkeypatch.setattr(tools, "_HERE", tmp_path / "plugin")
    monkeypatch.setenv("THOMAS_PYTHON", sys.executable)


def wait_state(name, want, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = json.loads(tools.thomas_run_status({"run_name": name}))
        if out["state"] == want:
            return out
        time.sleep(0.2)
    raise AssertionError(f"run never reached {want}: {out}")


def test_train_launches_and_status_reports_done(tmp_path, fake_thomas):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    assert tools.credit_gate(tool_name="thomas_encoder_train",
                             args={"data_path": str(p), "calib_size": 10, "run_name": "t1"})["action"] == "approve"
    out = json.loads(tools.thomas_encoder_train(
        {"data_path": str(p), "calib_size": 10, "run_name": "t1"}))
    assert out["state"] == "launched" and out["run_name"] == "t1"
    done = wait_state("t1", "done")
    assert done["temperature"] == 0.8 and done["ece_after"] == 0.01
    assert "fake training" in "\n".join(done["log_tail"])
    listing = json.loads(tools.thomas_run_status({}))
    assert [r["run_name"] for r in listing["runs"]] == ["t1"]


def test_train_refuses_bad_data_before_launch(tmp_path, fake_thomas, runs_dir):
    p = write_jsonl(tmp_path / "d.jsonl", rows(3))  # 9 rows < default calib 500
    out = json.loads(tools.thomas_encoder_train({"data_path": str(p), "run_name": "bad"}))
    assert "error" in out and not (runs_dir / "bad").exists()


def test_train_refuses_duplicate_run_name(tmp_path, fake_thomas):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(p), "calib_size": 10, "run_name": "dup"}
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    json.loads(tools.thomas_encoder_train(args))
    assert "already exists" in json.loads(tools.thomas_encoder_train(args))["error"]


def test_train_rejects_path_like_run_name(tmp_path, fake_thomas):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    out = json.loads(tools.thomas_encoder_train(
        {"data_path": str(p), "calib_size": 10, "run_name": "../escape"}))
    assert "run_name" in out["error"]


def test_status_rejects_path_like_run_name(runs_dir):
    out = json.loads(tools.thomas_run_status({"run_name": "../escape"}))
    assert "run_name" in out["error"]
    assert not (runs_dir.parent / "escape").exists()


def test_status_listing_skips_internal_dirs(runs_dir):
    ev = runs_dir / "_evals"
    ev.mkdir(parents=True)
    (ev / "1.cases.jsonl").write_text("", encoding="utf-8")
    real = runs_dir / "t1"
    real.mkdir()
    (real / "status.json").write_text('{"state": "done"}', encoding="utf-8")
    out = json.loads(tools.thomas_run_status({}))
    assert [r["run_name"] for r in out["runs"]] == ["t1"]


def test_train_without_thomas_python(tmp_path, monkeypatch):
    monkeypatch.delenv("THOMAS_PYTHON", raising=False)
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(p), "calib_size": 10, "run_name": "no-python"}
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    out = json.loads(tools.thomas_encoder_train(args))
    assert "THOMAS_PYTHON" in out["error"]


def test_approval_snapshot_survives_source_edit(tmp_path, fake_thomas, runs_dir):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(p), "calib_size": 10, "run_name": "frozen"}
    original = p.read_bytes()
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    p.write_text("changed after approval", encoding="utf-8")
    assert json.loads(tools.thomas_encoder_train(args))["state"] == "launched"
    assert (runs_dir / "frozen" / "data.jsonl").read_bytes() == original


def test_snapshot_tamper_and_unprepared_run_fail_closed(tmp_path, fake_thomas, runs_dir):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(p), "calib_size": 10, "run_name": "tamper"}
    assert "approval" in json.loads(tools.thomas_encoder_train(args))["error"]
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    (runs_dir / "_approvals" / "tamper" / "data.jsonl").write_text("other", encoding="utf-8")
    assert "changed" in json.loads(tools.thomas_encoder_train(args))["error"]
    assert not (runs_dir / "tamper").exists()


def test_rehashing_both_mutable_files_cannot_forge_approved_input(tmp_path, fake_thomas, runs_dir):
    source = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(source), "calib_size": 10, "run_name": "forged"}
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    pending = runs_dir / "_approvals" / "forged"
    changed = source.read_bytes().replace(b"text a 0", b"different a 0")
    (pending / "data.jsonl").write_bytes(changed)
    manifest = json.loads((pending / "approval.json").read_text())
    manifest["data_sha256"] = hashlib.sha256(changed).hexdigest()
    (pending / "approval.json").write_text(json.dumps(manifest))
    result = json.loads(tools.thomas_encoder_train(args))
    assert "manifest changed" in result["error"]
    assert not (runs_dir / "forged").exists()


def test_no_process_intent_refuses_stale_denied_or_restarted_snapshot(tmp_path, fake_thomas, runs_dir):
    source = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(source), "calib_size": 10, "run_name": "denied"}
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "approve"
    tools._APPROVAL_INTENTS.clear()  # simulate a denied approval or process restart
    result = json.loads(tools.thomas_encoder_train(args))
    assert "process-local approval intent" in result["error"]
    assert not (runs_dir / "denied").exists()


def test_real_runner_hash_guard_before_thomas_import(tmp_path):
    path = Path(__file__).resolve().parents[1] / "runners" / "train_runner.py"
    spec = importlib.util.spec_from_file_location("train_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (tmp_path / "data.jsonl").write_text("changed", encoding="utf-8")
    cfg = {"data_path": str(tmp_path / "data.jsonl"), "data_sha256": "0" * 64}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert runner.main(tmp_path, digest) == 1
    assert "hash mismatch" in json.loads((tmp_path / "status.json").read_text())["error"]


def test_runner_rejects_data_and_mutable_config_rehashed_together(tmp_path):
    path = Path(__file__).resolve().parents[1] / "runners" / "train_runner.py"
    spec = importlib.util.spec_from_file_location("train_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    snap = tmp_path / "data.jsonl"
    snap.write_bytes(b"original")
    approved = {"data_path": str(snap), "data_sha256": hashlib.sha256(b"original").hexdigest()}
    digest = hashlib.sha256(json.dumps(approved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    snap.write_bytes(b"modified")
    modified = {**approved, "data_sha256": hashlib.sha256(b"modified").hexdigest()}
    (tmp_path / "config.json").write_text(json.dumps(modified), encoding="utf-8")
    assert runner.main(tmp_path, digest) == 1
    assert "approved training config changed" in json.loads((tmp_path / "status.json").read_text())["error"]


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("epochs", 1.5), ("batch_size", -2), ("lr", float("nan")),
    ("lr", float("inf")), ("calib_size", 0), ("seed", -1), ("run_name", ""),
])
def test_invalid_paid_config_blocked_before_launch(tmp_path, runs_dir, field, value):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    args = {"data_path": str(p), "calib_size": 10, "run_name": "invalid", field: value}
    assert tools.credit_gate(tool_name="thomas_encoder_train", args=args)["action"] == "block"
    assert not (runs_dir / "invalid").exists()


def test_dead_runner_reads_as_failed(runs_dir):
    run = runs_dir / "ghost"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"state": "running"}', encoding="utf-8")
    (run / "config.json").write_text('{"pid": 999999}', encoding="utf-8")
    out = json.loads(tools.thomas_run_status({"run_name": "ghost"}))
    assert out["state"] == "failed" and "runner exited" in out["error"]


# --- eval verdict (gonogo on canned predictions) ------------------------------

def test_gonogo_verdict_from_predictions():
    preds = ([{"id": f"p{i}", "text": "t", "expected": "a", "output": "a", "confidence": 0.99}
              for i in range(180)]
             + [{"id": f"q{i}", "text": "t", "expected": "a", "output": "b", "confidence": 0.3}
                for i in range(20)])
    decision, report = tools._gonogo_report(preds, task="canned", target=0.95)
    assert decision.verdict.value == "AUTOMATE WITH REVIEW"
    assert decision.operating_point.n_deferred == 20
    assert len(report.to_dict()["cases"]) == 200


def test_eval_rejects_newer_contract(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    for f, body in {"config.json": "{}", "label2id.json": '{"a": 0}',
                    "temperature.json": '{"temperature": 1.0}',
                    "metrics.json": json.dumps({"contract_version": tools.CONTRACT_VERSION + 1})}.items():
        (art / f).write_text(body, encoding="utf-8")
    p = write_jsonl(tmp_path / "c.jsonl", rows())
    out = json.loads(tools.thomas_encoder_eval({"model_dir": str(art), "cases_path": str(p)}))
    assert "contract_version" in out["error"]


def test_eval_rejects_non_artifact_dir(tmp_path):
    p = write_jsonl(tmp_path / "c.jsonl", rows())
    out = json.loads(tools.thomas_encoder_eval({"model_dir": str(tmp_path), "cases_path": str(p)}))
    assert "not a thomas artifact dir" in out["error"]
