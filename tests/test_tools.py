"""Handler tests. No GPU, no Modal, no thomas install: the training subprocess
is replaced by a fake runner, and eval runs gonogo on canned predictions."""

from __future__ import annotations

import json
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
                          args={"data_path": str(p), "epochs": 5}, task_id="t")
    assert d["action"] == "approve"
    assert "PAID" in d["message"] and "60 rows" in d["message"] and "5 epochs" in d["message"]


def test_gate_rule_key_changes_with_config(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    k1 = tools.credit_gate(tool_name="thomas_encoder_train", args={"data_path": str(p)})["rule_key"]
    k2 = tools.credit_gate(tool_name="thomas_encoder_train",
                           args={"data_path": str(p), "epochs": 9})["rule_key"]
    k3 = tools.credit_gate(tool_name="thomas_encoder_train", args={"data_path": str(p)})["rule_key"]
    assert k1 != k2 and k1 == k3


# --- train + status (fake runner) --------------------------------------------

FAKE_RUNNER = """
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
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
    json.loads(tools.thomas_encoder_train(args))
    assert "already exists" in json.loads(tools.thomas_encoder_train(args))["error"]


def test_train_rejects_path_like_run_name(tmp_path, fake_thomas):
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    out = json.loads(tools.thomas_encoder_train(
        {"data_path": str(p), "calib_size": 10, "run_name": "../escape"}))
    assert "run_name" in out["error"]


def test_train_without_thomas_python(tmp_path, monkeypatch):
    monkeypatch.delenv("THOMAS_PYTHON", raising=False)
    p = write_jsonl(tmp_path / "d.jsonl", rows())
    out = json.loads(tools.thomas_encoder_train({"data_path": str(p), "calib_size": 10}))
    assert "THOMAS_PYTHON" in out["error"]


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
