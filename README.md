# thomas — a Hermes plugin

Train a small calibrated classifier on a Modal GPU and get a gonogo verdict on
it, without leaving the conversation. Every GPU launch stops at Hermes' human
approval gate with the exact config on screen. Checking the data and running
the eval are local and free.

```bash
hermes plugins install keppy/hermes-plugin-thomas
hermes plugins enable thomas
```

Pairs with [hermes-plugin-gonogo](https://github.com/keppy/hermes-plugin-gonogo):
`thomas_encoder_eval` saves a gonogo report JSON that `gonogo_compare` can pair
against a baseline or a later model.

## What it adds

| Tool | What it does | Costs |
| --- | --- | --- |
| `thomas_check_data` | Validates a `{text, label}` JSONL: schema, label counts, thin labels, duplicates, conflicting labels | free |
| `thomas_encoder_train` | Fine-tunes an encoder on a Modal L4, temperature-scales it on a held-out split, pulls the artifact back. Runs in the background | **GPU time, gated** |
| `thomas_run_status` | running / done / failed, calibration numbers, log tail; lists runs with no argument | free |
| `thomas_encoder_eval` | CPU inference on held-out cases → gonogo verdict, operating point, calibration error | free |

Plus a `pre_tool_call` hook, the credit gate, and a bundled skill,
`thomas-encoder`, that carries the procedure: check first, keep an eval split
the model never saw, poll rather than relaunch, report the interval.

## The credit gate

`thomas_encoder_train` is escalated to Hermes' approval gate on every call, by
a hook that runs before the handler. The model can't skip it. The prompt looks like:

```
thomas: launch a PAID GPU fine-tune — johnnyboycurtis/ModernBERT-small-v2 on
cases.jsonl (9503 rows), 3 epochs, batch 32, lr 2e-05, calib 500, seed 7, L4 (Modal)
```

The allowlist key is a hash of that resolved config. So answering `[a]lways`
approves that one config, and a different model, dataset or epoch count asks
again. A non-interactive session with no approval bridge fails closed.

## Setup

The plugin itself only needs `gonogo-eval`, which Hermes installs from
`pyproject.toml`. Training and inference run in a separate Python so torch and
Modal stay out of the Hermes venv:

```bash
git clone https://github.com/keppy/thomas && cd thomas
uv venv && uv pip install -e ".[encoder]"
.venv/Scripts/modal token new        # or .venv/bin/modal on macOS/Linux
```

Then set `THOMAS_PYTHON` to that interpreter in `$HERMES_HOME/.env`:

```
THOMAS_PYTHON=C:/Users/you/git/thomas/.venv/Scripts/python.exe
```

Runs live under `$HERMES_HOME/thomas/runs/<run_name>/` (`config.json`,
`status.json`, `log.txt`, `artifact/`). Set `THOMAS_RUNS_DIR` to put them
somewhere else.

## Checked against a real artifact

`thomas_encoder_eval` on the Banking77 canary artifact from the
[thomas](https://github.com/keppy/thomas) repo (ModernBERT-small-v2, 250 held-out
cases, target 95%):

```
AUTOMATE WITH REVIEW — overall pass rate 87.2% [82.5%, 90.8%] misses the 95% target,
but abstaining below confidence 0.91 reaches 98.3% precision on 71% of cases
Operating point: threshold 0.912, coverage 71.2%, 72 deferred, precision 98.3% [95.2%, 99.4%]
Calibration error 0.03
Note: The 0.91 threshold was chosen by searching this same case set, so its precision
is optimistically biased. Re-measure it on fresh cases before relying on it.
```

That's the same verdict as the gonogo example script run on the same artifact.
Confidence comes from `thomas.encoder_train.scaled_softmax`, the function the
training run used to fit the temperature, so there's one definition end to end.

## Development

```bash
pip install -e ".[dev]"
pytest
hermes plugins validate . && hermes plugins doctor .
```

The tests need no GPU, no Modal and no thomas install. Training is exercised
with a fake runner subprocess, and the verdict is checked on canned predictions.

## License

MIT
