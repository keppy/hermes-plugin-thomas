# thomas — a Hermes plugin

Train a small calibrated classifier on a Modal GPU and get a gonogo verdict on
it, without leaving the conversation. Launches through `thomas_encoder_train`
use Hermes' human approval gate; direct Python, terminal and Modal commands
are **outside this hook's scope**. Checking data and CPU eval are local and free.

## The ecosystem this belongs to

This plugin is the in-session half of a small ecosystem that takes a task
from *which model?* to *ship it or not*:

```
your task ──► evalroute ─ choose a (model, effort) arm;
                │         measured where available, priors marked
                ▼
your cases ──► thomas ── train on labels (encoder) or score_text (RL);
                │         evaluate on untouched cases with gonogo
                ▼
              gonogo ── ship it, test a threshold on fresh cases, or walk away
```

- **[thomas](https://github.com/keppy/thomas)** — the library behind this
  plugin. RL shares `score_text` across training and held-out evaluation;
  encoder SFT instead learns labels and needs a separate held-out evaluation.
- **[gonogo](https://github.com/keppy/gonogo)** — the decision layer, and the
  root of the map. Any agent, your real cases, a target; the verdict comes
  with the interval behind it.
- **[evalroute](https://github.com/keppy/hermes-plugin-evalroute)** — the
  routing layer: classify the task, hand back the arm with measured
  costs where available and priors otherwise; collect ratings to prioritize
  the next controlled batch.
- **The Hermes plugins** — the same three, inside your agent's session:
  [gonogo](https://github.com/keppy/hermes-plugin-gonogo) for decisions,
  this one's training tool with a human approval hook, and evalroute's
  `/route` before the first turn.

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

`thomas_encoder_train` calls through Hermes' `pre_tool_call` approval hook.
Choose a unique `run_name` for each attempt. Before asking, the hook validates
the hyperparameters and data, freezes the input bytes, and shows the input
SHA-256 alongside the resolved config and run name. The runner reads only
that snapshot (and checks its hash again before any Modal call), even if the
original file changes during approval. Invalid requests are blocked without
a launch. The rule key includes a fresh approval nonce as well as config and
data hash, so `[a]lways` does **not** silently approve repeated launches.
The approval intent is held outside the editable snapshot manifest in the
Hermes process. Editing both files cannot change the approved digest; a Hermes
restart invalidates pending intents, so use a fresh run name. The detached runner
also receives an approved config digest on its command line: editing both its
config and data files cannot authorize different bytes before a Modal call.
A run name is never reused; polling a run does not relaunch it.

This protection applies to plugin tool calls through Hermes, **not** a
`terminal` invocation, direct library import or Modal CLI. There is no claim
that the hook can intercept those routes. In a non-interactive session with
no approval bridge, the Hermes tool call fails closed. A denied prompt may
leave a pending snapshot under `_approvals/`; use a fresh run name for a new
approval or explicitly remove the stale entry. Run data (including the
snapshot) stays local under the runs directory.

## Contract

The plugin reads thomas artifacts against thomas's
[docs/CONTRACT.md](https://github.com/keppy/thomas/blob/main/docs/CONTRACT.md):
the confidence definition, the artifact layout, the split rules. It supports
**contract version 1** (thomas 0.2.x). An artifact with a newer
`contract_version` in its `metrics.json` is refused with a message to update
the plugin, rather than read wrong.

## Setup

The plugin itself only needs `gonogo-eval`, which Hermes installs from
`pyproject.toml`. Training and inference run in a separate Python so torch and
Modal stay out of the Hermes venv:

This plugin revision needs `thomas-train` v0.2.1 (the contract it reads
shipped there) and `gonogo-eval` 0.3; thomas v0.2.0 is not compatible:

```bash
git clone --branch v0.2.1 https://github.com/keppy/thomas && cd thomas
uv venv && uv pip install -e ".[encoder]"
.venv/Scripts/modal token new        # or .venv/bin/modal on macOS/Linux
```

Set the **nonsecret runtime setting** `THOMAS_PYTHON` in the environment
of the process that starts Hermes, then restart Hermes to inherit it. For
example, from Git Bash on Windows:

```bash
export THOMAS_PYTHON="C:/path/to/thomas/.venv/Scripts/python.exe"
hermes
```

On macOS/Linux, use the venv's `bin/python` instead. The plugin itself
registers without this setting; only train and CPU eval need it. Do **not**
put it in Hermes' `.env`: `requires_env` is the install-time credential
prompt, and the interpreter path is neither a secret nor a plugin admission
requirement. The catalog correctly keeps `requires_env: []`.

Runs live under `$HERMES_HOME/thomas/runs/<run_name>/` (`config.json`,
`status.json`, `log.txt`, `artifact/`). Set `THOMAS_RUNS_DIR` to put them
somewhere else.

## Checked against a real artifact

`thomas_encoder_eval` was run on the original **local-only** Banking77
ModernBERT-small-v2 artifact and 250 held-out cases (target 95%). The model
and case text are **not tracked** in either repository or available at an
advertised download URL. The thomas repo ships a text-free
[per-case prediction receipt](https://github.com/keppy/thomas/blob/main/examples/receipts/banking77_predictions.jsonl)
and CPU replay script; these reproduce the gonogo decision without the
checkpoint, but do not reproduce inference from an independently downloaded
model:

```
AUTOMATE WITH REVIEW — overall pass rate 87.2% [82.5%, 90.8%] misses the 95% target,
but abstaining below confidence 0.91 reaches 98.3% precision on 71% of cases
Operating point: threshold 0.912, coverage 71.2%, 72 deferred, precision 98.3% [95.2%, 99.4%]
Calibration error 0.03
Note: The 0.91 threshold was chosen by searching this same case set, so its precision
is optimistically biased. Re-measure it on fresh cases before relying on it.
```

That's the same verdict as the gonogo example script run on the same artifact.

A previous live run through the plugin, start to finish (not repeated in this review): `thomas_check_data` →
`thomas_encoder_train` (approval gate) → Modal L4 → `thomas_run_status` →
`thomas_encoder_eval`. The config was deliberately tiny (801 training rows,
about 10 per label, 1 epoch), so the model is weak and the verdict says so:

```
DO NOT AUTOMATE — pass rate 15.2% [11.3%, 20.2%] is below 50%
(trained in 85 s; T = 0.60; calibration error 0.07; chance is 1.3% on 77 labels)
```

The first artifact pull hit Modal's stale-volume snapshot and the built-in retry
recovered, which is why the retry is there.
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
