---
name: thomas-encoder
description: Use when training or rating a small classifier with thomas tools.
version: 0.1.0
author: James Dominguez (keppy)
license: MIT
---

# Training a calibrated encoder with thomas

The loop is: check the data, get approval, train, evaluate on cases the model
never saw, report the gonogo verdict. Skipping a step either wastes a GPU run
or produces a number nobody should trust.

## Procedure

1. **`thomas_check_data` first, always.** It is free. Fix every blocker before
   going further, and read the warnings out to the user — label noise, thin
   labels and duplicates all cap what training can reach.
2. **Keep an eval split the training run never touches.** `calib_size` rows
   are held out for temperature scaling, but they are *not* an eval set: T was
   fit on them. If the user has one file, split it before training (for example
   80/20 by a fixed seed) and train on the 80 only.
3. **Choose a unique `run_name` and call `thomas_encoder_train` through Hermes.**
   The plugin freezes and validates the JSONL before the approval prompt, shows
   its SHA-256 and resolved config, and trains only from those approved bytes.
   The Hermes hook does not gate terminal/Python/Modal commands. Do not try to
   talk the user past a denial or retry with a tweaked config; ask what changed.
   A denied approval may leave a pending snapshot: a new attempt needs a fresh
   run name (or explicit cleanup of a stale `_approvals` entry).
4. **Poll `thomas_run_status`; never launch a second run to "check".** A run
   takes minutes. `failed` comes with a log tail — read it before proposing a
   rerun, because a rerun costs the same money as the first.
5. **`thomas_encoder_eval` on the held-out cases**, with `save_report` set so
   the run can be paired later with `gonogo_compare`.
6. **Report the verdict the way gonogo does**: the interval, not the point
   estimate; the operating point as "abstain below T, handle X% at precision
   [low, high], send the rest to a human"; the calibration error. Keep the
   note that a threshold searched on the eval set is optimistically biased.

## Pitfalls

- A high `calib_accuracy` in `thomas_run_status` is not the eval result. It is
  measured on the rows T was fit on.
- `labels_unseen_in_training` non-empty means the eval contains classes the
  model cannot output; those cases are guaranteed failures. Say so.
- Training on a file that also holds the eval cases makes every number above
  meaningless. Ask where the eval cases came from if it is not obvious.
