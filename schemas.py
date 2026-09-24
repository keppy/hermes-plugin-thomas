"""Tool schemas — what the model reads to decide when to call each tool."""

_DATA = {
    "type": "string",
    "description": (
        "Path to a JSONL file, one object per line: {\"text\": ..., \"label\": ...} "
        "and optionally \"id\". Every label must be a string."
    ),
}

THOMAS_CHECK_DATA = {
    "name": "thomas_check_data",
    "description": (
        "Validate a classification dataset before any money is spent on it: row "
        "count, label count, per-label counts, rows that fail the schema, duplicate "
        "texts, labels too thin to learn or to hold out for calibration. Free and "
        "local. Call this first, every time, before thomas_encoder_train — a bad "
        "file found here costs nothing, found after a GPU run it costs the run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": _DATA,
            "calib_size": {
                "type": "integer",
                "description": (
                    "Rows the training run will hold out for temperature scaling "
                    "(default 500). Used to warn when the holdout would eat most "
                    "of a small dataset."
                ),
            },
        },
        "required": ["data_path"],
    },
}

THOMAS_ENCODER_TRAIN = {
    "name": "thomas_encoder_train",
    "description": (
        "Fine-tune a HuggingFace encoder with a classification head on a Modal L4 "
        "GPU, temperature-scale it on a held-out calibration split, and pull the "
        "artifact (model + label2id.json + temperature.json + metrics.json) back to "
        "disk. COSTS MONEY: every call stops at the human approval gate with the "
        "config shown, and runs only if the user approves. Returns immediately with "
        "a run_name — the job runs in the background; poll thomas_run_status. "
        "Run thomas_check_data on the file first. Use this for routing / intent / "
        "skill-tagging style problems where a fast calibrated classifier beats an "
        "LLM call — not for generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_path": _DATA,
            "model": {
                "type": "string",
                "description": (
                    "Base encoder, HF Hub id. Default johnnyboycurtis/ModernBERT-small-v2 "
                    "(the Banking77 canary: 87.2% [82.5%, 90.8%], AUTOMATE WITH REVIEW)."
                ),
            },
            "epochs": {"type": "integer", "description": "Training epochs. Default 3."},
            "batch_size": {"type": "integer", "description": "Rows per step. Default 32."},
            "lr": {"type": "number", "description": "AdamW learning rate. Default 2e-5."},
            "calib_size": {
                "type": "integer",
                "description": "Rows held out (never trained on) to fit the temperature. Default 500.",
            },
            "seed": {"type": "integer", "description": "Holdout + shuffle seed. Default 7."},
            "run_name": {
                "type": "string",
                "description": "Name for the run and its artifact dir. Default enc-<timestamp>.",
            },
        },
        "required": ["data_path"],
    },
}

THOMAS_RUN_STATUS = {
    "name": "thomas_run_status",
    "description": (
        "Check a thomas training run: running / done / failed, the artifact dir "
        "when done, temperature and calibration (ECE before -> after), and the tail "
        "of the log. With no run_name, lists recent runs. Poll this after "
        "thomas_encoder_train instead of launching a second run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_name": {"type": "string", "description": "Run to inspect. Omit to list runs."},
            "log_lines": {"type": "integer", "description": "Log lines to return. Default 20."},
        },
        "required": [],
    },
}

THOMAS_ENCODER_EVAL = {
    "name": "thomas_encoder_eval",
    "description": (
        "Rate a trained encoder on held-out cases with gonogo: run the model on CPU "
        "(free, local), score each case by exact label match with the calibrated "
        "confidence attached, and return the gonogo verdict — pass-rate interval, "
        "the confidence threshold that routes uncertain cases to a human, "
        "calibration error. The cases must NOT be rows the model trained on. Set "
        "save_report to keep the per-case JSON so gonogo_compare can pair this run "
        "against a baseline or a later model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_dir": {
                "type": "string",
                "description": "Artifact dir from thomas_encoder_train (or any dir with the same contract files).",
            },
            "cases_path": _DATA,
            "target": {
                "type": "number",
                "description": "Accuracy the deployment needs, as a fraction. Default 0.95.",
            },
            "task": {"type": "string", "description": "Task name for the report."},
            "save_report": {
                "type": "string",
                "description": "Path to write the gonogo report JSON (per-case outcomes).",
            },
            "write_html": {"type": "string", "description": "Path to write a standalone HTML report."},
        },
        "required": ["model_dir", "cases_path"],
    },
}
