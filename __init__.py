"""thomas plugin — train a calibrated encoder on Modal, rate it with gonogo.

Registers four tools, one bundled skill, and one ``pre_tool_call`` hook. The
hook is the credit gate: ``thomas_encoder_train`` launches a paid GPU job, so
every call escalates to Hermes' human-approval gate with the exact config in
the prompt. The model cannot skip it — the hook runs before the handler, and
a non-interactive session with no approval bridge fails closed.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from . import schemas, tools
except ImportError:  # pragma: no cover - pytest imports the plugin root as a top-level module
    import schemas  # type: ignore
    import tools  # type: ignore

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent / "skills"


def register(ctx):
    """Wire schemas to handlers, install the credit gate, register the skill."""
    for name, schema, handler in (
        ("thomas_check_data", schemas.THOMAS_CHECK_DATA, tools.thomas_check_data),
        ("thomas_encoder_train", schemas.THOMAS_ENCODER_TRAIN, tools.thomas_encoder_train),
        ("thomas_run_status", schemas.THOMAS_RUN_STATUS, tools.thomas_run_status),
        ("thomas_encoder_eval", schemas.THOMAS_ENCODER_EVAL, tools.thomas_encoder_eval),
    ):
        ctx.register_tool(name=name, toolset="thomas", schema=schema, handler=handler)

    ctx.register_hook("pre_tool_call", tools.credit_gate)

    if _SKILLS_DIR.is_dir():
        for child in sorted(_SKILLS_DIR.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md)
