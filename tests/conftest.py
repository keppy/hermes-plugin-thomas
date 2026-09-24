"""Test setup.

The plugin root is `hermes-plugin-thomas` — not a valid Python identifier — so
a test runner that walks the package tree reaches for the root ``__init__.py``
as a top-level module. Putting the plugin root on ``sys.path`` here (before any
test module is imported) is what makes its top-level ``import schemas`` /
``import tools`` fallback resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))