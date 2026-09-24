"""Contract tests: what plugin.yaml declares vs what register() actually does."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_plugin_package():
    spec = importlib.util.spec_from_file_location(
        "thomas_plugin_under_test", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self.skills: dict[str, Path] = {}
        self.hooks: dict[str, list] = {}

    def register_tool(self, name, toolset=None, schema=None, handler=None, **_):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}

    def register_skill(self, name, path, **_):
        self.skills[name] = Path(path)

    def register_hook(self, name, callback, **_):
        self.hooks.setdefault(name, []).append(callback)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ctx() -> FakeContext:
    context = FakeContext()
    load_plugin_package().register(context)
    return context


def test_declared_tools_match_registered(manifest, ctx):
    assert sorted(manifest["provides_tools"]) == sorted(ctx.tools)


def test_declared_hooks_match_registered(manifest, ctx):
    assert sorted(manifest["provides_hooks"]) == sorted(ctx.hooks)


def test_schema_names_match_tool_names(ctx):
    for name, entry in ctx.tools.items():
        assert entry["schema"]["name"] == name
        assert entry["toolset"] == "thomas"
        assert callable(entry["handler"])
        for req in entry["schema"]["parameters"].get("required", []):
            assert req in entry["schema"]["parameters"]["properties"]


def test_bundled_skill_registered(ctx):
    assert "thomas-encoder" in ctx.skills
    text = ctx.skills["thomas-encoder"].read_text(encoding="utf-8")
    assert text.startswith("---") and "name: thomas-encoder" in text


def test_train_description_says_it_costs_money(ctx):
    assert "COSTS MONEY" in ctx.tools["thomas_encoder_train"]["schema"]["description"]
