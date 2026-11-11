# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for api surface."""

from __future__ import annotations

import importlib
import inspect
import json
import pathlib
import re
import sys

import pytest

# Script mode puts tests/ on sys.path, not the repo root, so a bare import
# would pick up whatever daos_fsspec happens to be installed, which may be a
# different checkout. Pin it to the tree this file lives in.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SNAPSHOT = pathlib.Path(__file__).parent / "api_surface.json"

# Modules whose public surface users can reasonably depend on. _bindings is
# included because it is the ctypes layer: a dropped symbol there is an ABI
# break, which is the most expensive kind of regression in this project.
MODULES = (
    "daos_fsspec",
    "daos_fsspec.core",
    "daos_fsspec.dfs_file",
    "daos_fsspec.pool",
    "daos_fsspec.async_core",
    "daos_fsspec.metrics",
    "daos_fsspec._bindings",
    "daos_fsspec._rwlock",
    "daos_fsspec._fork",
    "daos_fsspec._fs_cache",
)


_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _sig(obj) -> str:
    """Signature as a stable string, or a marker when it cannot be read.

    Memory addresses are normalised: several fsspec parameters default to object
    instances whose repr embeds one, and an un-normalised snapshot would fail on
    every other run.
    """
    try:
        return _ADDR.sub(" at 0xADDR", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "<no signature>"


def _owned_by_us(value) -> bool:
    """True if this member is implemented inside daos_fsspec."""
    mod = getattr(value, "__module__", None)
    if mod is None and isinstance(value, property) and value.fget is not None:
        mod = getattr(value.fget, "__module__", None)
    return bool(mod) and mod.split(".")[0] == "daos_fsspec"


def _members(cls) -> dict[str, str]:
    """Public members of ``cls`` that daos_fsspec implements, with signatures."""
    out: dict[str, str] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue  # private helpers and dunders are not our public surface
        static = inspect.getattr_static(cls, name, None)
        if isinstance(static, property):
            if _owned_by_us(static):
                out[name] = "<property>"
            continue
        if isinstance(static, (staticmethod, classmethod)):
            if _owned_by_us(static.__func__):
                out[name] = f"<{type(static).__name__}>{_sig(static.__func__)}"
            continue
        value = getattr(cls, name, None)
        if callable(value) and _owned_by_us(value):
            out[name] = _sig(value)
    return out


def collect() -> dict:
    """Build the current surface description."""
    surface: dict = {}
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        entry: dict = {}

        exported = getattr(mod, "__all__", None)
        if exported is not None:
            entry["__all__"] = sorted(exported)

        classes: dict = {}
        functions: dict = {}
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            if inspect.isclass(value) and value.__module__ == mod_name:
                classes[name] = _members(value)
            elif inspect.isfunction(value) and getattr(value, "__module__", None) == mod_name:
                functions[name] = _sig(value)

        if classes:
            entry["classes"] = {k: classes[k] for k in sorted(classes)}
        if functions:
            entry["functions"] = {k: functions[k] for k in sorted(functions)}
        if entry:
            surface[mod_name] = entry
    return surface


def _diff(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences, oriented at what a reviewer needs to judge."""
    problems: list[str] = []

    def walk(path: str, exp, act):
        if isinstance(exp, dict) and isinstance(act, dict):
            for key in sorted(set(exp) - set(act)):
                problems.append(f"REMOVED  {path}{key}")
            for key in sorted(set(act) - set(exp)):
                problems.append(f"ADDED    {path}{key}")
            for key in sorted(set(exp) & set(act)):
                walk(f"{path}{key}.", exp[key], act[key])
        elif exp != act:
            name = path.rstrip(".")
            problems.append(f"CHANGED  {name}\n             was: {exp}\n             now: {act}")

    walk("", expected, actual)
    return problems


def test_public_api_surface_is_unchanged():
    """The public surface must match the committed snapshot exactly."""
    assert SNAPSHOT.exists(), (
        f"{SNAPSHOT.name} is missing. Generate it with: python3 tests/test_api_surface.py --update"
    )
    expected = json.loads(SNAPSHOT.read_text())
    actual = collect()

    problems = _diff(expected, actual)
    assert not problems, (
        "public API surface changed relative to tests/api_surface.json:\n\n  "
        + "\n  ".join(problems)
        + "\n\nIf deliberate, regenerate with "
        "`python3 tests/test_api_surface.py --update` and note it in CHANGELOG.md."
    )


@pytest.mark.parametrize(
    "name",
    [
        # A short, hand-written list of the methods most likely to be lost in a
        # move, independent of the generated snapshot. If the snapshot is ever
        # regenerated carelessly, this still fails.
        "ls",
        "info",
        "cat_file",
        "pipe_file",
        "find",
        "walk",
        "du",
        "mkdir",
        "rm",
        "mv",
        # `_open`, not `open`: fsspec's open() is generic and delegates here,
        # so we deliberately do not override it. Listing `open` asserted
        # nothing about our code.
        "_open",
        "exists",
        "invalidate_cache",
        "get_file",
        "put_file",
        "close",
    ],
)
def test_core_filesystem_keeps_its_essential_methods(name):
    """Belt-and-braces against a carelessly regenerated snapshot."""
    from daos_fsspec.core import DFSFileSystem

    member = getattr(DFSFileSystem, name, None)
    assert callable(member), f"DFSFileSystem.{name} is missing or not callable"

    # Ours, not merely inherited. 15 of these 16 names also exist on
    # fsspec.AbstractFileSystem, so a plain callable() check passed even with
    # our override deleted; only `close` would have failed. Deleting
    # DFSListingMixin.du still gave "16 passed". The override is the point:
    # inheriting fsspec's generic implementation silently replaces a
    # readdirplus-based method with the slow walk-and-stat base version.
    owner = getattr(member, "__module__", "")
    assert owner.startswith("daos_fsspec"), (
        f"DFSFileSystem.{name} resolves to {owner}.{name}, not to a "
        "daos_fsspec implementation; our override was removed and fsspec's "
        "base implementation is being inherited instead"
    )


def test_dfs_protocol_is_still_registered():
    """The dfs:// protocol must survive any reshuffling of modules."""
    import fsspec

    from daos_fsspec.core import DFSFileSystem

    cls = fsspec.get_filesystem_class("dfs")
    assert cls is DFSFileSystem, f"dfs:// resolves to {cls!r}, not DFSFileSystem"


if __name__ == "__main__":
    if "--update" in sys.argv:
        SNAPSHOT.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {SNAPSHOT}")
    else:
        print(json.dumps(collect(), indent=2, sort_keys=True))
