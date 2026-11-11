# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for binding patch targets."""

from __future__ import annotations

import ast
import pathlib

from conftest import BINDING_MODULES

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "daos_fsspec"


def _modules_importing_bindings() -> set[str]:
    """Modules that bind ``_bindings`` to a module-level name.

    Parsed rather than imported: importing every module to introspect it would
    need the DAOS libraries present, which is the situation this guard exists
    to protect.
    """
    found: set[str] = set()
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "_bindings.py":
            continue  # the bindings themselves
        tree = ast.parse(path.read_text(), filename=str(path))

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "daos_fsspec":
                if any(a.name == "_bindings" for a in node.names):
                    found.add(f"daos_fsspec.{path.stem}")
            elif isinstance(node, ast.Import):
                if any(a.name == "daos_fsspec._bindings" for a in node.names):
                    found.add(f"daos_fsspec.{path.stem}")
    return found


def test_every_module_importing_bindings_is_registered():
    """conftest.BINDING_MODULES must cover every module that imports B."""
    importers = _modules_importing_bindings()
    registered = set(BINDING_MODULES)

    unregistered = sorted(importers - registered)
    assert not unregistered, (
        "these modules import the ctypes bindings but are not in "
        "conftest.BINDING_MODULES, so patch_bindings() will not reach them and "
        f"mocked tests may call real DAOS: {unregistered}"
    )


def test_no_stale_entries_in_the_registry():
    """And nothing registered that no longer imports B.

    A stale entry is harmless at runtime but misleading: it suggests coverage
    that is not needed, and hides whether the list is actually maintained.
    """
    importers = _modules_importing_bindings()
    stale = sorted(set(BINDING_MODULES) - importers)
    assert not stale, (
        f"conftest.BINDING_MODULES lists modules that no longer import the bindings: {stale}"
    )


def test_patch_bindings_actually_replaces_B_everywhere():
    """Behavioural check, not just a list comparison.

    The list could be right while the helper is broken, so assert that every
    registered module really sees the substitute object.
    """
    import importlib
    from unittest.mock import MagicMock

    from conftest import patch_bindings

    sentinel = MagicMock(name="sentinel_bindings")
    with patch_bindings(sentinel):
        for name in BINDING_MODULES:
            module = importlib.import_module(name)
            assert module.B is sentinel, f"{name}.B was not patched"

    # And restored afterwards.
    for name in BINDING_MODULES:
        module = importlib.import_module(name)
        assert module.B is not sentinel, f"{name}.B stayed patched after exit"


def test_patch_bindings_does_not_depend_on_package_attribute_import_order():
    """A selected test module must work before package submodules are exposed."""
    import importlib
    from unittest.mock import MagicMock

    from conftest import patch_bindings

    import daos_fsspec

    listing = importlib.import_module("daos_fsspec._listing")
    sentinel = MagicMock(name="sentinel_bindings")
    saved = getattr(daos_fsspec, "_listing", None)
    if hasattr(daos_fsspec, "_listing"):
        delattr(daos_fsspec, "_listing")
    try:
        with patch_bindings(sentinel):
            assert listing.B is sentinel
    finally:
        if saved is not None:
            daos_fsspec._listing = saved


# ── _bindings.pyi must not drift ─────────────────────────────────────────────


def test_the_stub_file_declares_every_public_binding():
    """A .pyi fully shadows its module for mypy, so a gap is invisible.

    Anything added to ``_bindings.py`` and not mirrored into
    ``_bindings.pyi`` simply does not exist as far as importers' type checking
    is concerned; and mypy stays green, because nothing is wrong with the
    code it *can* see. ``dfs_readdir`` and ``dfs_stat_rel`` sat unstubbed that
    way; the only reason it never bit is that nothing outside the module
    called them. ``same_handle`` would have been the third, caught only
    because mypy happened to flag its first use.

    That makes the stub a silent-drift trap in the one file whose purpose is
    preventing drift. Private (underscore-prefixed) helpers are exempt --
    importers do not use them and the stub is not a second copy of the
    module.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "daos_fsspec"
    module = ast.parse((root / "_bindings.py").read_text())
    stub = ast.parse((root / "_bindings.pyi").read_text())

    def public_functions(tree):
        return {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }

    missing = sorted(public_functions(module) - public_functions(stub))

    assert not missing, (
        f"_bindings.pyi does not declare these public functions: {missing}. "
        "The stub shadows the module for mypy, so importers cannot type-check "
        "against them and nothing fails until someone notices by hand."
    )


def test_the_stub_does_not_declare_functions_that_are_gone():
    """And nothing left over, which would type-check against dead code."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "daos_fsspec"
    module = ast.parse((root / "_bindings.py").read_text())
    stub = ast.parse((root / "_bindings.pyi").read_text())

    module_names = {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
    stub_names = {
        n.name for n in stub.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }

    stale = sorted(stub_names - module_names)
    assert not stale, (
        f"_bindings.pyi declares functions that no longer exist: {stale}; "
        "callers would type-check successfully against code that is gone"
    )
