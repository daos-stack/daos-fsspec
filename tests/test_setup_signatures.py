# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for setup signatures."""

from __future__ import annotations

import pytest

from daos_fsspec import _bindings as B

# Symbols the detection probes for, and the flag each one sets.
_CAPABILITIES = [
    ("dfs_connect", "_HAS_DFS_CONNECT"),
    ("dfs_connect_snap", "_HAS_SNAP"),
    ("dfs_set_prefix", "_HAS_SET_PREFIX"),
    ("dfs_suggest_oclass", "_HAS_SUGGEST_OCLASS"),
    ("dfs_dup", "_HAS_DUP"),
    ("dfs_punch", "_HAS_PUNCH"),
]


class _FakeLib:
    """A CDLL stand-in that only has the symbols it was given.

    ctypes raises AttributeError for a missing symbol, and that is exactly
    what the capability probes catch. Anything present behaves like a function
    pointer whose argtypes/restype can be assigned.
    """

    def __init__(self, symbols):
        self._symbols = set(symbols)
        self._funcs: dict[str, _FakeFunc] = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._symbols:
            raise AttributeError(f"undefined symbol: {name}")
        return self._funcs.setdefault(name, _FakeFunc(name))


class _FakeFunc:
    """Records which attributes were *assigned*, not just their values."""

    def __init__(self, name):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "declared", set())

    def __setattr__(self, key, value):
        self.declared.add(key)
        object.__setattr__(self, key, value)

    def __call__(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise AssertionError(f"{self.name} should not be called during setup")


def _all_symbols():
    """Every symbol _setup_signatures touches, discovered by running it."""
    found: set[str] = set()

    class _Recorder(_FakeLib):
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            found.add(name)
            return self._funcs.setdefault(name, _FakeFunc(name))

    daos, dfs = _Recorder(()), _Recorder(())
    B._setup_signatures(daos, dfs)
    return found


@pytest.fixture
def restore_flags():
    """_setup_signatures writes module globals; put them back."""
    saved = {flag: getattr(B, flag) for _, flag in _CAPABILITIES}
    yield
    for flag, value in saved.items():
        setattr(B, flag, value)


class TestSignatureDeclaration:
    def test_it_runs_against_a_complete_library(self, restore_flags):
        """The baseline: a full libdfs must configure without raising."""
        symbols = _all_symbols()
        daos, dfs = _FakeLib(symbols), _FakeLib(symbols)

        B._setup_signatures(daos, dfs)  # must not raise

    def test_core_calls_get_argtypes_and_restype(self, restore_flags):
        """An undeclared signature means ctypes guesses the marshalling."""
        symbols = _all_symbols()
        daos, dfs = _FakeLib(symbols), _FakeLib(symbols)

        B._setup_signatures(daos, dfs)

        for name in ("dfs_read", "dfs_write", "dfs_lookup", "dfs_release"):
            func = getattr(dfs, name)
            assert "argtypes" in func.declared, f"{name} has no argtypes declared"
            assert "restype" in func.declared, f"{name} has no restype declared"

        for name in ("daos_init", "daos_fini"):
            func = getattr(daos, name)
            assert "restype" in func.declared, f"{name} has no restype declared"
            assert "argtypes" in func.declared, f"{name} has no argtypes declared"

        # DAOS 2.6/2.8 declare int daos_fini(void); retain cleanup failures.
        assert daos.daos_fini.restype is daos.daos_init.restype
        assert daos.daos_init.restype is not None, "daos_init returns int"


class TestCapabilityDetection:
    def test_a_complete_library_enables_every_capability(self, restore_flags):
        symbols = _all_symbols()

        B._setup_signatures(_FakeLib(symbols), _FakeLib(symbols))

        for symbol, flag in _CAPABILITIES:
            assert getattr(B, flag) is True, f"{flag} is False although {symbol} is present"

    @pytest.mark.parametrize(("symbol", "flag"), _CAPABILITIES)
    def test_a_missing_symbol_disables_exactly_its_own_flag(self, restore_flags, symbol, flag):
        """An older libdfs must degrade, not crash; and only where it must.

        The important half is "only its own": the probes share try blocks, so
        one missing symbol disabling an unrelated capability is an easy
        mistake and an invisible one, since both paths still work.
        """
        symbols = _all_symbols() - {symbol}

        B._setup_signatures(_FakeLib(symbols), _FakeLib(symbols))

        assert getattr(B, flag) is False, (
            f"{flag} stayed True although {symbol} is missing; the library "
            "would call a symbol that does not exist"
        )

    def test_dfs_connect_detection_decides_the_connect_path(self, restore_flags):
        """_HAS_DFS_CONNECT is the one with a visible consequence.

        core._connect branches on it, so a detection bug silently changes
        which of the two connect-and-teardown shapes every mount uses.
        """
        complete = _all_symbols()

        B._setup_signatures(_FakeLib(complete), _FakeLib(complete))
        assert B._HAS_DFS_CONNECT is True

        older = complete - {"dfs_connect"}
        B._setup_signatures(_FakeLib(older), _FakeLib(older))
        assert B._HAS_DFS_CONNECT is False, (
            "an older libdfs without dfs_connect would still be sent down the dfs_connect path"
        )
