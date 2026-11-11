# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for async concurrency."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings

# Long enough that ops genuinely overlap, short enough to keep tests quick.
_OP_SECONDS = 0.05
# Generous multiple of any legitimate runtime: only a real deadlock reaches it.
_JOIN_TIMEOUT = 30.0


class _FakeDaosError(Exception):
    is_retryable = False


def _mock_bindings() -> MagicMock:
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755),
    )
    return mock_B


def _slow_op(*args, **kwargs):
    """Stand-in for a DAOS call: takes real time, releases the GIL."""
    time.sleep(_OP_SECONDS)
    return ["/entry"]


def _run_scenario(body, size: int, op=_slow_op, timeout: float = _JOIN_TIMEOUT):
    """Run ``body(fs)`` against a pooled async filesystem of ``size`` handles.

    Owns the whole lifecycle: patches the bindings, builds the filesystem,
    runs the coroutine on a loop in a daemon thread, then closes the pool and
    collects; all before the patch is lifted, for the reason in the module
    docstring. Fails rather than hangs if the loop makes no progress.
    """
    mock_B = _mock_bindings()
    box: dict[str, object] = {}

    async def scenario():
        from daos_fsspec.async_core import DFSAsyncFileSystem

        fs = DFSAsyncFileSystem(pool="p", container="c", size=size)
        for conn in fs._pool._all:
            conn.ls = op
        try:
            return await body(fs)
        finally:
            # Inside the patch and inside the loop: close() dispatches through
            # to_thread, so it needs the loop still running.
            await fs.close()

    def target():
        try:
            box["result"] = asyncio.run(scenario())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    with patch_bindings(mock_B):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        alive = thread.is_alive()
        # Reap anything the loop dropped while the mock is still installed.
        gc.collect()

    if alive:
        pytest.fail(
            f"DFSAsyncFileSystem made no progress in {timeout:.0f}s; the event "
            "loop is blocked waiting for a pooled connection that only the "
            "event loop could release. See _run() in async_core.py."
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["result"]


@pytest.mark.parametrize(
    ("pool_size", "n_ops"),
    [
        (4, 4),  # exactly saturated: the only case that worked before
        (2, 4),  # mild oversubscription
        (4, 8),  # the README's gather example against a default pool
        (1, 8),  # a single connection, fully serialised
        (4, 40),  # far more work than connections
    ],
)
def test_concurrency_beyond_pool_size_does_not_deadlock(pool_size, n_ops):
    """More concurrent operations than connections must queue, not hang."""

    async def body(fs):
        return await asyncio.gather(*[fs._ls("/") for _ in range(n_ops)])

    results = _run_scenario(body, pool_size)

    assert len(results) == n_ops
    assert all(r == ["/entry"] for r in results), "an operation returned the wrong result"


def test_operations_actually_overlap_across_connections():
    """All four workers must enter their operation before any can finish."""
    entered = threading.Barrier(4, timeout=10)

    def synchronized_op(*args, **kwargs):
        entered.wait()
        return ["/entry"]

    async def body(fs):
        return await asyncio.gather(*[fs._ls("/") for _ in range(4)])

    results = _run_scenario(body, 4, op=synchronized_op)
    assert results == [["/entry"]] * 4


def test_the_event_loop_stays_responsive_while_the_pool_is_saturated():
    """The property underneath the deadlock, stated directly.

    Even with every connection checked out and more work queued, unrelated
    coroutines must keep being scheduled. If the loop thread is blocked in
    pool.acquire() this heartbeat stops dead.
    """

    async def body(fs):
        ticks = 0
        stop = False

        async def heartbeat():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await asyncio.gather(*[fs._ls("/") for _ in range(8)])
        stop = True
        await beat
        return ticks

    ticks = _run_scenario(body, 2)

    # 8 ops over 2 connections is ~4 * _OP_SECONDS of wall clock; a 10ms
    # heartbeat should fire many times. Asserted well below that to stay robust
    # on a loaded CI box, while still being zero if the loop ever froze.
    assert ticks > 5, f"event loop ticked only {ticks} times while the pool was saturated"


def test_connections_are_returned_to_the_pool_after_use():
    """A leaked connection shrinks the pool until it deadlocks again.

    The borrow now happens inside the worker thread; the release must happen
    there too, on every path.
    """

    async def body(fs):
        await asyncio.gather(*[fs._ls("/") for _ in range(9)])
        return len(fs._pool._available_queue), len(fs._pool._in_use)

    available, in_use = _run_scenario(body, 3)

    assert in_use == 0, f"{in_use} connections never returned to the pool"
    assert available == 3, f"pool holds {available} connections, expected 3"


def test_a_failing_operation_still_returns_its_connection():
    """An exception must not strand the connection it borrowed."""

    def boom(*args, **kwargs):
        raise _FakeDaosError("operation failed")

    async def body(fs):
        failures = 0
        for _ in range(4):
            try:
                await fs._ls("/")
            except _FakeDaosError:
                failures += 1
        return failures, len(fs._pool._available_queue), len(fs._pool._in_use)

    failures, available, in_use = _run_scenario(body, 3, op=boom)

    assert failures == 4
    assert in_use == 0, "a failed operation kept its connection"
    assert available == 3


def test_async_namespace_validation():
    async def body(fs):
        assert fs._strip_protocol("dfs://p/c/file") == "/file"
        with pytest.raises(ValueError, match="does not match"):
            fs._strip_protocol("dfs://other/other/file")
        with pytest.raises(ValueError, match="does not match"):
            await fs._cat_file("dfs://other/other/file")

    _run_scenario(body, 1)
