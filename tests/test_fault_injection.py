# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for fault injection."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

# Every fault-handling path must settle within this. Generous enough not to be
# flaky on a loaded machine, far below any plausible hang.
_DEADLINE = 15.0

# Retry backoff is exponential; keep it tiny so exhausting retries stays fast
# while still exercising the real sleep/backoff code rather than skipping it.
_FAST_BACKOFF = 0.001


def _within_deadline(fn, deadline: float = _DEADLINE):
    """Run fn in a thread; return (outcome, exc, elapsed)."""
    box: dict = {}

    def run():
        start = time.monotonic()
        try:
            box["value"] = fn()
            box["outcome"] = "returned"
        except BaseException as exc:  # noqa: BLE001 - recording, not handling
            box["exc"] = exc
            box["outcome"] = "raised"
        finally:
            box["elapsed"] = time.monotonic() - start

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError(
            f"operation did not raise or return within {deadline}s: it hung. "
            "This is the failure mode this module exists to prevent."
        )
    return box.get("outcome"), box.get("exc"), box.get("elapsed", 0.0)


class _FakeDaosError(Exception):
    """Stand-in for B.DaosError.

    A real class rather than Exception: ``_is_retryable()`` isinstance-checks
    against it, and matching every exception would push unrelated failures
    down the retry path and hide them.
    """

    def __init__(self, rc: int, msg: str = ""):
        self.rc = rc
        super().__init__(msg or f"rc={rc}")

    @property
    def is_retryable(self) -> bool:
        return self.rc in _RETRYABLE


# Mirrors B.RETRYABLE_ERRORS for the codes these tests use.
DER_NO_HDL = -1002
DER_NO_PERM = -1001
DER_UNREACH = -1006
DER_TIMEDOUT = -1012
DER_HG = -1020
_RETRYABLE = frozenset({DER_UNREACH, DER_TIMEDOUT, DER_HG})


@pytest.fixture
def fault_fs():
    """A connected DFSFileSystem whose libdfs layer can be made to fail."""
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = _RETRYABLE
    mock_B.DER_UNREACH = DER_UNREACH
    mock_B.DER_HG = DER_HG
    mock_B.DER_NO_HDL = DER_NO_HDL
    mock_B.DER_TIMEDOUT = DER_TIMEDOUT
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755),
    )

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(
            pool="test-pool",
            container="test-cont",
            retry_count=3,
            retry_backoff=_FAST_BACKOFF,
        )
        # Connect-time lookups must not count towards per-test assertions.
        mock_B.reset_mock()
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="root"),
            SimpleNamespace(size=0, mode=0o040755),
        )
        yield fs, mock_B
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server eviction mid-session
# ---------------------------------------------------------------------------


class TestServerEviction:
    """DER_UNREACH on a handle that was valid a moment ago."""

    def test_transient_unreach_is_retried_and_succeeds(self, fault_fs):
        """One DER_UNREACH then success must recover, not surface the error.

        Asserts the call count, not just the return value: if the retry never
        happened and the first call simply succeeded, the value assertion alone
        would pass and the test would be vacuous.
        """
        fs, mock_B = fault_fs
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeDaosError(DER_UNREACH, "server evicted")
            return b"recovered"

        mock_B.fake_op.side_effect = flaky
        with patch.object(fs, "_reconnect") as reconnect:
            outcome, exc, _ = _within_deadline(lambda: fs._retry(mock_B.fake_op))

        assert outcome == "returned", f"expected recovery, got {exc!r}"
        assert calls["n"] == 2, "retry did not happen: test would be vacuous"
        assert reconnect.call_count == 1, "DER_UNREACH must trigger a reconnect"

    def test_persistent_unreach_exhausts_retries_and_raises(self, fault_fs):
        """A permanently evicted server must give up, not retry forever."""
        fs, mock_B = fault_fs
        calls = {"n": 0}

        def always_unreach(*a, **k):
            calls["n"] += 1
            raise _FakeDaosError(DER_UNREACH, "server gone")

        mock_B.fake_op.side_effect = always_unreach
        with patch.object(fs, "_reconnect"):
            outcome, exc, _ = _within_deadline(lambda: fs._retry(mock_B.fake_op))

        assert outcome == "raised"
        assert isinstance(exc, _FakeDaosError)
        # retry_count=3 means 4 attempts total. An off-by-one that looped
        # forever would have hit the deadline above instead.
        assert calls["n"] == 4, f"expected 4 attempts, got {calls['n']}"

    def test_reconnect_failure_surfaces_the_original_error(self, fault_fs):
        """If reconnect also fails, the caller must see the DAOS error."""
        fs, mock_B = fault_fs
        mock_B.fake_op.side_effect = _FakeDaosError(DER_UNREACH, "evicted")

        with patch.object(fs, "_reconnect", side_effect=RuntimeError("no route")):
            outcome, exc, _ = _within_deadline(lambda: fs._retry(mock_B.fake_op))

        assert outcome == "raised"
        assert isinstance(exc, _FakeDaosError), f"original error lost, got {exc!r}"
        assert exc.rc == DER_UNREACH
        assert isinstance(exc.__cause__, RuntimeError), "reconnect failure not chained"

    def test_non_retryable_error_is_not_retried(self, fault_fs):
        """A permission error must fail immediately.

        Retrying a non-transient error wastes the backoff budget and delays a
        deterministic failure. Asserts exactly one attempt.
        """
        fs, mock_B = fault_fs
        calls = {"n": 0}

        def denied(*a, **k):
            calls["n"] += 1
            raise _FakeDaosError(DER_NO_PERM, "denied")

        mock_B.fake_op.side_effect = denied
        with patch.object(fs, "_reconnect") as reconnect:
            outcome, exc, _ = _within_deadline(lambda: fs._retry(mock_B.fake_op))

        assert outcome == "raised"
        assert calls["n"] == 1, f"non-retryable error was retried {calls['n']} times"
        assert reconnect.call_count == 0, "must not reconnect on a permission error"


# ---------------------------------------------------------------------------
# Rank crash with handles open
# ---------------------------------------------------------------------------


class TestRankCrash:
    """DER_NO_HDL: the handle outlived what it referred to."""

    def test_no_hdl_triggers_reconnect_then_raises_if_still_broken(self, fault_fs):
        """DER_NO_HDL is not in RETRYABLE_ERRORS, so it must raise at once."""
        fs, mock_B = fault_fs
        calls = {"n": 0}

        def crashed(*a, **k):
            calls["n"] += 1
            raise _FakeDaosError(DER_NO_HDL, "rank crashed")

        mock_B.fake_op.side_effect = crashed
        with patch.object(fs, "_reconnect") as reconnect:
            outcome, exc, _ = _within_deadline(lambda: fs._retry(mock_B.fake_op))

        assert outcome == "raised"
        assert isinstance(exc, _FakeDaosError) and exc.rc == DER_NO_HDL
        assert calls["n"] == 1
        assert reconnect.call_count == 0

    def test_health_check_reports_false_instead_of_raising(self, fault_fs):
        """health_check() is a predicate; a dead cluster must make it False.

        If it propagated the error, every caller polling connection health
        would need its own try/except and the pool's health sweep would abort
        on the first dead connection.
        """
        fs, mock_B = fault_fs
        mock_B.dfs_stat.side_effect = _FakeDaosError(DER_NO_HDL, "rank crashed")

        outcome, exc, _ = _within_deadline(fs.health_check)

        assert outcome == "returned", f"health_check raised: {exc!r}"
        assert fs.health_check() is False

    def test_health_check_false_after_close_without_touching_libdfs(self, fault_fs):
        """A closed filesystem is unhealthy, and must not call into libdfs."""
        fs, mock_B = fault_fs
        fs.close()
        mock_B.dfs_stat.reset_mock()

        assert fs.health_check() is False
        assert mock_B.dfs_stat.call_count == 0, "health_check touched libdfs after close"


# ---------------------------------------------------------------------------
# Handle lifetime across a reconnect
# ---------------------------------------------------------------------------


class TestHandleLifetimeAcrossReconnect:
    """Handles resolved before a retry must not outlive the mount they came from.

    _reconnect() umounts the filesystem and mounts a fresh one, so every DFS
    object handle obtained from the old mount is dangling afterwards. Any
    operation that resolves a handle *outside* the retry loop and uses it
    *inside* will therefore pass freed memory to libdfs on the second attempt --
    undefined behaviour, not an error code, so there is nothing for a caller to
    catch and nothing in a log to read.

    mv() was the only operation shaped that way. Everything else handed to
    _retry() either takes a path and resolves it internally
    (``_readdirplus_dir``) or is a closure that re-reads ``self._dfs`` and
    re-resolves per call. These tests exist because that is an easy property to
    break by refactoring and an almost impossible one to notice.
    """

    @staticmethod
    def _distinct_roots(mock_B):
        """Make each (re)connect produce a distinguishable root handle.

        The fixture returns one fixed root, which would make the old and new
        mounts compare equal and hide exactly the bug under test.
        """
        counter = {"n": 0}

        def lookup(*a, **k):
            counter["n"] += 1
            return (
                MagicMock(name=f"root-gen{counter['n']}"),
                SimpleNamespace(size=0, mode=0o040755),
            )

        mock_B.dfs_lookup_path.side_effect = lookup

    def test_mv_re_resolves_handles_after_a_reconnect(self, fault_fs):
        """The second dfs_move must use the new mount's root, not the old one."""
        fs, mock_B = fault_fs
        self._distinct_roots(mock_B)
        fs._reconnect()  # establish a known-current root
        root_before = fs._root

        attempts = []

        def flaky_move(dfs, src_parent, src_name, dst_parent, dst_name):
            attempts.append(src_parent)
            if len(attempts) == 1:
                raise _FakeDaosError(DER_UNREACH, "evicted mid-move")

        mock_B.dfs_move.side_effect = flaky_move

        outcome, exc, _ = _within_deadline(lambda: fs.mv("/a", "/b"))

        assert outcome == "returned", f"mv raised: {exc!r}"
        assert len(attempts) == 2, f"expected one retry, got {len(attempts)} attempts"
        assert fs._root is not root_before, "reconnect did not replace the root handle"
        assert attempts[0] is root_before, "first attempt used an unexpected handle"
        assert attempts[1] is fs._root, (
            "second dfs_move reused a handle from the umounted filesystem; "
            "with real libdfs this is a use-after-free, not an error"
        )

    def test_mv_does_not_release_the_old_root_twice(self, fault_fs):
        """_reconnect() already released it; releasing again is a double free.

        _release_parent() skips the root by identity, and after a reconnect
        ``self._root`` is a new object; so an old root held across the retry
        stopped matching and got released a second time. "/a" and "/b" are
        depth-1 paths, where _resolve_path returns the root itself for both
        parents, so the old code released it twice here on top of the release
        inside _reconnect(): three times for one handle.
        """
        fs, mock_B = fault_fs
        self._distinct_roots(mock_B)
        fs._reconnect()
        root_before = fs._root

        attempts = {"n": 0}

        def flaky_move(*a, **k):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _FakeDaosError(DER_UNREACH, "evicted mid-move")

        mock_B.dfs_move.side_effect = flaky_move
        mock_B.dfs_release.reset_mock()

        fs.mv("/a", "/b")

        released_old = [
            c for c in mock_B.dfs_release.call_args_list if c.args and c.args[0] is root_before
        ]
        # Exactly once, not "at most once". `<= 1` also accepted zero, so
        # _reconnect() never releasing the old root; a handle leak on every
        # reconnect; passed this test.
        assert len(released_old) == 1, (
            f"old root released {len(released_old)} times across the reconnect; "
            "more than once is a double free, zero is a leak"
        )

    def test_release_parent_refuses_a_handle_from_an_older_mount(self, fault_fs):
        """A generation-tagged handle from a dead mount must not be released.

        The mount it came from was umounted by the reconnect, which already
        freed it. Releasing again is a double free.
        """
        fs, mock_B = fault_fs
        stale_generation = fs._mount_generation
        stale_parent = MagicMock(name="parent-from-old-mount")
        fs._reconnect()
        assert fs._mount_generation != stale_generation, "reconnect did not bump the generation"

        mock_B.dfs_release.reset_mock()
        fs._release_parent(stale_parent, stale_generation)

        assert mock_B.dfs_release.call_count == 0, (
            "released a handle belonging to a filesystem that was already umounted"
        )

    def test_release_parent_still_releases_a_current_handle(self, fault_fs):
        """The guard must not turn into a blanket refusal."""
        fs, mock_B = fault_fs
        parent = MagicMock(name="current-parent")
        mock_B.dfs_release.reset_mock()

        fs._release_parent(parent, fs._mount_generation)

        assert mock_B.dfs_release.call_count == 1, "a live handle was not released"

    def test_an_open_file_does_not_release_into_a_new_mount(self, fault_fs):
        """The reachable case this guard exists for.

        Nothing in the read/write path retries, so a file cannot trigger a
        reconnect itself; but another thread calling ls() can, while this
        file is open. That umounts the filesystem the file's handles came from.
        Closing afterwards used to call dfs_close on freed memory.
        """
        fs, mock_B = fault_fs
        mock_B.dfs_open_file.return_value = MagicMock(name="file-obj")
        mock_B.dfs_lookup_rel_path.return_value = (
            MagicMock(name="file-obj"),
            SimpleNamespace(size=0, mode=0o100644),
        )

        handle = fs.open("/some-file", "rb")
        assert handle._mount_generation == fs._mount_generation

        fs._reconnect()  # another thread's retry, while the file is open
        mock_B.dfs_close.reset_mock()
        mock_B.dfs_release.reset_mock()

        handle.close()

        assert mock_B.dfs_close.call_count == 0, (
            "closed a file handle belonging to an umounted filesystem"
        )
        assert mock_B.dfs_release.call_count == 0, "released a parent from the old mount"
        assert handle._dfs_obj is None, "stale handle reference was not dropped"

    def test_the_root_handle_is_never_released_as_a_parent(self, fault_fs):
        """Even when the root arrives as a different wrapper at the same address.

        The old check was `parent != self._root`, which on c_void_p is a
        wrapper-identity test; two wrappers around the same root pointer
        compared unequal, so the root would have been released while still in
        use. Nothing did that yet, but the check was one refactor away from a
        double free.
        """
        from daos_fsspec._bindings import dfs_obj_p

        fs, mock_B = fault_fs
        fs._root = dfs_obj_p(0xABCD)
        duplicate = dfs_obj_p(0xABCD)
        assert duplicate is not fs._root, "test needs two distinct wrappers"
        assert duplicate != fs._root, "ctypes changed: == is no longer identity-based"

        mock_B.dfs_release.reset_mock()
        fs._release_parent(duplicate)

        assert mock_B.dfs_release.call_count == 0, (
            "released the root handle because it arrived as a different wrapper"
        )

    def test_mv_still_releases_non_root_parents_on_success(self, fault_fs):
        """The fix must not leak: a nested path's parent handle is still freed.

        Guards the opposite failure from the one above; moving the release
        inside the retried callable must not skip it.
        """
        fs, mock_B = fault_fs
        parents = []

        def lookup_rel(dfs, parent, name, flags):
            child = MagicMock(name=f"dir-{name}")
            parents.append(child)
            return (child, SimpleNamespace(size=0, mode=0o040755))

        mock_B.dfs_lookup_rel_path.side_effect = lookup_rel
        mock_B.dfs_release.reset_mock()

        fs.mv("/deep/dir/src", "/deep/dir/dst")

        assert parents, "test did not exercise the nested-path lookup"
        released = {id(c.args[0]) for c in mock_B.dfs_release.call_args_list if c.args}
        leaked = [p for p in parents if id(p) not in released]
        assert not leaked, f"{len(leaked)} intermediate/parent handles were never released"


# ---------------------------------------------------------------------------
# Agent socket unavailable (failure at connect time)
# ---------------------------------------------------------------------------


class TestAgentUnavailable:
    """Connect-time failure, as opposed to mid-session failure."""

    def test_connect_failure_raises_promptly(self):
        """No agent means construction fails fast, not hangs."""
        mock_B = MagicMock()
        mock_B._HAS_DFS_CONNECT = False
        mock_B.DaosError = _FakeDaosError
        mock_B.RETRYABLE_ERRORS = _RETRYABLE
        mock_B.pool_connect.side_effect = _FakeDaosError(DER_UNREACH, "no agent socket")

        with patch_bindings(mock_B):
            from daos_fsspec.core import DFSFileSystem

            outcome, exc, _ = _within_deadline(
                lambda: DFSFileSystem(pool="p", container="c", retry_backoff=_FAST_BACKOFF)
            )

        assert outcome == "raised"
        assert isinstance(exc, _FakeDaosError)

    def test_reconnect_failure_leaves_filesystem_closed(self, fault_fs):
        """A failed reconnect must not leave a half-open filesystem."""
        fs, mock_B = fault_fs
        mock_B.pool_connect.side_effect = _FakeDaosError(DER_UNREACH, "agent gone")

        outcome, exc, _ = _within_deadline(fs._reconnect)

        assert outcome == "raised"
        assert fs._closed is True, "filesystem left open after a failed reconnect"
        assert fs.health_check() is False


# ---------------------------------------------------------------------------
# Unmount while a write is in flight
# ---------------------------------------------------------------------------


class TestUnmountDuringIO:
    """close() racing live I/O: the one invariant the exclusive lock exists for."""

    def test_close_waits_for_inflight_operation(self, fault_fs):
        """close() must not tear down handles under a running call.

        The exclusive lock makes close() wait for in-flight shared holders. If
        it did not, the write would be operating on freed handles: a
        use-after-free, which in C is a crash or silent corruption rather than
        an exception.
        """
        fs, mock_B = fault_fs
        started = threading.Event()
        release = threading.Event()
        state = {"write_finished_before_close": None}

        def slow_write(*a, **k):
            started.set()
            release.wait(timeout=_DEADLINE)
            return 10

        mock_B.dfs_write_path.side_effect = slow_write

        def do_write():
            with fs._lock.read():
                return mock_B.dfs_write_path()

        writer = threading.Thread(target=do_write, daemon=True)
        writer.start()
        assert started.wait(timeout=_DEADLINE), "writer never started"

        def do_close():
            state["write_finished_before_close"] = not writer.is_alive()
            # Asserted, not merely recorded. This was written and never read,
            # so whichever way the race went looked the same.
            assert state["write_finished_before_close"] is not None
            fs.close()

        closer = threading.Thread(target=do_close, daemon=True)
        closer.start()
        # close() should still be blocked on the writer's shared hold.
        time.sleep(0.2)
        assert closer.is_alive(), "close() did not wait for the in-flight write"

        release.set()
        writer.join(timeout=_DEADLINE)
        closer.join(timeout=_DEADLINE)
        assert not closer.is_alive(), "close() hung after the write completed"
        assert fs._closed is True

    def test_operation_after_close_raises_rather_than_segfaulting(self, fault_fs):
        """Once closed, every entry point must refuse before touching libdfs."""
        fs, mock_B = fault_fs
        fs.close()
        mock_B.reset_mock()

        outcome, exc, _ = _within_deadline(lambda: fs.cat_file("/whatever"))

        assert outcome == "raised", "a closed filesystem served a read"
        assert mock_B.dfs_read.call_count == 0, "called libdfs after close"
        assert mock_B.dfs_lookup_path.call_count == 0, "looked up a path after close"

    def test_operation_that_passed_precheck_is_rejected_after_close(self, fault_fs):
        """Admission check and shared-lock acquisition must be one operation."""
        fs, mock_B = fault_fs
        passed_precheck = threading.Event()
        outcome = {}
        original_check = fs._check_closed

        def signalled_check():
            original_check()
            passed_precheck.set()

        # Hold the writer so cat_file can pass its old, outer check but cannot
        # yet enter the native-call scope. close() queues behind us and gets
        # writer preference when this hold is released.
        fs._lock.acquire_write()
        fs._check_closed = signalled_check
        mock_B.dfs_lookup_path.reset_mock()

        def read():
            try:
                fs.cat_file("/late-reader")
            except BaseException as exc:  # noqa: BLE001 - asserted below
                outcome["error"] = exc

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        assert passed_precheck.wait(_DEADLINE), "operation never passed its precheck"

        closer = threading.Thread(target=fs.close, daemon=True)
        closer.start()
        deadline = time.monotonic() + _DEADLINE
        while fs._lock._waiting_writers == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert fs._lock._waiting_writers > 0, "close never queued for the lifecycle lock"

        fs._lock.release_write()
        reader.join(_DEADLINE)
        closer.join(_DEADLINE)

        assert not reader.is_alive() and not closer.is_alive()
        assert isinstance(outcome.get("error"), RuntimeError)
        assert mock_B.dfs_lookup_path.call_count == 0, "reader entered libdfs after close"


# ---------------------------------------------------------------------------
# Silent truncation
# ---------------------------------------------------------------------------


class TestNoSilentTruncation:
    """The second half of the contract: failures must not look like success."""

    def test_read_error_is_not_reported_as_empty_data(self, fault_fs):
        """A failed read must raise, never return b""."""
        fs, mock_B = fault_fs
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="file"),
            SimpleNamespace(size=4096, mode=0o100644),
        )
        mock_B.dfs_read.side_effect = _FakeDaosError(DER_UNREACH, "evicted")

        with patch.object(fs, "_reconnect"):
            outcome, exc, _ = _within_deadline(lambda: fs.cat_file("/shard"))

        assert outcome == "raised", f"read error returned data instead: {exc!r}"
        assert mock_B.dfs_read.call_count > 0, "dfs_read was never reached: vacuous"

    def test_short_read_is_not_padded_or_hidden(self, fault_fs):
        """A read returning fewer bytes than requested must not be padded.

        Silent padding would turn a truncated shard into a plausible one. What
        the caller gets must be exactly what DAOS returned.
        """
        fs, mock_B = fault_fs
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="file"),
            SimpleNamespace(size=4096, mode=0o100644),
        )
        mock_B.dfs_read.return_value = b"short"

        data = fs.cat_file("/shard")

        assert data == b"short", f"read result was altered: {data!r}"


# ---------------------------------------------------------------------------
# Pool-level health
# ---------------------------------------------------------------------------


class TestPoolHealthCheck:
    """DFSFileSystemPool.health_check: flagged as entirely unexercised."""

    def test_reports_each_connection_independently(self):
        """One dead connection must not hide the healthy ones.

        health_check() snapshots under the lock and probes outside it, so a
        raising connection must not abort the sweep.
        """
        from daos_fsspec.pool import DFSFileSystemPool

        healthy = SimpleNamespace(health_check=lambda: True)
        dead = SimpleNamespace(health_check=lambda: False)

        pool = object.__new__(DFSFileSystemPool)
        pool._lock = threading.RLock()
        pool._all = [healthy, dead, healthy]

        outcome, exc, _ = _within_deadline(pool.health_check)
        assert outcome == "returned", f"health_check raised: {exc!r}"

        results = pool.health_check()
        assert results == {
            "connection_0": True,
            "connection_1": False,
            "connection_2": True,
        }

    def test_survives_a_connection_whose_probe_raises(self):
        """A connection that raises rather than returning False."""
        from daos_fsspec.pool import DFSFileSystemPool

        def explode():
            raise _FakeDaosError(DER_NO_HDL, "handle gone")

        pool = object.__new__(DFSFileSystemPool)
        pool._lock = threading.RLock()
        pool._all = [SimpleNamespace(health_check=explode)]

        pool._all = [
            SimpleNamespace(health_check=explode),
            SimpleNamespace(health_check=lambda: True),
        ]

        outcome, exc, _ = _within_deadline(pool.health_check)

        # `assert outcome in ("returned", "raised")` is what this used to say,
        # which the helper satisfies by construction; no implementation could
        # fail it, and the docstring above asserted the opposite of what the
        # code did.
        assert outcome == "returned", f"the sweep aborted instead of reporting: {exc!r}"
        results = pool.health_check()
        assert results == {"connection_0": False, "connection_1": True}, (
            f"a raising member lost the report for the others: {results}"
        )


# ---------------------------------------------------------------------------
# Live: teardown racing real I/O
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveFaultInjection:
    """Runs against a real cluster, on both matrix versions.

    Deliberately narrow. Server eviction and rank crash need `dmg`, which is in
    the server image and not reachable from the client container, so those stay
    simulated above. What *is* genuinely reproducible in-process is teardown
    racing live I/O: and that is also the one case where getting it wrong
    means a use-after-free rather than an exception.

    These use their own connections, never the session-scoped `dfs_fs` fixture,
    which is shared and intentionally never closed.
    """

    def test_close_during_concurrent_writes_never_hangs_or_corrupts(
        self, daos_pool, daos_container, dfs_fs
    ):
        """Hammer one filesystem with writes, then close() underneath them."""
        from daos_fsspec.core import DFSFileSystem

        base = f"{dfs_fs._test_dir}/fault-close"
        dfs_fs.mkdir(base)

        fs = DFSFileSystem(pool=daos_pool, container=daos_container)
        errors: list[BaseException] = []
        completed = {"n": 0}
        stop = threading.Event()
        lock = threading.Lock()

        def writer(idx: int):
            i = 0
            while not stop.is_set():
                try:
                    fs.pipe_file(f"{base}/w-{idx}-{i}.bin", b"x" * 256)
                    with lock:
                        completed["n"] += 1
                except Exception as exc:  # noqa: BLE001 - recording outcomes
                    with lock:
                        errors.append(exc)
                    return
                i += 1

        threads = [threading.Thread(target=writer, args=(n,), daemon=True) for n in range(4)]
        for t in threads:
            t.start()

        # Let real I/O get going before pulling the rug out.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with lock:
                if completed["n"] >= 4:
                    break
            time.sleep(0.05)

        fs.close()
        stop.set()
        for t in threads:
            t.join(timeout=_DEADLINE)
            assert not t.is_alive(), "writer hung after close(): must fail, not block"

        # Errors after close are expected and correct. What must not happen is
        # a hang (asserted above) or a crash (the process would be gone).
        assert completed["n"] > 0, "no writes completed; the test proved nothing"

    def test_health_check_is_false_after_close_live(self, daos_pool, daos_container):
        """The real health probe against a genuinely closed connection."""
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool=daos_pool, container=daos_container)
        assert fs.health_check() is True, "fresh connection reported unhealthy"

        fs.close()
        outcome, exc, _ = _within_deadline(fs.health_check)
        assert outcome == "returned", f"health_check raised after close: {exc!r}"
        assert fs.health_check() is False

    def test_operations_after_close_raise_promptly_live(self, daos_pool, daos_container):
        """Every entry point must refuse on a closed connection, quickly.

        Against real torn-down DAOS handles, so a missing closed-check shows up
        as a segfault here rather than a tidy exception.
        """
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool=daos_pool, container=daos_container)
        fs.close()

        for name, call in [
            ("ls", lambda: fs.ls("/")),
            ("info", lambda: fs.info("/")),
            ("cat_file", lambda: fs.cat_file("/nope")),
            ("pipe_file", lambda: fs.pipe_file("/nope", b"x")),
        ]:
            outcome, exc, _ = _within_deadline(call)
            assert outcome == "raised", f"{name}() succeeded on a closed filesystem"


# ---------------------------------------------------------------------------
# A reconnect landing inside a multi-chunk read or write
# ---------------------------------------------------------------------------


class TestReconnectDuringChunkedIO:
    """Multi-chunk I/O must notice the mount it started on has gone.

    ``_read_pipelined``, ``get_file`` and ``put_file`` all release the lock
    between chunks on purpose, so other operations can interleave. That window
    is exactly when another thread's retryable error can send ``_retry`` into
    ``_reconnect()``, which umounts the filesystem and mounts a fresh one. The
    object handle the in-flight operation holds then belongs to a mount that no
    longer exists, and the next chunk handed it to libdfs; a use-after-free,
    not an error.

    ``_mount_generation`` existed for this, but only teardown consulted it;
    the read and write paths did not. They now raise ``MountChangedError``,
    which ``_is_retryable`` treats as transient so ``_retry`` restarts the
    operation on the new mount rather than surfacing it.

    These advance ``_mount_generation`` directly between chunks rather than
    calling ``_reconnect()``. That is the condition the guard tests, and it is
    the only way to drive it deterministically: a real reconnect runs on
    another thread and cannot proceed until the in-flight chunk releases the
    shared lock, and calling ``_reconnect()`` from inside the read; which
    holds that lock; hits the lock-upgrade guard instead. Simulating the
    effect keeps the test about the guard rather than about thread timing.
    """

    @staticmethod
    def _reconnect_effect(fs):
        """What an in-flight operation observes when another thread reconnects."""
        fs._mount_generation += 1
        fs._dfs = MagicMock(name="dfs-after-reconnect")

    def test_a_chunked_read_raises_rather_than_using_a_dead_handle(self, fault_fs):
        """Reconnect between chunk one and chunk two."""
        from daos_fsspec.core import MountChangedError

        fs, mock_B = fault_fs
        chunk = 1 << 20
        mock_B.CHUNK_SIZE = chunk
        fs._PIPELINE_MIN_BYTES = chunk  # pipeline even a small read
        obj = MagicMock(name="file-obj")
        reads: list[int] = []

        def read(dfs, handle, off, length):
            reads.append(off)
            if len(reads) == 1:
                self._reconnect_effect(fs)
            return b"\0" * min(length, chunk)

        mock_B.dfs_read.side_effect = read

        with pytest.raises(MountChangedError):
            fs._read_pipelined(obj, 0, chunk * 4)

        assert len(reads) >= 1, "the test never reached a read"

    def test_cat_file_carries_lookup_generation_into_the_read(self, fault_fs):
        """Reconnect in the lookup-to-read gap must not bless the old object."""
        fs, mock_B = fault_fs
        from daos_fsspec.core import MountChangedError

        fs._retry_count = 0
        obj = MagicMock(name="old-mount-file")
        mock_B.dfs_lookup_path.return_value = (
            obj,
            SimpleNamespace(size=1024, mode=0o100644, is_dir=False),
        )
        real_read = fs._read_pipelined

        def reconnect_in_gap(handle, offset, size, *, generation=None):
            self._reconnect_effect(fs)
            return real_read(handle, offset, size, generation=generation)

        mock_B.dfs_read.reset_mock()
        mock_B.dfs_release.reset_mock()
        with patch.object(fs, "_read_pipelined", side_effect=reconnect_in_gap):
            with pytest.raises(MountChangedError):
                fs.cat_file("/old.bin")

        assert mock_B.dfs_read.call_count == 0, "read used an object from the old mount"
        assert not any(
            call.args and call.args[0] is obj for call in mock_B.dfs_release.call_args_list
        ), "released an object after its mount was replaced"

    def test_resolved_parent_generation_is_checked_before_use(self, fault_fs):
        """Reconnect after path resolution must stop the following mutation."""
        fs, mock_B = fault_fs
        from daos_fsspec.core import MountChangedError

        real_resolve = fs._resolve_path

        def reconnect_after_resolve(path):
            resolved = real_resolve(path)
            self._reconnect_effect(fs)
            return resolved

        mock_B.dfs_open_file.reset_mock()
        with patch.object(fs, "_resolve_path", side_effect=reconnect_after_resolve):
            with pytest.raises(MountChangedError):
                fs.pipe_file("/stale-parent.bin", b"payload")

        assert mock_B.dfs_open_file.call_count == 0, "used a parent from the old mount"

    def test_open_file_read_checks_its_original_mount_generation(self, fault_fs):
        fs, mock_B = fault_fs
        from daos_fsspec.core import MountChangedError

        mock_B.dfs_lookup_rel_path.return_value = (
            MagicMock(name="open-file-object"),
            SimpleNamespace(size=1024, mode=0o100644),
        )
        handle = fs.open("/open.bin", "rb")
        self._reconnect_effect(fs)
        mock_B.dfs_read.reset_mock()
        try:
            with pytest.raises(MountChangedError):
                handle._fetch_range(0, 1)
        finally:
            handle.close()

        assert mock_B.dfs_read.call_count == 0, "open file read through a stale handle"

    def test_open_file_write_checks_its_original_mount_generation(self, fault_fs):
        fs, mock_B = fault_fs
        from daos_fsspec.core import MountChangedError

        mock_B.dfs_open_file.return_value = MagicMock(name="open-file-object")
        handle = fs.open("/open.bin", "wb")
        handle.write(b"payload")
        self._reconnect_effect(fs)
        mock_B.dfs_write.reset_mock()
        try:
            with pytest.raises(MountChangedError):
                handle.flush(force=True)
        finally:
            handle.close()

        assert mock_B.dfs_write.call_count == 0, "open file wrote through a stale handle"

    def test_the_error_is_retryable_so_retry_restarts_the_operation(self, fault_fs):
        """It must be transient, not terminal.

        Every callable handed to _retry re-resolves its path, so a restart runs
        entirely on the new mount. Surfacing the error instead would turn a
        recoverable reconnect into a user-visible failure.
        """
        fs, _ = fault_fs
        from daos_fsspec.core import MountChangedError

        exc = MountChangedError("a read")

        assert fs._is_retryable(exc) is True, (
            "MountChangedError is not retryable, so _retry will not restart "
            "the operation on the new mount"
        )

    def test_a_download_does_not_release_a_handle_from_the_old_mount(self, fault_fs, tmp_path):
        """get_file's finally must not free what the reconnect already freed."""
        from daos_fsspec.core import MountChangedError

        fs, mock_B = fault_fs
        mock_B._CHUNK_SIZE = 1 << 20
        obj = MagicMock(name="remote-obj")
        mock_B.dfs_lookup_path.return_value = (
            obj,
            SimpleNamespace(
                size=4 << 20,
                mode=0o100644,
                uid=0,
                gid=0,
                nlink=1,
                mtime=0.0,
                atime=0.0,
                ctime=0.0,
                is_dir=False,
                type_name="file",
            ),
        )
        calls: list[int] = []

        def read(dfs, handle, off, length):
            calls.append(off)
            if len(calls) == 1:
                self._reconnect_effect(fs)
            return b"\0" * length

        mock_B.dfs_read.side_effect = read
        mock_B.dfs_release.reset_mock()

        with pytest.raises(MountChangedError):
            fs.get_file("/remote.bin", str(tmp_path / "local.bin"))

        released = [c for c in mock_B.dfs_release.call_args_list if c.args and c.args[0] is obj]
        assert not released, (
            "released the source handle after the reconnect had already torn "
            "down the mount it came from"
        )

    def test_an_upload_stops_rather_than_writing_into_the_new_mount(self, fault_fs, tmp_path):
        """A write is worse than a read: it would corrupt the new mount."""
        from daos_fsspec.core import MountChangedError

        fs, mock_B = fault_fs
        mock_B._CHUNK_SIZE = 1 << 10
        local = tmp_path / "upload.bin"
        local.write_bytes(b"x" * (4 << 10))
        writes: list[int] = []

        def write(dfs, handle, off, data):
            writes.append(off)
            if len(writes) == 1:
                self._reconnect_effect(fs)

        mock_B.dfs_write.side_effect = write

        with pytest.raises(MountChangedError):
            fs.put_file(str(local), "/remote.bin")

        assert len(writes) == 1, f"kept writing after the mount was replaced: {len(writes)} writes"
