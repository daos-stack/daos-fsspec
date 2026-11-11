# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for cache staleness."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid

import pytest

# Long enough that it cannot expire during the test, so an "invisible" result
# means genuine staleness rather than a race with the clock. Process startup
# plus a DAOS connect is on the order of a second, which is why the default
# 1.0s TTL is useless for demonstrating the effect.
_LONG_TTL = 300.0

# Short enough to wait out, long enough not to expire mid-setup.
_SHORT_TTL = 2.0

_PEER_TIMEOUT = 120.0


def _peer_creates_file(pool: str, container: str, path: str) -> None:
    """Create ``path`` from a separate process with its own DAOS connection.

    Deliberately a fresh interpreter. This is the scenario the cache cannot
    cope with, and it only exists across process boundaries; an in-process
    mutation invalidates the cache correctly.
    """
    program = (
        "import sys\n"
        "from daos_fsspec.core import DFSFileSystem\n"
        "fs = DFSFileSystem(pool=sys.argv[1], container=sys.argv[2])\n"
        "try:\n"
        "    fs.pipe_file(sys.argv[3], b'written by a peer process')\n"
        "finally:\n"
        "    fs.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, pool, container, path],
        capture_output=True,
        text=True,
        timeout=_PEER_TIMEOUT,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"peer process failed to create {path}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _names(entries) -> set[str]:
    return {e.rstrip("/").rsplit("/", 1)[-1] for e in entries}


@pytest.fixture
def peer_dir(dfs_fs):
    """A directory both this process and the peer will work in."""
    path = f"{dfs_fs._test_dir}/peer-{uuid.uuid4().hex[:8]}"
    dfs_fs.mkdir(path)
    return path


@pytest.mark.live
class TestPeerVisibilityWithCachingEnabled:
    """The documented hazard, reproduced rather than asserted from the design."""

    def test_a_peer_file_is_invisible_while_the_entry_is_cached(
        self, daos_pool, daos_container, peer_dir
    ):
        """Enabling the cache means missing a peer's writes for up to the TTL.

        This is the measurement that justifies the default. If it ever stops
        reproducing, the warning in the class docstring of DFSFileSystem is
        overstated and should be re-litigated with this test as the evidence.
        """
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(
            pool=daos_pool,
            container=daos_container,
            use_listings_cache=True,
            listings_expiry_time=_LONG_TTL,
        )
        try:
            assert _names(fs.ls(peer_dir)) == set(), "directory did not start empty"

            _peer_creates_file(daos_pool, daos_container, f"{peer_dir}/from-peer.bin")

            cached = _names(fs.ls(peer_dir))
            assert "from-peer.bin" not in cached, (
                "the peer's file was visible through a warm cache entry; the "
                "staleness window this test exists to document did not occur, so "
                "either the cache is not being consulted or the TTL expired"
            )

            # Invalidation is the documented escape hatch, e.g. after a barrier.
            fs.invalidate_cache()
            assert "from-peer.bin" in _names(fs.ls(peer_dir)), (
                "invalidate_cache() did not expose the peer's file"
            )
        finally:
            fs.close()

    def test_the_ttl_bounds_the_staleness(self, daos_pool, daos_container, peer_dir):
        """Stale is temporary, not indefinite; the TTL is the whole safety case."""
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(
            pool=daos_pool,
            container=daos_container,
            use_listings_cache=True,
            listings_expiry_time=_SHORT_TTL,
        )
        try:
            fs.ls(peer_dir)  # populate
            _peer_creates_file(daos_pool, daos_container, f"{peer_dir}/after-ttl.bin")

            deadline = time.monotonic() + _SHORT_TTL * 5
            while time.monotonic() < deadline:
                if "after-ttl.bin" in _names(fs.ls(peer_dir)):
                    break
                time.sleep(0.1)
            else:
                pytest.fail(
                    f"peer's file still invisible {_SHORT_TTL * 5:.0f}s after a "
                    f"{_SHORT_TTL:.0f}s TTL; the entry is not expiring"
                )
        finally:
            fs.close()


@pytest.mark.live
class TestPeerVisibilityWithTheDefaults:
    """Why off-by-default is the right call."""

    def test_a_peer_file_is_visible_immediately(self, daos_pool, daos_container, peer_dir):
        """No staleness window at all unless you opt in.

        The case the library is actually for; many ranks writing into one
        directory; is correct out of the box.
        """
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool=daos_pool, container=daos_container)
        try:
            # fsspec consumes use_listings_cache into the DirCache rather than
            # keeping it on the filesystem, so this is where the default lives.
            assert fs.dircache.use_listings_cache is False, (
                "the listings cache is no longer off by default; this test and "
                "the rationale in DFSFileSystem's docstring both need revisiting"
            )
            fs.ls(peer_dir)  # would populate a cache, if there were one

            _peer_creates_file(daos_pool, daos_container, f"{peer_dir}/immediate.bin")

            assert "immediate.bin" in _names(fs.ls(peer_dir)), (
                "a peer's file was not visible with caching off; this is the "
                "property that makes the default safe"
            )
        finally:
            fs.close()

    def test_a_local_mutation_is_visible_even_with_caching_on(
        self, daos_pool, daos_container, peer_dir
    ):
        """The limit is cross-process, not caching per se.

        Worth pinning alongside the staleness tests: mutations through *this*
        filesystem invalidate correctly, so single-process code never sees a
        stale listing. That asymmetry is the whole reason the cache is
        defensible at all when it is switched on.
        """
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(
            pool=daos_pool,
            container=daos_container,
            use_listings_cache=True,
            listings_expiry_time=_LONG_TTL,
        )
        try:
            fs.ls(peer_dir)  # populate
            fs.pipe_file(f"{peer_dir}/local.bin", b"written locally")

            assert "local.bin" in _names(fs.ls(peer_dir)), (
                "a local write did not invalidate our own cached listing"
            )
        finally:
            fs.close()
