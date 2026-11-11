# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for dircache."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings


class _FakeDaosError(Exception):
    is_retryable = False


def _stat(size: int = 10, mode: int = 0o100644, type_name: str = "file"):
    return SimpleNamespace(
        size=size,
        type_name=type_name,
        mode=mode,
        mtime=1.0,
        atime=1.0,
        uid=0,
        gid=0,
        is_dir=(type_name == "directory"),
    )


_UNSET = object()


def _entry(name: str, stat=_UNSET):
    """A DirEntry stand-in. Pass stat=None to mean "readdirplus gave no stat".

    A sentinel rather than a None default: None is the value under test, so a
    `stat if stat is not None else _stat()` default would silently hand back a
    real stat and make the fabricate-branch test vacuous. It did, at first.
    """
    return SimpleNamespace(name=name, stat=_stat() if stat is _UNSET else stat)


@pytest.fixture
def fs_factory():
    """Build a DFSFileSystem over a mocked binding layer.

    Yields a factory so individual tests can choose cache settings.
    """
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (MagicMock(name="obj"), _stat(type_name="directory"))
    # Must return a real 2-tuple: _resolve_path unpacks it, and a bare MagicMock
    # makes every mutating call raise ValueError. The invalidation tests would
    # then fail because the mutation never happened, not because invalidation
    # was missing; which is how this was first written.
    mock_B.dfs_lookup_rel_path.return_value = (
        MagicMock(name="child"),
        _stat(type_name="directory"),
    )
    mock_B.dfs_open_file.return_value = MagicMock(name="fileobj")
    mock_B.dfs_create_symlink.return_value = MagicMock(name="link")
    mock_B.dfs_readdirplus.return_value = [_entry("a.bin"), _entry("b.bin")]

    created = []

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        def make(**kwargs):
            # Caching is OFF by default as of the dircache measurements, so this
            # suite opts in explicitly. Without that, every invalidation test
            # below would pass for the wrong reason; "the mutation dropped the
            # cached listing" is trivially satisfied when nothing was cached.
            # TestDefaultIsOff covers the default itself.
            kwargs.setdefault("use_listings_cache", True)
            fs = DFSFileSystem(pool="p", container="c", **kwargs)
            created.append(fs)
            mock_B.dfs_readdirplus.reset_mock()
            return fs

        yield make, mock_B

        for fs in created:
            try:
                if not fs._closed:
                    fs.close()
            except Exception:
                pass


class TestDefaultIsOff:
    """Caching is opt-in. This is the decision the measurements produced."""

    def test_caching_is_off_by_default(self):
        """A plain DFSFileSystem must not cache listings."""
        from daos_fsspec.core import DFSFileSystem

        mock_B = MagicMock()
        mock_B._HAS_DFS_CONNECT = False
        mock_B.DaosError = _FakeDaosError
        mock_B.RETRYABLE_ERRORS = ()
        mock_B.pool_connect.return_value = MagicMock(name="poh")
        mock_B.cont_open.return_value = MagicMock(name="coh")
        mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="obj"),
            _stat(type_name="directory"),
        )
        mock_B.dfs_readdirplus.return_value = [_entry("a.bin")]

        with patch_bindings(mock_B):
            fs = DFSFileSystem(pool="p", container="c")
            try:
                assert fs.dircache.use_listings_cache is False
                mock_B.dfs_readdirplus.reset_mock()

                fs.ls("/d", detail=True)
                fs.ls("/d", detail=True)
                fs.ls("/d", detail=True)

                assert mock_B.dfs_readdirplus.call_count == 3, (
                    "listings were cached by default; caching is meant to be opt-in"
                )
                assert len(fs.dircache) == 0
            finally:
                try:
                    fs.close()
                except Exception:
                    pass

    def test_the_ttl_default_still_applies_when_opted_in(self, fs_factory):
        """Turning caching on must not inherit fsspec's never-expire default."""
        make, _ = fs_factory
        fs = make(use_listings_cache=True)

        assert fs.dircache.use_listings_cache is True
        assert fs.dircache.listings_expiry_time == 1.0


class TestCacheHits:
    def test_second_ls_does_not_hit_the_network(self, fs_factory):
        make, mock_B = fs_factory
        fs = make()

        first = fs.ls("/d", detail=True)
        second = fs.ls("/d", detail=True)

        assert first == second
        assert mock_B.dfs_readdirplus.call_count == 1, "cache was not consulted"

    def test_name_only_listing_is_derived_from_the_cached_detail(self, fs_factory):
        """detail=False must reuse the cached detailed listing.

        Caching the two shapes separately would double the round trips and let
        them drift apart.
        """
        make, mock_B = fs_factory
        fs = make()

        detailed = fs.ls("/d", detail=True)
        names = fs.ls("/d", detail=False)

        assert names == [e["name"] for e in detailed]
        assert mock_B.dfs_readdirplus.call_count == 1

    def test_detail_true_is_served_from_a_detail_false_call(self, fs_factory):
        """And the other way round: the first call populates for both shapes."""
        make, mock_B = fs_factory
        fs = make()

        names = fs.ls("/d", detail=False)
        detailed = fs.ls("/d", detail=True)

        assert [e["name"] for e in detailed] == names
        assert mock_B.dfs_readdirplus.call_count == 1

    def test_different_directories_are_cached_separately(self, fs_factory):
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d1")
        fs.ls("/d2")
        fs.ls("/d1")
        fs.ls("/d2")

        assert mock_B.dfs_readdirplus.call_count == 2

    def test_explicit_opt_out_disables_caching_entirely(self, fs_factory):
        make, mock_B = fs_factory
        fs = make(use_listings_cache=False)

        fs.ls("/d", detail=True)
        fs.ls("/d", detail=True)
        fs.ls("/d", detail=True)

        assert mock_B.dfs_readdirplus.call_count == 3, "caching was not disabled"


class TestExpiry:
    def test_the_default_ttl_is_one_second_not_forever(self, fs_factory):
        """fsspec's DirCache defaults to no expiry, which is unusable here."""
        make, _ = fs_factory
        fs = make()

        assert fs.dircache.listings_expiry_time == 1.0

    def test_an_entry_expires_and_is_refetched(self, fs_factory):
        """Staleness must be bounded, which is what makes the tradeoff tenable."""
        make, mock_B = fs_factory
        fs = make(listings_expiry_time=0.15)

        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 1
        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 1, "cached as expected"

        time.sleep(0.3)
        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2, "entry never expired"

    def test_a_peer_write_becomes_visible_after_the_ttl(self, fs_factory):
        """The documented peer-visibility contract, asserted rather than assumed."""
        make, mock_B = fs_factory
        fs = make(listings_expiry_time=0.15)

        assert len(fs.ls("/d", detail=True)) == 2

        # A peer creates c.bin. We are not told.
        mock_B.dfs_readdirplus.return_value = [
            _entry("a.bin"),
            _entry("b.bin"),
            _entry("c.bin"),
        ]

        assert len(fs.ls("/d", detail=True)) == 2, "cache should still be serving"
        time.sleep(0.3)
        assert len(fs.ls("/d", detail=True)) == 3, "peer write never became visible"


class TestInvalidationOnMutation:
    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda fs: fs.pipe_file("/d/new.bin", b"x"), id="pipe_file"),
            pytest.param(lambda fs: fs.touch("/d/new.bin"), id="touch"),
            pytest.param(lambda fs: fs.mkdir("/d/sub", create_parents=False), id="mkdir"),
            pytest.param(lambda fs: fs.rm("/d/a.bin"), id="rm"),
            pytest.param(lambda fs: fs.rmdir("/d/sub"), id="rmdir"),
            pytest.param(lambda fs: fs.mv("/d/a.bin", "/d/b2.bin"), id="mv"),
            pytest.param(lambda fs: fs.chmod("/d/a.bin", 0o600), id="chmod"),
            pytest.param(lambda fs: fs.chown("/d/a.bin", 1000, 1000), id="chown"),
            pytest.param(lambda fs: fs.truncate("/d/a.bin", 1), id="truncate"),
            pytest.param(lambda fs: fs.symlink("/d/link", "/d/a.bin"), id="symlink"),
        ],
    )
    def test_local_mutation_invalidates_the_parent_listing(self, fs_factory, mutate):
        """Single-process code must never observe a stale listing."""
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 1

        mutate(fs)

        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2, (
            "mutation did not invalidate the cached listing for /d"
        )

    def test_buffered_write_leaves_a_fresh_listing(self, fs_factory):
        """fs.open(..., "wb") must not be the one path that leaves a stale size."""
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 1

        with fs.open("/d/streamed.bin", "wb") as f:
            f.write(b"payload")

        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2, (
            "a buffered write left the parent listing cached"
        )

    def test_upload_chunk_invalidates_mid_write(self, fs_factory):
        """Each flushed chunk must invalidate, not just the final close().

        fsspec invalidates on close. That leaves a window on a long multi-chunk
        upload: list the directory after the first chunk lands and the cached
        size is already wrong, and it stays wrong for the whole TTL while the
        remaining chunks are still being written. Invalidating per chunk closes
        that window.
        """
        make, mock_B = fs_factory
        fs = make()

        f = fs.open("/d/streamed.bin", "wb")
        try:
            f.write(b"first chunk")
            fs.ls("/d", detail=True)
            before = mock_B.dfs_readdirplus.call_count

            # Flush a chunk without closing the file.
            f.flush(force=True)

            fs.ls("/d", detail=True)
            assert mock_B.dfs_readdirplus.call_count == before + 1, (
                "a flushed chunk left a stale listing cached mid-write"
            )
        finally:
            f.close()

    def test_recursive_rm_clears_everything(self, fs_factory):
        """recursive rm clears everything."""
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        fs.ls("/d/sub", detail=True)
        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 3

        fs.rm("/d", recursive=True)

        assert len(fs.dircache) == 0, "recursive rm left listings cached"

    def test_invalidate_cache_drops_everything(self, fs_factory):
        """The documented escape hatch for picking up peer writes early."""
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2

        fs.invalidate_cache()

        fs.ls("/d", detail=True)
        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 4

    def test_set_prefix_drops_every_cached_path(self, fs_factory):
        """A mount prefix changes the interpretation of every cache key."""
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        fs.ls("/other", detail=True)
        assert len(fs.dircache) == 2

        fs.set_prefix("/new-root")

        assert len(fs.dircache) == 0
        fs.ls("/d", detail=True)
        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 4

    def test_invalidate_cache_for_one_path_leaves_others(self, fs_factory):
        make, mock_B = fs_factory
        fs = make()

        fs.ls("/d", detail=True)
        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2

        fs.invalidate_cache("/d")

        fs.ls("/other", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 2, "/other was dropped too"
        fs.ls("/d", detail=True)
        assert mock_B.dfs_readdirplus.call_count == 3, "/d was not dropped"


class TestNeverCacheFabricatedMetadata:
    def test_a_listing_with_a_missing_stat_is_not_cached(self, fs_factory):
        """The one case where caching would make a lie authoritative."""
        make, mock_B = fs_factory
        fs = make()
        mock_B.dfs_readdirplus.return_value = [_entry("a.bin"), _entry("broken", stat=None)]

        first = fs.ls("/d", detail=True)
        second = fs.ls("/d", detail=True)

        assert any(e["size"] == 0 and "mode" not in e for e in first), (
            "test did not actually exercise the fabricate branch"
        )
        assert first == second
        assert mock_B.dfs_readdirplus.call_count == 2, (
            "a listing containing fabricated metadata was cached"
        )
        assert len(fs.dircache) == 0

    def test_a_fully_stat_ed_listing_is_cached(self, fs_factory):
        """Control for the test above: the normal case must still cache."""
        make, mock_B = fs_factory
        fs = make()
        mock_B.dfs_readdirplus.return_value = [_entry("a.bin"), _entry("b.bin")]

        fs.ls("/d", detail=True)
        fs.ls("/d", detail=True)

        assert mock_B.dfs_readdirplus.call_count == 1
        assert len(fs.dircache) == 1


class TestRecursiveWalksStayFresh:
    def test_find_does_not_consult_the_cache(self, fs_factory):
        """find() is deliberately uncached, so it cannot mix fresh and stale.

        If this ever starts passing with a lower count, the recursive paths have
        been routed through the cache and the docstring needs revisiting.
        """
        make, mock_B = fs_factory
        fs = make()
        mock_B.dfs_readdirplus.return_value = [_entry("a.bin")]

        fs.ls("/d", detail=True)
        before = mock_B.dfs_readdirplus.call_count

        fs.find("/d")

        assert mock_B.dfs_readdirplus.call_count > before, (
            "find() served a cached listing; it is documented as always-fresh"
        )


@pytest.mark.live
class TestDircacheLive:
    """Against a real cluster, on both matrix versions."""

    def test_cache_hit_is_faster_and_mutations_are_visible(self, dfs_fs):
        """End-to-end on real DAOS: caching works and does not lie locally."""
        from daos_fsspec.core import DFSFileSystem

        base = f"{dfs_fs._test_dir}/dircache-live"
        dfs_fs.mkdir(base)
        for i in range(50):
            dfs_fs.pipe_file(f"{base}/f-{i:03d}.bin", b"x" * (i + 1))

        fs = DFSFileSystem(
            pool=dfs_fs._pool_label,
            container=dfs_fs._cont_label,
            # Explicit: caching is off by default, so a test that only set a TTL
            # would silently exercise the uncached path and prove nothing.
            use_listings_cache=True,
            listings_expiry_time=30.0,
        )
        try:
            cold = fs.ls(base, detail=True)
            assert len(cold) == 50
            warm = fs.ls(base, detail=True)
            assert warm == cold

            # Sizes must be real, not the fabricated 0.
            by_name = {e["name"].rsplit("/", 1)[-1]: e for e in warm}
            assert by_name["f-000.bin"]["size"] == 1
            assert by_name["f-049.bin"]["size"] == 50

            # A local write must be visible at once despite the 30s TTL.
            fs.pipe_file(f"{base}/added.bin", b"new")
            after = fs.ls(base, detail=True)
            assert len(after) == 51, "local write not visible through the cache"

            # And a local delete.
            fs.rm(f"{base}/added.bin")
            assert len(fs.ls(base, detail=True)) == 50
        finally:
            fs.close()

    def test_peer_write_needs_invalidate_cache(self, dfs_fs):
        """The documented peer limitation, on real DAOS.

        Two independent connections stand in for two ranks. The second writes;
        the first keeps serving its cached listing until told to drop it.
        """
        from daos_fsspec.core import DFSFileSystem

        base = f"{dfs_fs._test_dir}/dircache-peer"
        dfs_fs.mkdir(base)
        dfs_fs.pipe_file(f"{base}/original.bin", b"x")

        reader = DFSFileSystem(
            pool=dfs_fs._pool_label,
            container=dfs_fs._cont_label,
            # Explicit: caching is off by default, so a test that only set a TTL
            # would silently exercise the uncached path and prove nothing.
            use_listings_cache=True,
            listings_expiry_time=30.0,
        )
        writer = DFSFileSystem(pool=dfs_fs._pool_label, container=dfs_fs._cont_label)
        try:
            assert len(reader.ls(base)) == 1

            writer.pipe_file(f"{base}/from-peer.bin", b"y")

            # Still stale: nothing local told the reader to drop its cache.
            assert len(reader.ls(base)) == 1, (
                "peer write appeared immediately; the cache is not actually caching"
            )

            reader.invalidate_cache()
            assert len(reader.ls(base)) == 2, "invalidate_cache() did not expose the peer write"
        finally:
            reader.close()
            writer.close()
