# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for daos_fsspec.metrics (no DAOS needed)."""

from unittest.mock import patch

import pytest

from daos_fsspec.metrics import DFSFileSystemWithMetrics, DFSMetrics, OperationMetrics


class TestOperationMetrics:
    """OperationMetrics dataclass behavior."""

    def test_record_count_and_duration(self):
        m = OperationMetrics()
        m.record(duration=0.1)
        m.record(duration=0.2)
        assert m.count == 2
        assert m.total_duration == pytest.approx(0.3)
        assert m.min_duration == pytest.approx(0.1)
        assert m.max_duration == pytest.approx(0.2)

    def test_record_error(self):
        m = OperationMetrics()
        m.record(duration=0.1, error=True)
        m.record(duration=0.1)
        assert m.count == 2
        assert m.errors == 1

    def test_record_bytes_read(self):
        m = OperationMetrics()
        m.record(duration=0.1, bytes_count=1024, is_read=True)
        assert m.bytes_read == 1024
        assert m.bytes_written == 0

    def test_record_bytes_written(self):
        m = OperationMetrics()
        m.record(duration=0.1, bytes_count=512, is_read=False)
        assert m.bytes_written == 512
        assert m.bytes_read == 0

    def test_avg_duration_zero_count(self):
        m = OperationMetrics()
        assert m.avg_duration == 0.0

    def test_error_rate_zero_count(self):
        m = OperationMetrics()
        assert m.error_rate == 0.0

    def test_error_rate(self):
        m = OperationMetrics()
        m.record(duration=0.1, error=True)
        m.record(duration=0.1)
        m.record(duration=0.1)
        assert m.error_rate == pytest.approx(1 / 3)


class TestDFSMetrics:
    """DFSMetrics aggregation."""

    def test_summary_empty(self):
        m = DFSMetrics()
        s = m.summary()
        assert s["total_operations"] == 0
        assert s["total_errors"] == 0
        assert s["error_rate"] == 0.0

    def test_summary_with_data(self):
        m = DFSMetrics()
        m.ls.record(duration=0.01)
        m.cat.record(duration=0.05, bytes_count=1024, is_read=True)
        m.write.record(duration=0.02, bytes_count=512, is_read=False)

        s = m.summary()
        assert s["total_operations"] == 3
        assert s["total_bytes_read"] == 1024
        assert s["total_bytes_written"] == 512
        assert "ls" in s["details"]
        assert "cat" in s["details"]
        assert "write" in s["details"]

    def test_reset(self):
        m = DFSMetrics()
        m.ls.record(duration=0.01)
        assert m.ls.count == 1
        m.reset()
        assert m.ls.count == 0
        assert m.ls.total_duration == 0.0


class TestDFSFileSystemWithMetrics:
    """DFSFileSystemWithMetrics — mock-based, no DAOS needed."""

    @patch("daos_fsspec.core.DFSFileSystem.__init__", return_value=None)
    def _make_fs(self, mock_init):
        fs = DFSFileSystemWithMetrics.__new__(DFSFileSystemWithMetrics)
        fs.metrics = DFSMetrics()
        fs._pool_label = "test-pool"
        fs._cont_label = "test-cont"
        fs._readonly = False
        fs._closed = False
        return fs

    def test_metrics_attribute(self):
        fs = self._make_fs()
        assert isinstance(fs.metrics, DFSMetrics)

    def test_ls_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "ls", return_value=["/a", "/b"]):
            result = fs.ls("/")
        assert result == ["/a", "/b"]
        assert fs.metrics.ls.count == 1
        assert fs.metrics.ls.errors == 0

    def test_ls_records_error(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "ls", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                fs.ls("/")
        assert fs.metrics.ls.count == 1
        assert fs.metrics.ls.errors == 1

    def test_cat_records_once_via_cat_file(self):
        # cat() funnels through cat_file (fsspec contract). The metrics wrapper
        # instruments only cat_file, so a single cat() must record exactly once
        # — guards against the old double-counting bug.
        fs = self._make_fs()
        base = DFSFileSystemWithMetrics.__bases__[0]
        with (
            patch.object(base, "cat_file", return_value=b"hello"),
            patch.object(fs, "expand_path", return_value=["/file"]),
        ):
            result = fs.cat("/file")
        assert result == b"hello"
        assert fs.metrics.cat.count == 1
        assert fs.metrics.cat.bytes_read == 5

    def test_pipe_file_records_bytes(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "pipe_file"):
            fs.pipe_file("/file", b"data")
        assert fs.metrics.write.count == 1
        assert fs.metrics.write.bytes_written == 4

    def test_mkdir_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "mkdir"):
            fs.mkdir("/dir")
        assert fs.metrics.mkdir.count == 1

    def test_rm_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "rm"):
            fs.rm("/file")
        assert fs.metrics.rm.count == 1

    def test_mv_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "mv"):
            fs.mv("/a", "/b")
        assert fs.metrics.mv.count == 1

    def test_info_records_metric(self):
        fs = self._make_fs()
        with patch.object(
            DFSFileSystemWithMetrics.__bases__[0],
            "info",
            return_value={"name": "/f", "size": 100, "type": "file"},
        ):
            fs.info("/f")
        assert fs.metrics.info.count == 1

    def test_exists_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "exists", return_value=True):
            fs.exists("/f")
        assert fs.metrics.exists.count == 1

    def test_walk_records_metric(self):
        fs = self._make_fs()
        with patch.object(
            DFSFileSystemWithMetrics.__bases__[0],
            "walk",
            return_value=iter([("/dir", [], ["file.txt"])]),
        ):
            list(fs.walk("/dir"))
        assert fs.metrics.walk.count == 1

    def test_cat_file_records_metric(self):
        fs = self._make_fs()
        with patch.object(DFSFileSystemWithMetrics.__bases__[0], "cat_file", return_value=b"hello"):
            result = fs.cat_file("/file")
        assert result == b"hello"
        assert fs.metrics.cat.count == 1
        assert fs.metrics.cat.bytes_read == 5

    def test_cat_ranges_records_metric(self):
        fs = self._make_fs()
        with patch.object(
            DFSFileSystemWithMetrics.__bases__[0],
            "cat_ranges",
            return_value=[b"aaa", b"bb"],
        ):
            result = fs.cat_ranges(["/a", "/b"], [0, 0], [3, 2])
        assert result == [b"aaa", b"bb"]
        assert fs.metrics.cat_ranges.count == 1
        assert fs.metrics.cat_ranges.bytes_read == 5

    def test_find_records_metric(self):
        fs = self._make_fs()
        with patch.object(
            DFSFileSystemWithMetrics.__bases__[0],
            "find",
            return_value=["/a.txt", "/b.txt"],
        ):
            result = fs.find("/dir")
        assert result == ["/a.txt", "/b.txt"]
        assert fs.metrics.find.count == 1

    @staticmethod
    def _spy_record():
        """Spy on OperationMetrics.record (calling through) to capture is_read.

        Zero-byte ops (mkdir/rm/mv) produce no bytes_read/bytes_written signal,
        so the ONLY way to verify write-vs-read attribution is to inspect the
        is_read kwarg actually passed to record(). Asserting count==1 (as the
        old tests did) is identical to the plain records-metric tests and could
        never catch a read/write misattribution.
        """
        orig = OperationMetrics.record
        calls = []

        def spy(self, *a, **k):
            calls.append(k)
            return orig(self, *a, **k)

        return patch.object(OperationMetrics, "record", spy), calls

    def test_mkdir_is_write_metric(self):
        """mkdir must reach the metric with is_read=False."""
        fs = self._make_fs()
        spy, calls = self._spy_record()
        with spy, patch.object(DFSFileSystemWithMetrics.__bases__[0], "mkdir"):
            fs.mkdir("/dir")
        assert fs.metrics.mkdir.count == 1
        assert calls and calls[-1].get("is_read") is False

    def test_rm_is_write_metric(self):
        """rm must reach the metric with is_read=False."""
        fs = self._make_fs()
        spy, calls = self._spy_record()
        with spy, patch.object(DFSFileSystemWithMetrics.__bases__[0], "rm"):
            fs.rm("/file")
        assert fs.metrics.rm.count == 1
        assert calls and calls[-1].get("is_read") is False

    def test_mv_is_write_metric(self):
        """mv must reach the metric with is_read=False."""
        fs = self._make_fs()
        spy, calls = self._spy_record()
        with spy, patch.object(DFSFileSystemWithMetrics.__bases__[0], "mv"):
            fs.mv("/a", "/b")
        assert fs.metrics.mv.count == 1
        assert calls and calls[-1].get("is_read") is False
