# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry-style instrumentation for daos-fsspec.

Opt-in metrics collection for monitoring DAOS operations in production.

Usage::

    from daos_fsspec.metrics import DFSFileSystemWithMetrics

    fs = DFSFileSystemWithMetrics(pool="mypool", container="mycont")
    data = fs.cat("/path/to/file")

    # Access metrics
    print(fs.metrics.summary())
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from daos_fsspec.core import DFSFileSystem

_T = TypeVar("_T")


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""

    count: int = 0
    errors: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    bytes_read: int = 0
    bytes_written: int = 0

    def __post_init__(self):
        # += on instance attrs is not atomic (4-5 bytecodes), so protect
        # concurrent updates when a metrics object is shared across threads.
        self._lock = threading.Lock()

    def record(
        self, duration: float, error: bool = False, bytes_count: int = 0, is_read: bool = True
    ):
        """Record an operation."""
        with self._lock:
            self.count += 1
            if error:
                self.errors += 1
            self.total_duration += duration
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            if is_read:
                self.bytes_read += bytes_count
            else:
                self.bytes_written += bytes_count

    @property
    def avg_duration(self) -> float:
        """Average duration in seconds."""
        return self.total_duration / self.count if self.count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        return self.errors / self.count if self.count > 0 else 0.0


@dataclass
class DFSMetrics:
    """Aggregated metrics for all DFS operations."""

    ls: OperationMetrics = field(default_factory=OperationMetrics)
    info: OperationMetrics = field(default_factory=OperationMetrics)
    exists: OperationMetrics = field(default_factory=OperationMetrics)
    cat: OperationMetrics = field(default_factory=OperationMetrics)
    write: OperationMetrics = field(default_factory=OperationMetrics)
    mkdir: OperationMetrics = field(default_factory=OperationMetrics)
    rm: OperationMetrics = field(default_factory=OperationMetrics)
    mv: OperationMetrics = field(default_factory=OperationMetrics)
    walk: OperationMetrics = field(default_factory=OperationMetrics)
    cat_ranges: OperationMetrics = field(default_factory=OperationMetrics)
    find: OperationMetrics = field(default_factory=OperationMetrics)
    other: OperationMetrics = field(default_factory=OperationMetrics)
    _start_time: float = field(default_factory=time.monotonic)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all metrics."""
        elapsed = time.monotonic() - self._start_time
        ops = {
            "ls": self.ls,
            "info": self.info,
            "exists": self.exists,
            "cat": self.cat,
            "write": self.write,
            "mkdir": self.mkdir,
            "rm": self.rm,
            "mv": self.mv,
            "walk": self.walk,
            "cat_ranges": self.cat_ranges,
            "find": self.find,
            "other": self.other,
        }
        total_ops = sum(op.count for op in ops.values())
        total_errors = sum(op.errors for op in ops.values())
        total_bytes_read = sum(op.bytes_read for op in ops.values())
        total_bytes_written = sum(op.bytes_written for op in ops.values())

        return {
            "uptime_seconds": elapsed,
            "total_operations": total_ops,
            "total_errors": total_errors,
            "error_rate": total_errors / total_ops if total_ops > 0 else 0.0,
            "operations_per_second": total_ops / elapsed if elapsed > 0 else 0.0,
            "total_bytes_read": total_bytes_read,
            "total_bytes_written": total_bytes_written,
            "details": {
                name: {
                    "count": op.count,
                    "errors": op.errors,
                    "avg_duration_ms": op.avg_duration * 1000,
                    "min_duration_ms": op.min_duration * 1000
                    if op.min_duration != float("inf")
                    else 0,
                    "max_duration_ms": op.max_duration * 1000,
                }
                for name, op in ops.items()
                if op.count > 0
            },
        }

    def reset(self):
        """Reset all metrics."""
        for attr in [
            "ls",
            "info",
            "exists",
            "cat",
            "write",
            "mkdir",
            "rm",
            "mv",
            "walk",
            "cat_ranges",
            "find",
            "other",
        ]:
            setattr(self, attr, OperationMetrics())
        self._start_time = time.monotonic()


class DFSFileSystemWithMetrics(DFSFileSystem):
    """DFSFileSystem wrapper that records operation metrics.

    Subclasses DFSFileSystem, overriding key methods to record timing
    and byte counts. Access metrics via the ``.metrics`` attribute.

    Parameters
    ----------
    pool : str
        DAOS pool label.
    container : str
        DAOS container label.
    **kwargs
        Forwarded to DFSFileSystem.
    """

    def __init__(self, pool: str, container: str, **kwargs: Any):
        super().__init__(pool=pool, container=container, **kwargs)
        self.metrics = DFSMetrics()

    def _timed_call(
        self, op: str, fn: Callable[..., _T], *args: Any, is_read: bool = True, **kwargs: Any
    ) -> _T:
        """Execute fn with timing, recording to the named metric."""
        metric = getattr(self.metrics, op, self.metrics.other)
        start = time.monotonic()
        error = False
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception:
            error = True
            raise
        finally:
            duration = time.monotonic() - start
            metric.record(duration=duration, error=error, is_read=is_read)

    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        return self._timed_call("ls", super().ls, path, detail=detail, is_read=True, **kwargs)

    def info(self, path: str, **kwargs: Any) -> dict:
        return self._timed_call("info", super().info, path, is_read=True, **kwargs)

    def exists(self, path: str, **kwargs: Any) -> bool:
        return self._timed_call("exists", super().exists, path, is_read=True, **kwargs)

    # NOTE: cat() is intentionally NOT overridden. AbstractFileSystem.cat()
    # funnels every read through cat_file() (verified against fsspec), so the
    # cat_file override below is the single instrumentation point. Overriding
    # cat() as well would double-count every cat into self.metrics.cat.

    def pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        metric = self.metrics.write
        start = time.monotonic()
        error = False
        try:
            super().pipe_file(path, value, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            duration = time.monotonic() - start
            metric.record(duration=duration, error=error, bytes_count=len(value), is_read=False)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        self._timed_call(
            "mkdir", super().mkdir, path, create_parents=create_parents, is_read=False, **kwargs
        )

    def rm(
        self, path: str, recursive: bool = False, maxdepth: int | None = None, **kwargs: Any
    ) -> None:
        self._timed_call(
            "rm", super().rm, path, recursive=recursive, maxdepth=maxdepth, is_read=False, **kwargs
        )

    def mv(self, path1: str, path2: str, **kwargs: Any) -> None:
        self._timed_call("mv", super().mv, path1, path2, is_read=False, **kwargs)

    def walk(self, top: str, maxdepth: int | None = None, topdown: bool = True, **kwargs: Any):
        return self._timed_call(
            "walk", super().walk, top, maxdepth=maxdepth, topdown=topdown, is_read=True, **kwargs
        )

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any):
        metric = self.metrics.cat
        start_time = time.monotonic()
        error = False
        result = None
        try:
            result = super().cat_file(path, start=start, end=end, **kwargs)
            return result
        except Exception:
            error = True
            raise
        finally:
            duration = time.monotonic() - start_time
            bytes_count = len(result) if isinstance(result, bytes) else 0
            metric.record(duration=duration, error=error, bytes_count=bytes_count, is_read=True)

    def cat_ranges(
        self,
        paths: list[str],
        starts: list[int] | int,
        ends: list[int] | int,
        max_gap: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ):
        metric = self.metrics.cat_ranges
        start_time = time.monotonic()
        error = False
        result = None
        try:
            result = super().cat_ranges(
                paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs
            )
            return result
        except Exception:
            error = True
            raise
        finally:
            duration = time.monotonic() - start_time
            bytes_count = sum(len(b) for b in result if isinstance(b, bytes)) if result else 0
            metric.record(duration=duration, error=error, bytes_count=bytes_count, is_read=True)

    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ):
        return self._timed_call(
            "find",
            super().find,
            path,
            maxdepth=maxdepth,
            withdirs=withdirs,
            detail=detail,
            is_read=True,
            **kwargs,
        )
