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

import functools
import threading
import time
import types
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


class _Op:
    """How one filesystem method maps onto a metric bucket."""

    __slots__ = ("bytes_of", "is_read", "metric")

    def __init__(
        self,
        metric: str,
        is_read: bool,
        bytes_of: Callable[[Any, tuple, dict], int] | None = None,
    ):
        self.metric = metric
        self.is_read = is_read
        # Byte accounting differs per method: pipe_file knows its count from an
        # argument, cat_file from its return value, cat_ranges from summing a
        # list. A callable is the only thing general enough, and it keeps the
        # knowledge next to the method it belongs to.
        self.bytes_of = bytes_of


def _bytes_written(result: Any, args: tuple, kwargs: dict) -> int:
    """len() of pipe_file's payload, whether passed positionally or by name."""
    if "value" in kwargs:
        return len(kwargs["value"])
    return len(args[1]) if len(args) > 1 else 0


def _bytes_returned(result: Any, args: tuple, kwargs: dict) -> int:
    return len(result) if isinstance(result, (bytes, bytearray)) else 0


def _bytes_in_list(result: Any, args: tuple, kwargs: dict) -> int:
    if not result:
        return 0
    return sum(len(b) for b in result if isinstance(b, (bytes, bytearray)))


#: The instrumented surface. ``tests/test_metrics.py`` asserts this covers
#: every method it claims to, and that each name actually exists on
#: DFSFileSystem; so a rename upstream fails loudly instead of silently
#: dropping a metric.
INSTRUMENTED: dict[str, _Op] = {
    "ls": _Op("ls", is_read=True),
    "info": _Op("info", is_read=True),
    "exists": _Op("exists", is_read=True),
    "walk": _Op("walk", is_read=True),
    "find": _Op("find", is_read=True),
    "mkdir": _Op("mkdir", is_read=False),
    "rm": _Op("rm", is_read=False),
    "mv": _Op("mv", is_read=False),
    "pipe_file": _Op("write", is_read=False, bytes_of=_bytes_written),
    "cat_file": _Op("cat", is_read=True, bytes_of=_bytes_returned),
    "cat_ranges": _Op("cat_ranges", is_read=True, bytes_of=_bytes_in_list),
}

# cat() is deliberately absent. AbstractFileSystem.cat() funnels every read
# through cat_file(), so instrumenting both double-counts; which is the bug
# fixed in 1.2.1. tests/test_metrics.py pins that cat() stays uninstrumented.


def _timed_generator(gen, metric, start, op):
    """Consume a generator, recording the metric when it finishes.

    walk() returns a generator (``_walk_impl`` yields), so timing the call
    itself measured only generator *creation*; a near-zero duration no matter
    how large the tree. The count was right, which is why the existing test
    passed and nobody noticed the durations were meaningless.

    The record happens in ``finally`` so a caller who abandons the walk part-way
    still gets the time actually spent, rather than nothing at all.
    """
    error = False
    try:
        yield from gen
    except Exception:
        error = True
        raise
    finally:
        metric.record(duration=time.monotonic() - start, error=error, is_read=op.is_read)


def _make_instrumented(name: str, op: _Op) -> Callable[..., Any]:
    """Build the wrapper for one method, delegating to the superclass version."""

    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        # Resolved per call via super(), not captured when this closure is
        # built. Binding the parent at import time looks like a harmless
        # micro-optimisation and is not: it makes the wrapper immune to
        # patching, so the method can never be stubbed in a test or swapped at
        # runtime. It also walks the real MRO, which matters now that listing
        # methods live on a mixin rather than on DFSFileSystem itself.
        parent = getattr(super(DFSFileSystemWithMetrics, self), name)
        metric = getattr(self.metrics, op.metric, None) or self.metrics.other
        start = time.monotonic()
        try:
            result = parent(*args, **kwargs)
        except Exception:
            metric.record(duration=time.monotonic() - start, error=True, is_read=op.is_read)
            raise

        if isinstance(result, types.GeneratorType):
            return _timed_generator(result, metric, start, op)

        metric.record(
            duration=time.monotonic() - start,
            error=False,
            bytes_count=op.bytes_of(result, args, kwargs) if op.bytes_of else 0,
            is_read=op.is_read,
        )
        return result

    # Copy the wrapped method's metadata so the generated method keeps the real
    # signature. Without this, every instrumented method introspects as
    # (*args, **kwargs); help(), IDE completion and the API snapshot all
    # degrade, which the snapshot test caught as eleven CHANGED entries.
    #
    # Note the deliberate asymmetry: the parent is resolved here, once, purely
    # for *metadata*, while the call target above is resolved per call via
    # super(). Binding the metadata early is safe; binding the dispatch early
    # is what made the method unpatchable.
    functools.update_wrapper(wrapper, getattr(DFSFileSystem, name))
    wrapper.__qualname__ = f"DFSFileSystemWithMetrics.{name}"
    wrapper.__doc__ = (
        f"Instrumented {name}(); records to metrics.{op.metric}.\n\n"
        f"Generated from daos_fsspec.metrics.INSTRUMENTED. Behaviour is "
        f"identical to DFSFileSystem.{name}().\n\n"
        f"{getattr(DFSFileSystem, name).__doc__ or ''}"
    )
    return wrapper


class DFSFileSystemWithMetrics(DFSFileSystem):
    """DFSFileSystem that records timing, byte counts and error rates.

    Access the figures via the ``.metrics`` attribute::

        fs = DFSFileSystemWithMetrics(pool="mypool", container="mycont")
        fs.cat("/data/file.bin")
        print(fs.metrics.summary())

    The instrumented methods are generated from :data:`INSTRUMENTED` at import
    time rather than written out by hand, so the set is auditable in one place
    and cannot drift silently from the filesystem it wraps.

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
        """Record one externally-timed operation.

        Retained because it is part of the public surface and is useful for
        instrumenting a call this class does not wrap. The generated wrappers
        above no longer go through it.
        """
        metric = getattr(self.metrics, op, self.metrics.other)
        start = time.monotonic()
        error = False
        try:
            return fn(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            metric.record(duration=time.monotonic() - start, error=error, is_read=is_read)


for _name, _op in INSTRUMENTED.items():
    setattr(DFSFileSystemWithMetrics, _name, _make_instrumented(_name, _op))
del _name, _op
