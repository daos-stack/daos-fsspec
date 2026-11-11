# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for rwlock."""

from __future__ import annotations

import threading
import time

import pytest

from daos_fsspec._rwlock import RWLock


def test_readers_run_concurrently():
    """The point of the exercise: shared holders must not serialize.

    Fails against a plain mutex, which is what this replaced.
    """
    lock = RWLock()
    barrier = threading.Barrier(4, timeout=5)
    overlapped = []

    def reader():
        with lock.read():
            # If readers were serialized this barrier can never be reached by
            # all four threads, and every one of them times out.
            try:
                barrier.wait()
                overlapped.append(True)
            except threading.BrokenBarrierError:
                overlapped.append(False)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert overlapped == [True] * 4


def test_writer_excludes_readers():
    lock = RWLock()
    events = []
    reader_entered = threading.Event()

    def reader():
        with lock.read():
            reader_entered.set()
            time.sleep(0.2)
            events.append("reader_exit")

    t = threading.Thread(target=reader)
    t.start()
    assert reader_entered.wait(timeout=5)

    # Writer must wait for the in-flight reader to drain.
    with lock.write():
        events.append("writer_enter")
    t.join(timeout=5)

    assert events == ["reader_exit", "writer_enter"]


def test_reader_blocks_until_writer_releases():
    lock = RWLock()
    order = []
    writer_holding = threading.Event()
    release_writer = threading.Event()

    def writer():
        with lock.write():
            order.append("writer_enter")
            writer_holding.set()
            release_writer.wait(timeout=5)
            order.append("writer_exit")

    t = threading.Thread(target=writer)
    t.start()
    assert writer_holding.wait(timeout=5)

    # Confirm the reader cannot get in while the writer holds it.
    assert lock.acquire_read(blocking=False) is False

    release_writer.set()
    t.join(timeout=5)

    with lock.read():
        order.append("reader_enter")

    assert order == ["writer_enter", "writer_exit", "reader_enter"]


def test_read_is_reentrant():
    """Helpers like _release_parent are reached from both locked and unlocked
    contexts, so nesting must not self-deadlock."""
    lock = RWLock()
    with lock.read():
        with lock.read():
            with lock.read():
                assert lock.readers == 1  # nesting must not inflate the count
    assert lock.readers == 0


def test_write_is_reentrant():
    lock = RWLock()
    with lock.write():
        with lock.write():
            pass
    # Fully released: a reader can now get in.
    assert lock.acquire_read(blocking=False) is True
    lock.release_read()


def test_read_inside_write_is_allowed():
    """A thread holding exclusive access already has stronger rights."""
    lock = RWLock()
    with lock.write():
        with lock.read():
            pass
    assert lock.readers == 0


def test_upgrade_raises_instead_of_deadlocking():
    """A read->write upgrade would wait on a reader count that includes the
    caller. Surface it loudly rather than hanging."""
    lock = RWLock()
    with lock.read():
        with pytest.raises(RuntimeError, match="lock upgrade would deadlock"):
            lock.acquire_write()


def test_nonblocking_read_acquire_when_free():
    lock = RWLock()
    assert lock.acquire_read(blocking=False) is True
    lock.release_read()
    assert lock.readers == 0


def test_write_times_out_rather_than_hanging():
    lock = RWLock()
    held = threading.Event()
    finish = threading.Event()

    def reader():
        with lock.read():
            held.set()
            finish.wait(timeout=5)

    t = threading.Thread(target=reader)
    t.start()
    assert held.wait(timeout=5)

    start = time.monotonic()
    assert lock.acquire_write(timeout=0.2) is False
    assert time.monotonic() - start < 2.0

    finish.set()
    t.join(timeout=5)


def test_release_without_acquire_raises():
    lock = RWLock()
    with pytest.raises(RuntimeError, match="without holding the shared lock"):
        lock.release_read()
    with pytest.raises(RuntimeError, match="without holding the exclusive lock"):
        lock.release_write()


def test_reader_count_tracks_distinct_threads():
    lock = RWLock()
    entered = threading.Barrier(3, timeout=5)
    release = threading.Event()
    counts = []

    def reader():
        with lock.read():
            entered.wait()
            counts.append(lock.readers)
            release.wait(timeout=5)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert counts == [3, 3, 3]
    assert lock.readers == 0


def test_reader_defers_to_a_waiting_writer():
    """Writer preference: this is what lets the reader count reach zero."""
    lock = RWLock()
    holding = threading.Event()
    release = threading.Event()

    def long_reader():
        with lock.read():
            holding.set()
            release.wait(timeout=5)

    t = threading.Thread(target=long_reader)
    t.start()
    assert holding.wait(timeout=5)

    writer_waiting = threading.Event()

    def writer():
        writer_waiting.set()
        with lock.write():
            pass

    w = threading.Thread(target=writer)
    w.start()
    assert writer_waiting.wait(timeout=5)
    time.sleep(0.1)  # let the writer park in acquire_write

    # A fresh reader must yield to the parked writer.
    assert lock.acquire_read(blocking=False) is False

    release.set()
    t.join(timeout=5)
    w.join(timeout=5)


def test_reader_deferral_is_bounded():
    """Deferral is only a fairness hint. If a reader waited on a parked writer
    forever, a caller holding the shared lock while waiting on worker threads
    that need it would deadlock. After the window, readers proceed."""
    lock = RWLock()
    lock.reader_defer_timeout = 0.3

    holding = threading.Event()
    release = threading.Event()

    def long_reader():
        with lock.read():
            holding.set()
            release.wait(timeout=5)

    t = threading.Thread(target=long_reader)
    t.start()
    assert holding.wait(timeout=5)

    def writer():
        with lock.write():
            pass

    w = threading.Thread(target=writer)
    w.start()
    time.sleep(0.1)

    # Blocked at first...
    assert lock.acquire_read(blocking=False) is False
    # ...but a blocking reader gets through once the window closes, even
    # though the writer is still parked.
    start = time.monotonic()
    assert lock.acquire_read(timeout=5) is True
    elapsed = time.monotonic() - start
    lock.release_read()
    assert 0.1 < elapsed < 3.0

    release.set()
    t.join(timeout=5)
    w.join(timeout=5)


def test_readers_never_bypass_an_active_writer():
    """The deferral window must not weaken the exclusion invariant that
    teardown relies on: only an *active* writer is absolute."""
    lock = RWLock()
    lock.reader_defer_timeout = 0.1

    holding = threading.Event()
    release = threading.Event()

    def writer():
        with lock.write():
            holding.set()
            release.wait(timeout=5)

    w = threading.Thread(target=writer)
    w.start()
    assert holding.wait(timeout=5)

    # Well past the deferral window, a reader still must not get in.
    time.sleep(0.4)
    assert lock.acquire_read(blocking=False) is False
    assert lock.acquire_read(timeout=0.3) is False

    release.set()
    w.join(timeout=5)


def test_writer_eventually_acquires_under_read_load():
    """Reader-preference risks writer starvation in theory. Each reader holds
    for one bounded operation, so a writer must still get through."""
    lock = RWLock()
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            with lock.read():
                time.sleep(0.001)

    threads = [threading.Thread(target=churn) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        assert lock.acquire_write(timeout=10) is True
        lock.release_write()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
