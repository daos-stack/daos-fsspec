# Contributing to daos-fsspec

Thanks for your interest in daos-fsspec! This is a community-maintained fsspec backend for DAOS DFS (Distributed File System).

## Development Setup

### Prerequisites

- Python 3.9+
- DAOS 2.6+ client libraries (`libdaos.so`, `libdfs.so`, `libgurt.so`) on `LD_LIBRARY_PATH`
- A running DAOS cluster for live tests

### Install for development

```bash
git clone https://github.com/daos-stack/daos-fsspec.git
cd daos-fsspec
pip install -e ".[dev]"
```

### Running Tests

#### Quick start (automated)

```bash
# Unit tests only (no DAOS needed)
docker/test.sh

# Full test suite (builds containers, starts DAOS, runs all tests, tears down)
docker/test.sh --live

# Keep containers running after tests (for development)
docker/test.sh --live --keep
```

#### Unit tests (no DAOS required)

```bash
pytest tests/ -v -m "not live"
```

#### Live tests (requires DAOS cluster)

```bash
# Create a test container
daos container create <pool> fsspec-test --type=POSIX

# Run tests
DAOS_FSSPEC_TEST_POOL=<pool> DAOS_FSSPEC_TEST_CONTAINER=fsspec-test pytest tests/ -v

# Clean up
daos container destroy <pool> fsspec-test
```

#### Running tests in Docker (with DAOS client)

If you have a DAOS agent container running:

```bash
docker run --rm --network host --privileged --entrypoint="" \
  -v /var/run/daos_agent:/var/run/daos_agent \
  -v /etc/daos:/etc/daos:ro \
  -v $(pwd):/daos-fsspec \
  -e DAOS_FSSPEC_TEST_POOL=<pool> \
  -e DAOS_FSSPEC_TEST_CONTAINER=fsspec-test \
  <daos-client-image> \
  bash -c "pip3 install --quiet fsspec pytest && pip3 install --quiet --no-deps -e /daos-fsspec && cd /daos-fsspec && pytest tests/ -v"
```

## Architecture

```
daos_fsspec/
├── __init__.py        # Package exports + register_implementation
├── _bindings.py       # ctypes FFI to libdaos + libdfs (struct defs, function wrappers)
├── _bindings.pyi      # Type stubs for IDE/mypy support
├── core.py            # DFSFileSystem(AbstractFileSystem): most fsspec methods
├── _listing.py        # DFSListingMixin: ls/walk/find/du + the listings cache
├── _rwlock.py         # RWLock: shared/exclusive mount lock (see below)
├── _runtime.py        # DAOS/DFS initialization ownership and ordered exit cleanup
├── _fork.py           # fork detection: poisons inherited handles in the child
├── _fs_cache.py       # get_fs()/cache_info()/clear_fs_cache(): per-process cache
├── dfs_file.py        # DFSFile(AbstractBufferedFile): buffered file handle
├── async_core.py      # DFSAsyncFileSystem(AsyncFileSystem): asyncio wrapper
├── pool.py            # DFSFileSystemPool: connection pool for concurrent access
├── metrics.py         # DFSFileSystemWithMetrics: operation timing/instrumentation
└── py.typed           # PEP 561 marker
```

### Key Design Decisions

1. **ctypes FFI** (not pydaos): Full control, tested against DAOS 2.6.5 and 2.8.0, no build-time compilation needed. Other versions remain unverified.

2. **Thread safety**: one **shared/exclusive** lock per DFSFileSystem instance (`_rwlock.py`), not a single mutex. DFS does its own locking for mount state, so concurrent I/O on one mount needs no serialisation; the invariant we enforce is that teardown and reconnect wait for in-flight libdfs *calls* to drain. Note "calls", not "operations": chunked reads and writes release the lock between chunks, so they additionally record `_mount_generation` and raise `MountChangedError` if a reconnect replaced the mount underneath them. The writer-preferring design matters: without it, continuous read load starved `close()` indefinitely.

3. **Fork safety**: DAOS handles do not survive `os.fork()`, and using an inherited one **hangs** rather than failing. `_fork.py` registers an `os.register_at_fork` hook that poisons every inherited handle in the child, so the next call raises a clear error instead of blocking forever. Use `multiprocessing` with the **spawn** start method; `daos_fsspec.get_fs(pool, container)` gives each process its own cached filesystem and is what the fork error points users at.

4. **Chunked I/O**: Large reads/writes are split into 16MB chunks to avoid massive single allocations.

5. **`cachable = False`**: We manage our own lifecycle instead of relying on fsspec's `_Cached` metaclass.

6. **readdirplus**: `ls(detail=True)` uses `dfs_readdirplus()` to get stat info in the same call as directory listing (no N+1 lookup pattern).

## Code Style

- Type annotations on all public methods
- Docstrings on public classes and non-obvious methods
- `_check(rc, msg)` pattern for all C library calls: maps DAOS errors to Python exceptions

## Submitting Changes

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass (both unit and live if you have a cluster)
4. Submit a pull request with a clear description

## License

Apache-2.0. By contributing, you agree that your contributions will be licensed under the same license.
