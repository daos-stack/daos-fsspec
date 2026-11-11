# daos-fsspec

[![PyPI version](https://img.shields.io/pypi/v/daos-fsspec)](https://pypi.org/project/daos-fsspec/)
[![Tests](https://github.com/daos-stack/daos-fsspec/actions/workflows/test.yml/badge.svg)](https://github.com/daos-stack/daos-fsspec/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/daos-stack/daos-fsspec)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/daos-fsspec)](https://pypi.org/project/daos-fsspec/)

**fsspec filesystem interface for [DAOS](https://docs.daos.io/) DFS (Distributed File System)**

Use pandas, Dask, Hugging Face Datasets, PyArrow, and other fsspec clients with
DAOS POSIX containers. The package calls the native DFS library directly.

```python
import pandas as pd

# Read parquet directly from a DAOS DFS container
df = pd.read_parquet("dfs://mypool/mycont/train/shard-00000.parquet")

# Or use the filesystem API
import fsspec
fs = fsspec.filesystem("dfs", pool="mypool", container="mycont")
files = fs.ls("/train/")
data = fs.cat("/train/shard-00000.parquet")
```

## How it works

`daos-fsspec` implements the fsspec filesystem interface using ctypes bindings
to `libdfs.so`. It supports file I/O, directory operations, streaming, connection
pooling, and async applications. See [tested versions](#tested-versions) for
the supported DAOS client libraries.

## Installation

```bash
pip install daos-fsspec
```

**Requirements:**
- Python 3.9+
- DAOS client libraries (`libdaos.so`, `libdfs.so`, `libgurt.so`) on `LD_LIBRARY_PATH` or in `/usr/lib64`: tested against 2.6.5 and 2.8.0, see [Tested versions](#tested-versions)
- A running DAOS agent (for pool/container access)

### Tested versions

| DAOS | Status | Evidence |
|---|---|---|
| **2.8.0** | Supported | Full live suite passes; ABI verified against its own headers |
| **2.6.5** | Supported | Full live suite passes; ABI verified against its own headers |
| Other 2.6+ | Unverified | Expected to work; nothing is run against it |

Both supported versions are exercised by a CI matrix that brings up a real
RAM-backed DAOS cluster and runs the full suite. Results and reproduction steps are
in the [1.4.1 validation record](docs/validation.md), with cluster setup notes in [`docker/LIVE_TEST_RESULTS.md`](docker/LIVE_TEST_RESULTS.md). Because a ctypes
layout mismatch returns plausible garbage instead of crashing, the struct
layouts are additionally checked field-by-field against each version's own
`daos-devel` headers with a compiled `offsetof()` probe
([`docker/abi_probe.py`](docker/abi_probe.py)): 12 structs and 59 field offsets
match on both, with no drift between them.

Other DAOS releases are unverified.

## Quick verification

After installing, verify the connection works:

```python
import fsspec

fs = fsspec.filesystem("dfs", pool="mypool", container="mycont")
print(fs.ls("/"))
```

## Usage

### Direct filesystem API

```python
import fsspec

fs = fsspec.filesystem("dfs", pool="mypool", container="datasets")

# List files
fs.ls("/train/")

# Read a file
data = fs.cat("/train/shard-00000.parquet")

# Write a file
with fs.open("/output/results.csv", "wb") as f:
    f.write(csv_data)

# Directory operations
fs.mkdir("/experiments/run-42", create_parents=True)
fs.mv("/output/temp.csv", "/output/final.csv")
fs.rm("/output/temp/", recursive=True)
```

**Optional parameters:**

```python
fs = fsspec.filesystem(
    "dfs",
    pool="mypool",
    container="datasets",
    readonly=True,          # Mount read-only (default: False)
    object_class=0,         # DAOS object class ID for new files (0 = container default)
    chunk_size=0,           # DFS stripe/chunk size for new files (0 = DAOS default)
    retry_count=3,          # Retries for transient errors (default: 3)
    retry_backoff=0.1,      # Initial backoff in seconds, doubles each retry (default: 0.1)
)
```

### URI shorthand

Once installed, the `dfs://` scheme is registered globally via fsspec entry points:

```python
import pandas as pd

# Read
df = pd.read_parquet("dfs://pool/container/data.parquet")

# Write
df.to_parquet("dfs://pool/container/output.parquet")
```

Full URIs passed to an existing filesystem must use the same pool and container
identifiers used to construct it. Different identifiers are rejected, including
UUID/label aliases; use relative paths when accessing the mounted namespace.

### HuggingFace Datasets

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="dfs://pool/container/train/*.parquet")
```

### Dask

```python
import dask.dataframe as dd
ddf = dd.read_parquet("dfs://pool/container/partitioned/")
```

### Connection pool

For concurrent access from multiple threads, use a connection pool. Each connection has its own DFS mount and lock, so operations on different connections don't contend:

```python
from daos_fsspec.pool import DFSFileSystemPool

with DFSFileSystemPool(pool="mypool", container="mycont", size=4) as pool:
    with pool.connection() as fs:
        data = fs.cat("/path/to/file")

    # Or acquire/release manually with a timeout
    fs = pool.acquire(timeout=10.0)
    try:
        fs.ls("/")
    finally:
        pool.release(fs)
```

### Async

For asyncio-based applications, all operations are dispatched to Python worker threads via `asyncio.to_thread()`. Concurrency is bounded by `size` (default 4): the number of pooled DFS handles. Operations beyond that queue; they do not fail and do not block the event loop:

```python
import asyncio
from daos_fsspec.async_core import DFSAsyncFileSystem

async def main():
    # size=8 gives eight DFS handles, so up to eight operations run at once.
    fs = DFSAsyncFileSystem(pool="mypool", container="mycont", size=8)
    data = await fs._cat_file("/path/to/file")
    await fs._pipe_file("/output/data.bin", data)
    entries = await fs._ls("/")

    # Gathering more than `size` operations is fine: the surplus queues.
    shards = await asyncio.gather(*[
        fs._cat_file(f"/data/shard-{i:05d}.parquet") for i in range(64)
    ])
    await fs.close()

asyncio.run(main())
```

### Metrics

Wrap any `DFSFileSystem` to record timing, byte counts, and error rates:

```python
from daos_fsspec.metrics import DFSFileSystemWithMetrics

fs = DFSFileSystemWithMetrics(pool="mypool", container="mycont")
fs.cat("/data/file.bin")
fs.pipe_file("/output/result.bin", b"data")

print(fs.metrics.summary())
# {'uptime_seconds': 1.23, 'total_operations': 2, 'total_errors': 0,
#  'total_bytes_read': 1024, 'total_bytes_written': 4, ...}
```

## Performance

`daos-fsspec` is designed for throughput, not just correctness:

- **16 MB chunked I/O**: reads and writes are automatically split into 16 MB chunks. Single-chunk paths (most files) have zero overhead.
- **`readdirplus` batching**: `ls()` and `walk()` use `dfs_readdirplus` to stat all entries in one call per batch (128 entries). No N+1 query pattern.
- **`cat_file` fast path**: uses a single `dfs_lookup_path` call instead of the `open` → `DFSFile` → `_resolve_path` chain (3 C calls vs 7 for depth-3 paths).
- **`cat_ranges` parallelism**: reads multiple byte ranges through a `ThreadPoolExecutor` (up to 8 workers). These genuinely overlap: the mount lock is shared/exclusive, so concurrent readers do not serialise, and ctypes releases the GIL around each libdfs call. `DFSFileSystemPool` is still the better fit when you want independent handles with independent locks.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Python application (pandas, HF, Dask, PyArrow, ...)     │
├──────────────────────────────────────────────────────────┤
│  fsspec AbstractFileSystem / AsyncFileSystem API          │
├──────────────────────────────────────────────────────────┤
│  daos-fsspec                                             │
│  ┌─────────────┬──────────────┬────────────────────────┐ │
│  │ core.py     │ async_core.py│ pool.py / metrics.py   │ │
│  │ DFSFile-    │ DFSAsync-    │ DFSFileSystemPool      │ │
│  │ System      │ FileSystem   │ DFSFileSystemWith-     │ │
│  │             │ (to_thread)  │ Metrics                │ │
│  └──────┬──────┴──────────────┴────────────────────────┘ │
│  ┌──────┴──────────────────────────────────────────────┐ │
│  │ _bindings.py: ctypes FFI to libdfs.so + libdaos.so │ │
│  └─────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  DAOS client library (DAOS agent → engines)               │
└──────────────────────────────────────────────────────────┘
```

### Directory listings

Listings caching is disabled by default. Enable it with `use_listings_cache=True`
and a bounded `listings_expiry_time` (default 1 second). Local mutations invalidate
cached entries, but peer processes cannot invalidate them. Call
`fs.invalidate_cache()` after coordinating with other writers when fresh listings
are required. Only `ls()` consults this cache; recursive traversal remains uncached.

### Multiprocessing: use `get_fs()`

`DFSFileSystem` **cannot be pickled**, and raises if you try. DAOS handles do not
survive a process boundary, and reconnecting on unpickle would make
`self.fs = fs` on a `Dataset` look like it worked while silently opening one
pool+container handle per worker: 24 ranks × 4 workers is 96 connections. The
refusal keeps that visible.

`daos_fsspec.get_fs(pool, container)` is the supported alternative: a
per-process cached filesystem, keyed so that a forked child never receives the
parent's handle. Keep only the strings on your object and call it where you
need I/O:

```python
import daos_fsspec, torch

class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, pool, container, files):
        # strings only: no handle on self, so this pickles for spawn
        self.pool, self.container, self.files = pool, container, files

    def __getitem__(self, i):
        fs = daos_fsspec.get_fs(self.pool, self.container)   # one per process
        return fs.cat_file(self.files[i])
```

Repeated calls with the same arguments return the same instance.
`daos_fsspec.cache_info()` reports what is cached, if you ever need to answer
"why do I have N connections"; `clear_fs_cache()` drops them.

**`fork()` is not supported**: use `spawn`. A DAOS client keeps background
threads that `fork()` does not duplicate, so the child inherits their mutexes
locked with no owner. The child cannot be repaired, so every entry point raises
immediately and names `spawn` rather than hanging, which is what it used to do.

### Thread safety

Each filesystem uses a shared/exclusive lock. Reads, writes, and metadata calls
hold shared access, allowing concurrent native calls on one mount. Closing or
reconnecting takes exclusive access and waits for in-flight calls to finish.

Individual file objects are not thread-safe. Use `DFSFileSystemPool` when workers
need independent connections with separate DFS mounts and locks.

### Process exit and resource cleanup

Use file/filesystem context managers, finish outstanding worker and async tasks,
and close writers before closing their filesystem. For cached connections,
`clear_fs_cache(close=True)` closes them after their users have finished.

The package owns one DAOS initialization reference and, when needed, one DFS
reference. A process-exit callback closes remaining known filesystems, then
releases those references in DFS-before-DAOS order. It skips inherited state in
forked children. Exit cleanup discards unclosed write buffers with a warning;
it never implicitly commits them. Cleanup failures are reported and prevent
unsafe finalization or destructive retries.

This is cleanup after application work has stopped, including any daemon workers.
It is not a public concurrent-shutdown API. The private `_bindings.daos_fini()`
requires callers to close resources first and is terminal; standalone binding
users remain responsible for their own handles. Automatic-exit failures are
logged and do not reliably change the process exit status. Abrupt termination
such as `SIGKILL` or `os._exit()` does not run the callback.

The 1.4.1 cleanup fix has passed local native exit tests on both supported DAOS
versions. The original multi-node CXI exit-crash workload still needs a rerun;
concurrent performance acceptance also remains open. See the
[validation record](docs/validation.md). The reported 128 MiB read
stragglers remain unresolved; the local measurements do not establish a fix.

### Fork safety

DAOS handles and the process-global DAOS runtime do not survive `fork()`.
Forked children therefore cannot reuse an inherited filesystem *or* construct a
new one; both fail immediately instead of risking a hang. Use the `spawn` start
method, then create or obtain the worker's filesystem with `get_fs()` inside the
spawned process.

### Async support

`daos-fsspec` provides `DFSAsyncFileSystem` for asyncio compatibility. This is a **thin wrapper** that runs all DAOS operations in a thread pool via `asyncio.to_thread()`. It is not a true non-blocking implementation, and it cannot be: of the DAOS C API, only `dfs_read` and `dfs_write` accept a `daos_event_t`. Every metadata call (lookup, open, stat, readdirplus, mkdir, remove, move) is unconditionally blocking, so a thread pool is the only way to get concurrency across them.

Each pooled connection has an independent DFS handle and lock. `size=8` allows up to eight operations at once. Additional operations wait in worker threads, leaving the event loop free.

Pick `size` to match your intended concurrency rather than leaving the default of 4. For a dedicated synchronous thread pool instead, use `DFSFileSystemPool` directly:

```python
from daos_fsspec.pool import DFSFileSystemPool

pool = DFSFileSystemPool(pool="mypool", container="mycont", size=8)
with pool.connection() as fs:
    data = fs.cat("/path/to/file")
```

### Why ctypes to libdfs?

| Approach | Pros | Cons |
|----------|------|------|
| **ctypes to libdfs** (this project) | Fast, full control, no pydaos dependency | More upfront FFI work; ABI must be re-verified per DAOS version |
| pydaos | Official bindings | DFS API incomplete, hard to install |
| subprocess `daos fs` | Zero deps | Slow (fork per op), parsing stdout, no streaming |

## Known Limitations

- **Append is not atomic**: DAOS DFS has no `O_APPEND`, so `"ab"` records the file size at open time and writes from there. Correct for a single writer; a peer that extends the file between your `open()` and your `write()` will be overwritten. Use explicit offsets if several writers share a file.
- **No concurrent writes to same file**: Multiple writers to the same file will produce undefined behavior. Coordinate writes externally.
- **No hard links**: DAOS DFS does not support hard links, only symbolic links.
- **No mmap**: Memory-mapped files are not supported.
- **No file locking**: `flock()` and `fcntl()` locks are not supported.
- **No depth-limited remove**: DAOS DFS does not provide native depth-limited removal, so `rm(maxdepth=...)` raises `NotImplementedError`.
- **glibc/Linux struct layout**: the pure-ctypes bindings validate known Linux `struct stat` layouts at import time. Alpine/musl and unknown architectures may need additional validation before production use.

## API Reference

### `DFSFileSystem`

Main filesystem class. Instantiate via `fsspec.filesystem("dfs", ...)` or directly.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool` | `str` | Required | DAOS pool label (required) |
| `container` | `str` | Required | DAOS container label (required) |
| `readonly` | `bool` | `False` | Mount read-only |
| `object_class` | `int` | `0` | DAOS object class ID for newly-created files. `0` uses the directory/container default. Use `suggest_oclass("file:single")` etc. to query an ID. |
| `chunk_size` | `int` | `0` | DFS stripe/chunk size for newly-created files. `0` uses the DAOS default (commonly 1 MiB). This is distinct from the internal 16 MiB transfer buffer. |
| `retry_count` | `int` | `3` | Retries for transient errors |
| `retry_backoff` | `float` | `0.1` | Initial backoff in seconds (doubles each retry) |

**Filesystem operations:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `ls` | `(path, detail=False) → list` | List directory contents. With `detail=True`, returns list of dicts with `name`, `size`, `type`, `mode`, `mtime`, `atime`, `uid`, `gid`. |
| `info` | `(path) → dict` | Stat a single path. Returns dict with `name`, `size`, `type`, `mode`, `mtime`, `atime`, `uid`, `gid`. |
| `exists` | `(path) → bool` | Check if path exists. Uses `O_NOFOLLOW`, so a dangling symlink is reported as existing: identical to `lexists()`. |
| `lexists` | `(path) → bool` | Like `exists()`, does not follow symlinks. In DFS these are equivalent. |
| `islink` | `(path) → bool` | Return `True` if path is a symbolic link. |
| `mkdir` | `(path, create_parents=True) → None` | Create a directory. With `create_parents=True`, creates intermediate directories. |
| `rmdir` | `(path) → None` | Remove an empty directory. |
| `rm` | `(path, recursive=False, maxdepth=None) → None` | Remove a file or directory. With `recursive=True`, removes trees. Non-`None` `maxdepth` raises `NotImplementedError`. |
| `mv` | `(path1, path2) → None` | Rename/move a file or directory. |
| `touch` | `(path, truncate=True) → None` | Create empty file or update mtime. |
| `open` | `(path, mode="rb", **kwargs) → DFSFile` | Open a file. Supports `"rb"`, `"wb"`, `"ab"`. Append is not atomic: see Limitations. |
| `cat` | `(path, recursive=False, on_error="raise") → bytes` | Read file(s) into memory. Inherited from fsspec. |
| `cat_file` | `(path, start=None, end=None) → bytes` | Read a file (or byte range) with optimized single-lookup path. |
| `cat_ranges` | `(paths, starts, ends, max_gap=None, on_error="return", **kwargs) → list` | Read byte ranges from multiple files in parallel. With the default `on_error="return"`, a failed range comes back **as an exception object inside the list**: check types before use, or pass `on_error="raise"`. |
| `pipe` | `(path, value) → None` | Write bytes to a file. Inherited from fsspec; the keyword is `value`, so `pipe(path, data=b"...")` fails. |
| `pipe_file` | `(path, value) → None` | Write bytes to a file in one shot (no buffering). |
| `head` | `(path, size=1024) → bytes` | Read the first `size` bytes. |
| `tail` | `(path, size=1024) → bytes` | Read the last `size` bytes. |
| `find` | `(path, maxdepth=None, withdirs=False, detail=False) → list \| dict` | Recursively list all files under path. |
| `find_parallel` | `(path, maxdepth=None, withdirs=False, detail=False, num_workers=4) → list \| dict` | **Deprecated alias for `find()`.** `num_workers` is ignored. The anchor-split implementation was removed: it measured 0.99-1.01x of `find()`, never executed its parallel path on any available cluster (DAOS rejects any anchor count other than the object's shard count, which is 1 on a single-target pool), shared one object handle across threads, and discarded worker errors. Use `find()`. |
| `walk` | `(top, maxdepth=None, topdown=True) → generator` | Walk directory tree yielding `(dirpath, dirnames, filenames)`. |
| `du` | `(path, total=False, maxdepth=None) → dict \| int` | Disk usage of files under path. |
| `get_file` | `(rpath, lpath) → None` | Download a file from DFS to a local path (streams in 16 MiB chunks). |
| `put_file` | `(lpath, rpath) → None` | Upload a local file to DFS (streams in 16 MiB chunks). |
| `truncate` | `(path, size=0) → None` | Truncate a file to the given size. Requires `dfs_punch` support. |
| `checksum` | `(path) → str` | Lightweight change indicator (`mode:size:mtime`). Not a content hash. |
| `symlink` | `(path, target) → None` | Create a symbolic link. |
| `readlink` | `(path) → str` | Read the target of a symbolic link. |
| `chmod` | `(path, mode) → None` | Change permissions (e.g., `0o644`). |
| `modified` | `(path) → float` | Return mtime as Unix timestamp. |
| `created` | `(path) → float` | Return ctime as Unix timestamp. |
| `pool_space` | `() → PoolSpace` | Query DAOS pool storage space (SCM + NVMe total/free). |
| `health_check` | `() → bool` | Verify the DAOS connection is alive. |
| `close` | `() → None` | Release all DAOS handles. Idempotent. Also usable as a context manager. |

Transfers use temporary files and replace the destination only after copying
and closing successfully. An upload opens its local source before creating a
remote temporary file. A download raises on EOF before the initially reported
size. This detects truncation but does not provide a snapshot of a concurrently
modified source. Replacement creates a new file; existing destination metadata
and hard-link identity are not preserved. Temporary remote files may remain if
cleanup itself fails or the process terminates unexpectedly.

Automatic reconnect is refused while files are open on that connection. The
operation reports its connection error without discarding another caller's
buffered writes. Explicit filesystem teardown still discards pending buffers;
closing an affected writer subsequently raises an error.


**Extended attributes:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `getxattr` | `(path, attr_name) → bytes` | Get an extended attribute value. |
| `setxattr` | `(path, attr_name, value, flags=0) → None` | Set an extended attribute. Flags: `0`, `XATTR_CREATE`, `XATTR_REPLACE`. |
| `listxattr` | `(path) → list[str]` | List all extended attribute names. |
| `removexattr` | `(path, attr_name) → None` | Remove an extended attribute. |

### `DFSFile`

File handle returned by `fs.open()`. Supports the standard buffered file interface.

| Method | Description |
|--------|-------------|
| `read(size=-1)` | Read up to `size` bytes (or to EOF). |
| `write(data)` | Write bytes at current position. |
| `seek(offset, whence=0)` | Move to position. `whence`: 0=absolute, 1=relative, 2=from end. |
| `tell()` | Return current position. |
| `close()` | Flush and release the DFS handle. Idempotent. |

### `DFSFileSystemPool`

Connection pool for concurrent access. Each connection has its own DFS mount and lock.

```python
from daos_fsspec.pool import DFSFileSystemPool

pool = DFSFileSystemPool(pool="mypool", container="mycont", size=4)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `acquire` | `(timeout=None) → DFSFileSystem` | Get a connection from the pool. Blocks if pool is full. |
| `release` | `(fs) → None` | Return a connection to the pool. |
| `connection` | `() → context manager` | Context manager that acquires and releases automatically. |
| `close` | `(force=False) → None` | Shut down the pool. With `force=True`, closes in-use connections. |
| `health_check` | `() → dict[str, bool]` | Per-connection health, keyed `connection_0`, `connection_1`, … **Not a bool**: `if pool.health_check():` is truthy for any non-empty pool. |
| `status` | `() → dict` | Pool status: `size`, `available`, `in_use`, `closed`. |

### `DFSAsyncFileSystem`

Async wrapper that dispatches all operations to a thread pool via `asyncio.to_thread()`, each borrowing one of `size` pooled DFS handles.

```python
from daos_fsspec.async_core import DFSAsyncFileSystem

fs = DFSAsyncFileSystem(pool="mypool", container="mycont")
```

Most methods are prefixed with `_` (fsspec async convention): `_ls`, `_info`, `_exists`, `_cat_file`, `_pipe_file`, `_mkdir`, `_rm`, `_mv`, `_find`, `_walk`, `_cat_ranges`, etc. The lifecycle and query helpers are not: `close()`, `health_check()`, `pool_space()` and `mount_attr()`. There is no `_close`.

### `DFSFileSystemWithMetrics`

Drop-in wrapper that records timing, byte counts, and error rates for all operations.

```python
from daos_fsspec.metrics import DFSFileSystemWithMetrics

fs = DFSFileSystemWithMetrics(pool="mypool", container="mycont")
# ... use normally ...
print(fs.metrics.summary())
```

| Attribute / Method | Description |
|--------------------|-------------|
| `fs.metrics` | `DFSMetrics` instance with aggregated counters. |
| `fs.metrics.summary()` | Returns dict with `uptime_seconds`, `total_operations`, `total_errors`, `total_bytes_read`, `total_bytes_written`, per-operation breakdowns. |
| `fs.metrics.reset()` | Reset all counters. |

## Deprecation Policy

This project follows [Semantic Versioning](https://semver.org/). Deprecated features will:
1. Emit a `DeprecationWarning` for at least 2 minor versions before removal.
2. Be documented in the [CHANGELOG](CHANGELOG.md).

Public API surface:
- `DFSFileSystem`: main filesystem class
- `DFSFile`: file handle class
- `DFSAsyncFileSystem`: async variant (in `daos_fsspec.async_core`)
- `DFSFileSystemPool`: connection pool (in `daos_fsspec.pool`)
- `DFSFileSystemWithMetrics`: instrumented wrapper (in `daos_fsspec.metrics`)
- `DFSMetrics` / `OperationMetrics`: metrics containers (in `daos_fsspec.metrics`)

Internal modules (`_bindings.py`) may change without notice.

## Examples

See the [examples guide](examples/README.md) for application patterns and the
21 Argonne scientific Python examples, including their dependencies, environment
variables, MPI usage, and known limitations.

## Testing

### Quick start (Docker)

```bash
# Unit tests only (no DAOS needed)
docker/test.sh

# Full test suite (builds containers, starts DAOS, runs all tests, tears down)
docker/test.sh --live

# Keep containers running after tests (for development)
docker/test.sh --live --keep
```

Or manually:

```bash
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
sudo sysctl vm.nr_hugepages=0
docker compose -f docker/docker-compose.test.yml up -d --build
docker exec daos-test-client daos container create test-pool fsspec-live-test --type=POSIX
docker exec -e DAOS_FSSPEC_TEST_POOL=test-pool -e DAOS_FSSPEC_TEST_CONTAINER=fsspec-live-test \
  daos-test-client python3 -m pytest /tmp/daos-fsspec/tests/ -v
```

See [`docker/README.md`](docker/README.md) for full Docker setup documentation.

The Docker setup automatically creates the pool on first start. The test container (`fsspec-live-test`) is created by the script or manually. pandas + pyarrow are installed in the container for parquet round-trip tests.

### Against a real DAOS cluster

```bash
export DAOS_FSSPEC_TEST_POOL="mypool"
export DAOS_FSSPEC_TEST_CONTAINER="mycontainer"
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite creates temporary directories (prefixed `fsspec-test-`) and cleans them up after each test.

## Contributing

Contributions welcome. This is a community project for the DAOS ecosystem.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture details, and submission guidelines.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Related projects

- [DAOS](https://github.com/daos-stack/daos): Distributed Asynchronous Object Storage
- [fsspec](https://github.com/fsspec/filesystem_spec): Pythonic filesystem interfaces
- [pydaos](https://github.com/daos-stack/daos/tree/master/src/client/pydaos): Official (limited) Python bindings
