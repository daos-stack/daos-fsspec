# daos-fsspec

[![PyPI version](https://img.shields.io/pypi/v/daos-fsspec)](https://pypi.org/project/daos-fsspec/)
[![Tests](https://github.com/daos-stack/daos-fsspec/actions/workflows/test.yml/badge.svg)](https://github.com/daos-stack/daos-fsspec/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/daos-stack/daos-fsspec)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/daos-fsspec)](https://pypi.org/project/daos-fsspec/)

**fsspec filesystem interface for [DAOS](https://docs.daos.io/) DFS (Distributed File System)**

Access DAOS POSIX containers from the entire PyData/ML ecosystem — pandas, Dask, HuggingFace Datasets, PyArrow, MLflow — without dfuse mounts, custom Dataset classes, or shelling out to `daos fs`.

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

## Why?

Most DAOS users in the HPC/AI space store training datasets, checkpoints, and experiment artifacts in DFS containers (DAOS's POSIX-compatible filesystem layer). There's no maintained fsspec backend for DAOS, which means the entire Python data ecosystem can't natively access these containers.

`daos-fsspec` fixes that with a lightweight ctypes binding to `libdfs.so` — no pydaos dependency, no subprocess overhead, works with any DAOS 2.6+ installation.

## Installation

```bash
pip install daos-fsspec
```

**Requirements:**
- Python 3.9+
- DAOS 2.6+ client libraries (`libdaos.so`, `libdfs.so`, `libgurt.so`) on `LD_LIBRARY_PATH` or in `/usr/lib64`
- A running DAOS agent (for pool/container access)

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

For asyncio-based applications, all operations are dispatched to Python worker threads via `asyncio.to_thread()`:

```python
import asyncio
from daos_fsspec.async_core import DFSAsyncFileSystem

async def main():
    fs = DFSAsyncFileSystem(pool="mypool", container="mycont")
    data = await fs._cat_file("/path/to/file")
    await fs._pipe_file("/output/data.bin", data)
    entries = await fs._ls("/")
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

- **16 MB chunked I/O** — reads and writes are automatically split into 16 MB chunks. Single-chunk paths (most files) have zero overhead.
- **`readdirplus` batching** — `ls()` and `walk()` use `dfs_readdirplus` to stat all entries in one call per batch (128 entries). No N+1 query pattern.
- **`cat_file` fast path** — uses a single `dfs_lookup_path` call instead of the `open` → `DFSFile` → `_resolve_path` chain (3 C calls vs 7 for depth-3 paths).
- **`cat_ranges` compatibility** — reads multiple byte ranges through a `ThreadPoolExecutor` (up to 8 workers). Calls on one `DFSFileSystem` mount are still serialized by the libdfs safety lock; use `DFSFileSystemPool` for real multi-handle parallelism.

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
│  │ _bindings.py — ctypes FFI to libdfs.so + libdaos.so │ │
│  └─────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  DAOS client library (DAOS agent → engines)               │
└──────────────────────────────────────────────────────────┘
```

### Thread safety

libdfs handles are not thread-safe. `daos-fsspec` uses a per-filesystem lock that serializes all libdfs calls through a single DFS mount. This is correct and sufficient for typical workloads (DataLoader workers, notebook cells, Dask threads).

For true concurrent DAOS I/O, use `DFSFileSystemPool`: every pooled connection has an independent DFS mount and lock.

### Fork safety

DAOS handles don't survive `fork()`. For `multiprocessing`-based DataLoaders, each worker process must create its own `DFSFileSystem` instance. Do not share a `DFSFileSystem` across `fork()`.

### Async support

`daos-fsspec` provides `DFSAsyncFileSystem` for asyncio compatibility. This is a **thin wrapper** that runs all DAOS operations in a thread pool via `asyncio.to_thread()`. It is not a true non-blocking implementation.

For most workloads (dataset loading, notebooks, moderate-concurrency web services), this is sufficient. For high-concurrency async workloads, use `DFSFileSystemPool` with a dedicated thread pool:

```python
from daos_fsspec.pool import DFSFileSystemPool

pool = DFSFileSystemPool(pool="mypool", container="mycont", size=8)
with pool.connection() as fs:
    data = fs.cat("/path/to/file")
```

### Why ctypes to libdfs?

| Approach | Pros | Cons |
|----------|------|------|
| **ctypes to libdfs** (this project) | Fast, full control, works with any DAOS 2.6+ | More upfront FFI work |
| pydaos | Official bindings | DFS API incomplete, hard to install |
| subprocess `daos fs` | Zero deps | Slow (fork per op), parsing stdout, no streaming |

## Known Limitations

- **O_APPEND not supported**: DAOS DFS does not support `O_APPEND` mode. Opening an existing file with `"ab"` does **not** append — writes begin at offset 0 and overwrite the existing bytes (opening a non-existent path creates it). To add to a file, read it, seek/`truncate`, and write explicitly, or write to a new object.
- **No concurrent writes to same file**: Multiple writers to the same file will produce undefined behavior. Coordinate writes externally.
- **No hard links**: DAOS DFS does not support hard links, only symbolic links.
- **No mmap**: Memory-mapped files are not supported.
- **No file locking**: `flock()` and `fcntl()` locks are not supported.
- **No depth-limited remove**: DAOS DFS does not provide native depth-limited removal, so `rm(maxdepth=...)` raises `NotImplementedError`.
- **glibc/Linux struct layout**: the pure-ctypes bindings validate the `struct stat` size at import time and refuse to load on a known architecture whose size doesn't match. Only **x86_64** is offset-verified field-by-field against real glibc; aarch64/ppc64le/s390x share its size and LP64 field order but are size-checked, not hardware-verified. Alpine/musl and unlisted architectures emit a warning and may need additional validation before production use.

## API Reference

### `DFSFileSystem`

Main filesystem class. Instantiate via `fsspec.filesystem("dfs", ...)` or directly.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool` | `str` | — | DAOS pool label (required) |
| `container` | `str` | — | DAOS container label (required) |
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
| `exists` | `(path) → bool` | Check if path exists (follows symlinks). |
| `lexists` | `(path) → bool` | Like `exists()`, does not follow symlinks. In DFS these are equivalent. |
| `islink` | `(path) → bool` | Return `True` if path is a symbolic link. |
| `mkdir` | `(path, create_parents=True) → None` | Create a directory. With `create_parents=True`, creates intermediate directories. |
| `rmdir` | `(path) → None` | Remove an empty directory. |
| `rm` | `(path, recursive=False, maxdepth=None) → None` | Remove a file or directory. With `recursive=True`, removes trees. Non-`None` `maxdepth` raises `NotImplementedError`. |
| `mv` | `(path1, path2) → None` | Rename/move a file or directory. |
| `touch` | `(path, truncate=True) → None` | Create empty file or update mtime. |
| `open` | `(path, mode="rb", **kwargs) → DFSFile` | Open a file. Supports `"rb"` and `"wb"`. `"ab"` is accepted but **does not truly append** — DFS has no `O_APPEND`, so writes start at offset 0 and overwrite existing content. See [Known Limitations](#known-limitations). |
| `cat` | `(path, recursive=False, on_error="raise") → bytes` | Read file(s) into memory. Inherited from fsspec. |
| `cat_file` | `(path, start=None, end=None) → bytes` | Read a file (or byte range) with optimized single-lookup path. |
| `cat_ranges` | `(paths, starts, ends) → list[bytes]` | Read byte ranges from multiple files in parallel. |
| `pipe` | `(path, data) → None` | Write bytes to a file. Inherited from fsspec. |
| `pipe_file` | `(path, value) → None` | Write bytes to a file in one shot (no buffering). |
| `head` | `(path, size=1024) → bytes` | Read the first `size` bytes. |
| `tail` | `(path, size=1024) → bytes` | Read the last `size` bytes. |
| `find` | `(path, maxdepth=None, withdirs=False, detail=False) → list \| dict` | Recursively list all files under path. |
| `find_parallel` | `(path, maxdepth=None, withdirs=False, detail=False, num_workers=4) → list \| dict` | Like `find()` but can use DAOS anchor splitting where available. Calls on one mount remain serialized by the libdfs safety lock. |
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
| `health_check` | `() → bool` | Verify all pool connections are alive. |
| `status` | `() → dict` | Pool status: `size`, `available`, `in_use`, `closed`. |

### `DFSAsyncFileSystem`

Async wrapper that dispatches all operations to a thread pool via `asyncio.to_thread()`.

```python
from daos_fsspec.async_core import DFSAsyncFileSystem

fs = DFSAsyncFileSystem(pool="mypool", container="mycont")
```

All methods are prefixed with `_` (fsspec async convention): `_ls`, `_info`, `_exists`, `_cat_file`, `_pipe_file`, `_mkdir`, `_rm`, `_mv`, `_find`, `_walk`, `_cat_ranges`, `_close`, etc.

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

This project follows [Semantic Versioning](https://semver.org/). Deprecated features will
emit a `DeprecationWarning` for at least 2 minor versions before removal.

Public API surface:
- `DFSFileSystem` — main filesystem class
- `DFSFile` — file handle class
- `DFSAsyncFileSystem` — async variant (in `daos_fsspec.async_core`)
- `DFSFileSystemPool` — connection pool (in `daos_fsspec.pool`)
- `DFSFileSystemWithMetrics` — instrumented wrapper (in `daos_fsspec.metrics`)
- `DFSMetrics` / `OperationMetrics` — metrics containers (in `daos_fsspec.metrics`)

Internal modules (`_bindings.py`) may change without notice.

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
sudo sysctl vm.nr_hugepages=4096
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

- [DAOS](https://github.com/daos-stack/daos) — Distributed Asynchronous Object Storage
- [fsspec](https://github.com/fsspec/filesystem_spec) — Pythonic filesystem interfaces
- [pydaos](https://github.com/daos-stack/daos/tree/master/src/client/pydaos) — Official (limited) Python bindings
