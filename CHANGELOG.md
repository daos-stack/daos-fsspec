# Changelog

All notable changes to `daos-fsspec` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.2.1] - 2026-07

Initial public release: an `fsspec` filesystem backend for DAOS DFS containers,
built on a pure-ctypes binding to `libdfs.so` / `libdaos.so` (DAOS 2.6+).

### Added
- `DFSFileSystem` — full fsspec `AbstractFileSystem` over DAOS DFS, with the
  `dfs://pool/container/path` URI scheme registered as an fsspec entry point.
- `DFSFile` buffered file handle; chunked 16 MiB I/O; `readdirplus`-batched
  `ls` / `find` / `walk` / `du`; optimized `cat_file` / `cat_ranges`.
- `DFSAsyncFileSystem` (asyncio), `DFSFileSystemPool` (concurrent connections),
  and `DFSFileSystemWithMetrics` (operation instrumentation).
- Retrying, reconnecting I/O path for transient DAOS errors (covering the
  `pipe_file` / `put_file` / `get_file` write paths); xattrs, symlinks,
  chmod/chown, truncate, and pool-space queries.
- Import-time `struct stat` ABI size gate, and a nightly self-hosted
  live-integration CI job against a real DAOS cluster.

[1.2.1]: https://github.com/daos-stack/daos-fsspec/releases/tag/v1.2.1
