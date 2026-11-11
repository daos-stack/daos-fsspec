# Changelog

## 1.4.1

- Preserve existing destinations on failed uploads and downloads; reject premature download EOF.
- Refuse automatic reconnect while files are open and report discarded buffered writes on close.
- Reject URIs targeting a different pool/container and validate path lengths in UTF-8 bytes.

- Balance DAOS and DFS initialization references and finalize them in order at normal process exit.
- Close known file handles before filesystem teardown; report cleanup errors and discard abandoned write buffers with a warning.
- Reject inherited DAOS state after fork; use the spawn start method for multiprocessing.
- Add process-local filesystem caching through `get_fs()`.
- Fix async pool saturation, pipelined-read cleanup and short reads, reconnect handle generations, and concurrent directory operations.
- Add optional bounded directory-listing caching, disabled by default.
- Retain `find_parallel()` as a compatibility alias for serial `find()`.
- Expand regression, native lifecycle, ABI, and DAOS 2.6.5/2.8.0 integration coverage.

See [validation and limitations](docs/validation.md) for tested configurations and outstanding validation.
