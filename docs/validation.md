# Validation and limitations

The 1.4.1 runtime has been tested locally against DAOS 2.6.5 and 2.8.0 over TCP.
Recorded full-suite results are 801 passed / 11 skipped on 2.6.5 and
807 passed / 5 skipped on 2.8.0. Skips cover optional example dependencies and
the unresolved `ex20` native crash. These are prior native runs; the source
export preserves executable Python behavior and retains the regression suite.

Python 3.13 offline validation recorded 644 passed, with 168 live tests deselected.
Targeted compatibility checks recorded 65 passed on each of Python 3.9-3.12.
Native tests cover initialization reference counts, failure paths, fork guards,
actual process exit, file cleanup ordering, and independent native consumers.

## Reproduce

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/ -m 'not live' --timeout=120 --timeout-method=thread
DAOS_VERSION=2.6.5 bash docker/test.sh --live --skip-unit
DAOS_VERSION=2.8.0 bash docker/test.sh --live --skip-unit
```

Run the two native versions sequentially. See [Docker setup](../docker/README.md).
The compiled ABI probe is `docker/abi_probe.py`; lifecycle fixtures are in
`tests/native/lifecycle.c`. Optional examples skip when their dependencies are absent.

## Open validation

- The original multi-node CXI exit-crash workload has not been rerun with the
  finalization fix. Local TCP exit tests do not establish elimination of that crash.
- Single-caller measurements were near baseline. Concurrent native measurements
  were too variable to establish performance parity; controlled repetitions are needed.
- The reported 128 MiB read stragglers remain unresolved.
- `examples/argonne/ex20_tokenizers_transformers.py` remains excluded from the live
  suite because of an unresolved native SIGSEGV; a native backtrace is needed.

Applications must finish outstanding work and close writers before filesystem
teardown. Exit cleanup is a backstop, not coordination of arbitrary daemon workers.

## Example output checks

The live suite checks `ex09` for a 9,559-byte HDF5 image, shape `(100, 10)`,
and equal round-trip data. It checks `ex18` for 2,103-byte MessagePack and
4,127-byte JSON output. Whitespace formatting is excluded from these checks.
