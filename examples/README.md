# Examples

The four scripts in this directory demonstrate common application patterns.
The [Argonne examples](argonne/README.md) cover individual scientific Python
libraries and multi-process workloads. Both sets require DAOS client libraries,
a running agent, and an existing POSIX container. DAOS 2.6.5 and 2.8.0 are tested.

## Application examples

```sh
python -m pip install 'daos-fsspec[parquet]'
export DAOS_POOL=mypool
export DAOS_CONTAINER=mycont
```

If you need to create a container first:

```sh
daos container create "$DAOS_POOL" "$DAOS_CONTAINER" --type=POSIX
```

| Script | Demonstrates |
|--------|--------------|
| [dataset_staging.py](dataset_staging.py) | Write a sharded Parquet dataset and read it through `dfs://` URLs |
| [checkpoint_store.py](checkpoint_store.py) | Write checkpoints to temporary paths, rename them, and resume from the latest checkpoint |
| [concurrent_pool.py](concurrent_pool.py) | Share a connection pool across 16 threads and verify the data |
| [capacity_report.py](capacity_report.py) | Query pool capacity, directory usage, connection health, and operation metrics |

Run an individual script or all four:

```sh
python examples/dataset_staging.py
bash examples/run_all.sh
```

The first three scripts write into `/datasets/sensor-v1/train`,
`/experiments/run-42`, and `/concurrent-example`. They leave their output for
inspection and may replace existing example data when rerun. Use a test container.
The capacity report reads the existing container without creating example data.

## Argonne scientific Python examples

The 21 scripts in [argonne/](argonne/README.md) were originally written by
Kaushik Velusamy at Argonne Leadership Computing Facility. They demonstrate
streaming I/O, NumPy, pandas, PyArrow, PyTorch, Hugging Face, HDF5, images,
SciPy, scikit-learn, MPI, serialization, and local staging for SQLite.

Start with `ex01_hello_fsspec.py`, then choose an example for the library you use.
Install that example's optional dependencies before running it. For example:

```sh
export DAOS_POOL=mypool
export DAOS_CONT=mycont
python examples/argonne/ex01_hello_fsspec.py

python -m pip install numpy
python examples/argonne/ex04_numpy_samples.py
```

The Argonne scripts use **`DAOS_CONT`**, while the four application examples use
**`DAOS_CONTAINER`**. Set the appropriate variable explicitly.

Most Argonne scripts use a separate directory for each rank and default to a
single process. `ex14_mpi_ranks.py` uses a shared directory, per-rank files, and
MPI barriers. Run its collective path under an MPI launcher in an allocation
with DAOS access. Concurrent runs should use separate containers to avoid
collisions between example directories.

The live test suite runs examples as subprocesses and checks their exit status,
cleanup, and selected output values. Missing optional dependencies cause skips.
`ex20_tokenizers_transformers.py` remains skipped because of an unresolved native
SIGSEGV. See [validation and limitations](../docs/validation.md).

## Filesystem details

- Create parent directories with `fs.makedirs(parent, exist_ok=True)` before
  writing a nested path.
- `du()` returns a per-file dictionary by default. Pass `total=True` for a byte count.
- Append is not atomic. Coordinate writers if more than one process may update a file.
