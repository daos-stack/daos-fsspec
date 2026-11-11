# Scientific Python examples

These examples were originally written by Kaushik Velusamy at Argonne Leadership
Computing Facility. They cover streaming files, array and table formats, model
checkpoints, image processing, MPI, and local staging.

Set an existing POSIX container before running:

```sh
export DAOS_POOL=test-pool DAOS_CONT=fsspec-live-test
python examples/argonne/ex01_hello_fsspec.py
```

Most examples use a per-rank directory. Rank information comes from the process
environment; standalone execution defaults to one rank. `ex14` uses a shared
directory and MPI barriers when launched with MPI. Install the optional libraries imported by
the example you want to run.

The live suite executes examples as subprocesses, checking exit status and cleanup.
Examples with unavailable dependencies are skipped. `ex20` is skipped because of
an unresolved native SIGSEGV; its import ordering must be preserved pending a
native backtrace. See [validation](../../docs/validation.md).

See the [examples guide](../README.md) for dependencies, environment variables,
and how these scripts differ from the application examples.
