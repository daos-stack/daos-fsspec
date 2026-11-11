#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex14 - MPI file-per-rank checkpointing
======================================
File-per-rank is the simplest correct checkpoint pattern on an object store:
every rank owns its own object, so there is no shared-file locking and no
coordination on the write path. This is how most HPC checkpointing works."""

import io
import os

import numpy as np

# MPI init can ABORT the process on a login node, so guard the import.
have_mpi = False
rank, size, comm = 0, 1, None

_under_mpi = any(
    os.environ.get(v) is not None
    for v in ("PALS_RANKID", "PMI_RANK", "PMI_SIZE", "PALS_LOCAL_SIZE")
)
if _under_mpi or os.environ.get("PBS_JOBID"):
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        have_mpi = True
    except Exception as e:
        print("mpi4py unavailable, single-rank fallback:", type(e).__name__)

# Imported after the guarded mpi4py block, not at the top of the file.
#
# E402 is silenced rather than satisfied: mpi4py cannot move to the top,
# because importing it calls MPI_Init, which aborts the process outside an
# allocation; that is what the guard above exists for. Hoisting fsspec above
# it would be harmless in itself (daos-fsspec dlopens libdaos lazily, so no
# native DAOS library is loaded at import), but ex20 in this directory is an
# unresolved SIGSEGV that depends on native-library initialisation order, so
# the delivered ordering is left exactly as validated.
import fsspec  # noqa: E402


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")
D = "/ex14_mpi_ckpt"  # ONE shared dir: this example is collective
fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)


def log(*a):
    print(f"[rank {rank:d}/{size:d}]", *a, flush=True)


# --- rank 0 owns the directory; everyone else waits for it -------------------
if rank == 0:
    if fs.exists(D):
        fs.rm(D, recursive=True)
    fs.mkdirs(D)
    log("created shared dir:", D)
if have_mpi:
    comm.Barrier()  # nobody writes before the directory exists

# --- each rank writes its own shard: no locking, no coordination -------------
payload = np.full(1000, rank, dtype="float32")
buf = io.BytesIO()
np.save(buf, payload)
shard = f"{D}/rank_{rank:04d}.npy"
fs.pipe_file(shard, buf.getvalue())
log("wrote shard   :", shard.rsplit("/", 1)[-1])

# rank-local read-back only - no extra collectives, which risk deadlock
mine = np.load(io.BytesIO(fs.cat_file(shard)))
log("my shard      : shape", mine.shape, "all == rank:", bool(np.all(mine == rank)))

if have_mpi:
    comm.Barrier()  # all shards on disk before rank 0 lists them

# --- rank 0 verifies the WHOLE checkpoint ------------------------------------
# Safe to list here only because the barrier guarantees nobody is still writing
# and nobody deletes until after the next barrier.
if rank == 0:
    files = sorted(fs.find(D))
    log("checkpoint has:", len(files), f"shards (expected {size:d})")
    ok = True
    for f in files:
        r = int(f.rsplit("_", 1)[-1].split(".")[0])
        arr = np.load(io.BytesIO(fs.cat_file(f)))
        if not (arr.shape == (1000,) and np.all(arr == r)):
            ok = False
            log("  MISMATCH in", f)
    log("all shards ok :", ok)

if have_mpi:
    comm.Barrier()  # nobody deletes while rank 0 is still reading

if rank == 0:
    fs.rm(D, recursive=True)
    log("cleaned up    :", D)

if not have_mpi:
    log("(no PBS allocation - ran single-rank; use mpiexec inside a job)")
log("done")
