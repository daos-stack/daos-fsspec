#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex05 - torch Dataset + DataLoader
=================================
daos-fsspec ships no torch integration; pydaos offers daos_torch.Dataset,
this does not; so you write the Dataset yourself. It is about twenty lines,
and that is the whole porting cost."""

import io
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from daos_fsspec import get_fs


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex05_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


# PBS sets TMPDIR to a ~78-char path; python's multiprocessing binds a Unix
# socket under it and blows the 108-byte AF_UNIX limit, so every spawn worker
# dies and the parent hangs. Must be set BEFORE torch spawns anything.
os.environ["TMPDIR"] = f"/tmp/r{RANK:d}"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# get_fs() is imported at the top: a module-level pid-keyed dict used to live
# here, hand-rolled so a DFSFileSystem is never pickled into a spawn worker.
# That pattern is now shipped as daos_fsspec.get_fs(); which this example
# motivated; with the same pid keying plus a fork-generation key,
# closed-instance replacement, and a lock so racing workers open one
# connection rather than several.


class DaosNpzDataset(Dataset):
    """Holds ONLY strings, so it pickles cleanly for spawn workers."""

    def __init__(self, pool, cont, prefix):
        self.pool, self.cont = pool, cont
        self.files = sorted(get_fs(pool, cont).find(prefix))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        raw = get_fs(self.pool, self.cont).cat_file(self.files[i])
        return torch.from_numpy(np.load(io.BytesIO(raw))["x"].astype("float32"))


def main():
    fs = get_fs(POOL, CONT)
    log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")
    if fs.exists(RDIR):
        fs.rm(RDIR, recursive=True)
    fs.mkdirs(RDIR)

    for i in range(20):
        buf = io.BytesIO()
        np.savez(buf, x=np.full((3, 3), float(i)))
        fs.pipe_file(f"{RDIR}/s{i:03d}.npz", buf.getvalue())
    log("generated     : 20 samples in", RDIR)

    ds = DaosNpzDataset(POOL, CONT, RDIR)
    log("dataset length:", len(ds))
    log("sample 0      :", ds[0].flatten()[:3].tolist(), "shape", tuple(ds[0].shape))

    n = sum(b.shape[0] for b in DataLoader(ds, batch_size=4))
    log("serial pass   :", n, "samples")

    # spawn, NOT fork - fork deadlocks inside libdaos with no error
    loader = DataLoader(ds, batch_size=4, num_workers=2, multiprocessing_context="spawn")
    log("2 spawn worker:", sum(b.shape[0] for b in loader), "samples")

    fs.rm(RDIR, recursive=True)
    log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
    log("done")


if __name__ == "__main__":  # REQUIRED: spawn re-imports this module
    main()
