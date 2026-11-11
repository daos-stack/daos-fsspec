#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex07 - safetensors checkpoints
==============================
safetensors is the HuggingFace default: zero-copy, no pickle, safe to load from
an untrusted source. Unlike torch it works in bytes rather than file objects,
so the pairing is save()/pipe_file() and cat_file()/load()."""

import os
import time

import fsspec
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex07_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

P, SD = f"{RDIR}/model.safetensors", f"{RDIR}/sd.safetensors"

tensors = {"w": torch.ones(1024, 1024), "b": torch.arange(1024, dtype=torch.float32)}
mb = (1024 * 1024 * 4 + 1024 * 4) / 1024 / 1024

t0 = time.perf_counter()
fs.pipe_file(P, st_save(tensors))
dt = time.perf_counter() - t0
log(f"saved  {mb:.1f} MB : {dt:.3f}s  ({mb / dt:.1f} MB/s)")

t0 = time.perf_counter()
back = st_load(fs.cat_file(P))
dt = time.perf_counter() - t0
log(f"loaded {mb:.1f} MB : {dt:.3f}s  ({mb / dt:.1f} MB/s)")

log("keys          :", sorted(back))
log("w equal       :", torch.equal(back["w"], tensors["w"]))
log("dtype kept    :", back["w"].dtype == tensors["w"].dtype)


# --- a real model ------------------------------------------------------------
def net():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))


model, x = net(), torch.randn(4, 16)
with torch.no_grad():
    before = model(x)

# .contiguous(): safetensors rejects shared / non-contiguous storage
fs.pipe_file(SD, st_save({k: v.contiguous() for k, v in model.state_dict().items()}))
clone = net()
clone.load_state_dict(st_load(fs.cat_file(SD)))
with torch.no_grad():
    after = clone(x)
log("reload identic:", torch.allclose(before, after))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
