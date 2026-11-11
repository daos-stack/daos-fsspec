#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex06 - torch checkpoint save/load
=================================
torch.save and torch.load take a file object, so fs.open() plugs straight in
with no staging step. The pydaos equivalents are ckpt.writer(name) and
ckpt.reader(name)."""

import io
import os
import time

import fsspec
import torch


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex06_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

CKPT, MODEL = f"{RDIR}/ckpt.pt", f"{RDIR}/model.pt"

# --- a 4 MB tensor -----------------------------------------------------------
data = {"a": torch.ones(1048576)}

t0 = time.perf_counter()
with fs.open(CKPT, "wb") as f:
    torch.save(data, f)
dt = time.perf_counter() - t0
log(f"saved  4 MB   : {dt:.3f}s  ({4 / dt:.1f} MB/s)")

t0 = time.perf_counter()
with fs.open(CKPT, "rb") as f:
    back = torch.load(io.BytesIO(f.read()), weights_only=True)
dt = time.perf_counter() - t0
log(f"loaded 4 MB   : {dt:.3f}s  ({4 / dt:.1f} MB/s)")
log("tensors equal :", torch.equal(back["a"], data["a"]))


# --- a real model state_dict -------------------------------------------------
def net():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))


model = net()
x = torch.randn(4, 16)
with torch.no_grad():
    before = model(x)

with fs.open(MODEL, "wb") as f:
    torch.save(model.state_dict(), f)

clone = net()
with fs.open(MODEL, "rb") as f:
    clone.load_state_dict(torch.load(io.BytesIO(f.read()), weights_only=True))
with torch.no_grad():
    after = clone(x)

log("max abs diff  :", (before - after).abs().max().item())
log("reload identic:", torch.allclose(before, after))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
