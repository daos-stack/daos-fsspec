#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex19 - archives and pickle
==========================
Bundling many small files into one archive object is a genuine performance
strategy here. Small-file I/O is the slow path on an object store, so packing a
thousand samples into a single tar can beat writing a thousand objects."""

import io
import os
import pickle
import tarfile
import zipfile

import cloudpickle
import fsspec
import numpy as np


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex19_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

TAR, ZIP = f"{RDIR}/bundle.tar.gz", f"{RDIR}/docs.zip"
PKL, FNPKL = f"{RDIR}/obj.pkl", f"{RDIR}/fn.pkl"

# --- tar bundle of 50 small arrays -> ONE object -----------------------------
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for i in range(50):
        payload = np.full(100, i, dtype="int32").tobytes()
        info = tarfile.TarInfo(name=f"sample_{i:03d}.bin")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
fs.pipe_file(TAR, buf.getvalue())
log("tar.gz        :", fs.info(TAR)["size"], "bytes (1 object, not 50)")

with tarfile.open(fileobj=io.BytesIO(fs.cat_file(TAR)), mode="r:gz") as tar:
    names = tar.getnames()
    first = np.frombuffer(tar.extractfile(names[0]).read(), dtype="int32")
log("members       :", len(names), "| first array[0] =", first[0])

# --- zip, with random access to members --------------------------------------
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for i in range(20):
        z.writestr(f"doc_{i:d}.txt", f"document number {i:d}\n" * 10)
fs.pipe_file(ZIP, buf.getvalue())

with zipfile.ZipFile(io.BytesIO(fs.cat_file(ZIP))) as z:
    log("zip members   :", len(z.namelist()))
    log("one member    :", z.read("doc_7.txt").split(b"\n")[0])

# --- pickle, and cloudpickle for things pickle refuses -----------------------
obj = {"arr": np.arange(10), "meta": {"tag": "run42"}}
fs.pipe_file(PKL, pickle.dumps(obj))
back = pickle.loads(fs.cat_file(PKL))
log("pickle        :", np.array_equal(back["arr"], obj["arr"]), back["meta"])

scale = 3
fs.pipe_file(FNPKL, cloudpickle.dumps(lambda x: x * scale))
log("cloudpickle   :", cloudpickle.loads(fs.cat_file(FNPKL))(14), "(lambda, x*3)")

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
