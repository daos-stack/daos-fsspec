#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex18 - config + metadata formats
================================
YAML, JSON, TOML and msgpack; the unglamorous ninety percent of the files in
a run directory."""

import json
import os

import fsspec
import msgpack
import tomllib
import yaml


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex18_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

cfg = {"model": "resnet50", "lr": 0.001, "epochs": 90, "layers": [64, 128, 256], "augment": True}

# --- YAML: bytes in, bytes out -----------------------------------------------
fs.pipe_file(f"{RDIR}/config.yaml", yaml.safe_dump(cfg).encode())
back = yaml.safe_load(fs.cat_file(f"{RDIR}/config.yaml"))
log("yaml          :", back["model"], back["lr"], "| equal:", back == cfg)

# --- JSON: via a TEXT-mode file object ---------------------------------------
with fs.open(f"{RDIR}/metrics.json", "w") as f:
    json.dump({"loss": [0.9, 0.5, 0.3], "acc": [0.4, 0.7, 0.88]}, f)
with fs.open(f"{RDIR}/metrics.json", "r") as f:
    m = json.load(f)
log("json          :", len(m["loss"]), "epochs, final acc", m["acc"][-1])

# --- TOML: stdlib, read-only --------------------------------------------------
fs.pipe_file(f"{RDIR}/pyproject.toml", b'[tool.run]\nname = "exp42"\nseed = 7\n')
t = tomllib.loads(fs.cat_file(f"{RDIR}/pyproject.toml").decode())
log("toml          :", t["tool"]["run"])

# --- msgpack: compact binary, good for many small records --------------------
records = [{"step": i, "loss": 1.0 / (i + 1)} for i in range(100)]
fs.pipe_file(f"{RDIR}/log.msgpack", msgpack.packb(records))
r = msgpack.unpackb(fs.cat_file(f"{RDIR}/log.msgpack"))
size_mp = fs.info(f"{RDIR}/log.msgpack")["size"]
size_js = len(json.dumps(records).encode())
log("msgpack       :", len(r), "records,", size_mp, "bytes vs json", size_js)

# safe: only this rank writes to RDIR
log("all files     :", sorted(p.rsplit("/", 1)[-1] for p in fs.ls(RDIR, detail=False)))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
