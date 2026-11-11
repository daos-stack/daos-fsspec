#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex17 - matplotlib figures to DAOS
=================================
Saving figures from a batch job with no display attached. matplotlib.use("Agg")
must be called before pyplot is imported, or matplotlib goes looking for an X
display and fails."""

import io
import os

import matplotlib

matplotlib.use("Agg")  # headless: MUST be set before pyplot
import fsspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex17_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

PNG, PDF = f"{RDIR}/curves.png", f"{RDIR}/hist.pdf"

x = np.linspace(0, 4 * np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos")
ax.set_title("training curves")
ax.legend()
fig.tight_layout()

# figure -> bytes -> one object write
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
fs.pipe_file(PNG, buf.getvalue())
log("wrote png     :", fs.info(PNG)["size"], "bytes")

# PDF too - vector, for papers
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.hist(np.random.default_rng(0).normal(size=1000), bins=30)
buf2 = io.BytesIO()
fig2.savefig(buf2, format="pdf")
plt.close(fig2)
fs.pipe_file(PDF, buf2.getvalue())
log("wrote pdf     :", fs.info(PDF)["size"], "bytes")

# --- verify it is a real PNG, by magic bytes and by reopening ----------------
log("png magic ok  :", fs.cat_file(PNG, start=0, end=8) == b"\x89PNG\r\n\x1a\n")
img = Image.open(io.BytesIO(fs.cat_file(PNG)))
log("reopened as   :", img.format, img.size)

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
