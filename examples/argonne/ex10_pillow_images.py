#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex10 - images (Pillow)
======================
The real image-training data path: one object per image, decoded straight from
bytes with no temporary files. Pillow works in bytes in both directions, so
BytesIO pairs with pipe_file() and cat_file()."""

import io
import os

import fsspec
import numpy as np
from PIL import Image


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex10_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

# --- write 5 JPEGs -----------------------------------------------------------
for i in range(5):
    arr = np.full((64, 64, 3), i * 50, dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    fs.pipe_file(f"{RDIR}/img_{i:d}.jpg", buf.getvalue())
log("wrote         : 5 JPEGs")

files = sorted(fs.find(RDIR))
log("found         :", [f.rsplit("/", 1)[-1] for f in files])

# --- read one back -----------------------------------------------------------
img = Image.open(io.BytesIO(fs.cat_file(files[0])))
log("format        :", img.format, img.size, img.mode)
log("mean pixel    :", round(float(np.asarray(img).mean()), 1), "(img_0 is black)")

# --- PNG is lossless, so bytes round-trip exactly ----------------------------
arr = np.random.randint(0, 255, (32, 32, 3), dtype="uint8")
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format="PNG")
fs.pipe_file(f"{RDIR}/exact.png", buf.getvalue())
back = np.asarray(Image.open(io.BytesIO(fs.cat_file(f"{RDIR}/exact.png"))))
log("PNG lossless  :", np.array_equal(arr, back))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
