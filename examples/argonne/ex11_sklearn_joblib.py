#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex11 - scikit-learn + joblib
============================
Train, persist with joblib, reload, predict. joblib takes a file object, so
fs.open() works directly."""

import io
import os

import fsspec
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex11_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

MODEL, DATA = f"{RDIR}/rf.joblib", f"{RDIR}/dataset.npz"

X, y = make_classification(n_samples=500, n_features=10, random_state=0)
clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
log("trained acc   :", round(clf.score(X, y), 4))

# --- persist -----------------------------------------------------------------
with fs.open(MODEL, "wb") as f:
    joblib.dump(clf, f)
log("saved model   :", fs.info(MODEL)["size"], "bytes")

# --- reload and re-predict ---------------------------------------------------
with fs.open(MODEL, "rb") as f:
    clone = joblib.load(f)
log("preds identic :", np.array_equal(clf.predict(X[:20]), clone.predict(X[:20])))
log("reloaded acc  :", round(clone.score(X, y), 4))

# --- the arrays themselves, as compressed npz --------------------------------
buf = io.BytesIO()
np.savez_compressed(buf, X=X, y=y)
fs.pipe_file(DATA, buf.getvalue())
z = np.load(io.BytesIO(fs.cat_file(DATA)))
log("arrays roundtr:", np.allclose(z["X"], X), np.array_equal(z["y"], y))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
