#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex20 - tokenizers + transformers
================================
A tokenizer trained from scratch and a small transformer model, both persisted
to DAOS. Everything runs offline; the model is built from a config rather
than downloaded, because Aurora compute nodes have no outbound internet."""

import os
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Preserve native-library import order while the SIGSEGV remains unresolved.
# See README.md and ../../docs/validation.md.
# isort: off
import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402

import fsspec  # noqa: E402


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex20_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)


# --- train a tiny BPE tokenizer, no download ---------------------------------
corpus = [
    "daos is an object store",
    "fsspec gives a filesystem api",
    "aurora runs on intel gpus",
    "tokenizers train fast",
] * 50
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    corpus, trainers.BpeTrainer(vocab_size=200, special_tokens=["[UNK]", "[PAD]"])
)

# a tokenizer serialises to a JSON string -> one object write
TOKJSON = f"{RDIR}/tokenizer.json"
fs.pipe_file(TOKJSON, tok.to_str().encode())
log("tokenizer     :", fs.info(TOKJSON)["size"], "bytes")

back = Tokenizer.from_str(fs.cat_file(TOKJSON).decode())
enc_a, enc_b = tok.encode("daos is fast").ids, back.encode("daos is fast").ids
log("vocab size    :", back.get_vocab_size())
log("encoding      :", enc_a, "| roundtrip equal:", enc_a == enc_b)

# --- a tiny transformer built from config (no hub download) ------------------
cfg = BertConfig(
    vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
)
model = BertModel(cfg)
model.eval()  # dropout is ON by default -> nondeterministic output
log("model params  :", sum(p.numel() for p in model.parameters()))

ids = torch.tensor([enc_a])
with torch.no_grad():
    out_before = model(ids).last_hidden_state

# save_pretrained writes a local dir; upload it file by file
with tempfile.TemporaryDirectory() as td:
    model.save_pretrained(td)
    for name in os.listdir(td):
        with open(os.path.join(td, name), "rb") as fh:
            fs.pipe_file(f"{RDIR}/{name}", fh.read())
log("uploaded      :", sorted(p.rsplit("/", 1)[-1] for p in fs.ls(RDIR, detail=False)))

# --- download back and confirm identical inference ---------------------------
with tempfile.TemporaryDirectory() as td:

    local = os.path.join(td, "model")
    fs.get(RDIR, local, recursive=True)
    clone = BertModel.from_pretrained(local)
    clone.eval()
with torch.no_grad():
    out_after = clone(ids).last_hidden_state

log("max abs diff  :", (out_before - out_after).abs().max().item())
log("inference same:", torch.allclose(out_before, out_after, atol=1e-6))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
