#!/usr/bin/env python3
"""Verify every ctypes Structure in _bindings.py against the installed DAOS headers.

A ctypes layout mismatch does not crash; it silently returns plausible
garbage. So a green test suite cannot prove ABI correctness, and the version's
own daos-devel headers are the only authority. This generates a C program that
prints sizeof() and offsetof() for every field we declare, compiles it against
those headers, and diffs the result against what ctypes computes.

Run it inside the client container, which has the matching daos-devel:

    docker cp docker/abi_probe.py daos-test-client:/tmp/abi_probe.py
    docker exec -e DAOS_VERSION=$DAOS_VERSION daos-test-client \
        python3 /tmp/abi_probe.py

gcc and libuuid-devel are present in the test client image so the live matrix
can enforce this probe on every run, not merely record an occasional manual
check.

Set DAOS_FSSPEC_SRC if the package is not at /tmp/daos-fsspec.

Exit 0 = every checked field and size matches. Exit 1 = at least one mismatch.
2 = the probe itself failed to build or run.
"""

import ctypes
import os
import subprocess
import sys

sys.path.insert(0, os.environ.get("DAOS_FSSPEC_SRC", "/tmp/daos-fsspec"))
from daos_fsspec import _bindings as b  # noqa: E402

# (python_struct, c_type_name, is_typedef, [(py_field, c_offsetof_expr), ...])
# c_offsetof_expr is the member expression as written in C, which differs from
# our field name wherever we flatten a nested struct (e.g. doi_oid.lo).
SPECS = [
    (
        b.dfs_obj_info_t,
        "dfs_obj_info_t",
        True,
        [
            ("doi_oclass_id", "doi_oclass_id"),
            ("doi_chunk_size", "doi_chunk_size"),
            ("doi_dir_oclass_id", "doi_dir_oclass_id"),
            ("doi_file_oclass_id", "doi_file_oclass_id"),
            ("doi_oid_lo", "doi_oid.lo"),
            ("doi_oid_hi", "doi_oid.hi"),
        ],
    ),
    (
        b.dfs_attr_t,
        "dfs_attr_t",
        True,
        [
            ("da_id", "da_id"),
            ("da_chunk_size", "da_chunk_size"),
            ("da_oclass_id", "da_oclass_id"),
            ("da_props", "da_props"),
            ("da_mode", "da_mode"),
            ("da_dir_oclass_id", "da_dir_oclass_id"),
            ("da_file_oclass_id", "da_file_oclass_id"),
            ("da_hints", "da_hints"),
        ],
    ),
    (
        b.d_iov_t,
        "d_iov_t",
        True,
        [
            ("iov_buf", "iov_buf"),
            ("iov_buf_len", "iov_buf_len"),
            ("iov_len", "iov_len"),
        ],
    ),
    (
        b.d_sg_list_t,
        "d_sg_list_t",
        True,
        [
            ("sg_nr", "sg_nr"),
            ("sg_nr_out", "sg_nr_out"),
            ("sg_iovs", "sg_iovs"),
        ],
    ),
    (
        b.daos_anchor_t,
        "daos_anchor_t",
        True,
        [
            ("da_type", "da_type"),
            ("da_shard", "da_shard"),
            ("da_flags", "da_flags"),
            ("da_sub_anchors", "da_sub_anchors"),
            ("da_buf", "da_buf"),
        ],
    ),
    (b.daos_handle_t, "daos_handle_t", True, [("cookie", "cookie")]),
    (
        b.daos_pool_cont_info,
        "struct daos_pool_cont_info",
        False,
        [("pci_uuid", "pci_uuid"), ("pci_label", "pci_label")],
    ),
    (
        b._daos_space,
        "struct daos_space",
        False,
        [("s_total", "s_total"), ("s_free", "s_free")],
    ),
    (
        b._daos_pool_space,
        "struct daos_pool_space",
        False,
        [
            ("ps_space", "ps_space"),
            ("ps_free_min", "ps_free_min"),
            ("ps_free_max", "ps_free_max"),
            ("ps_free_mean", "ps_free_mean"),
            ("ps_ntargets", "ps_ntargets"),
        ],
    ),
    (
        b._daos_pool_info,
        "daos_pool_info_t",
        True,
        [
            ("pi_uuid", "pi_uuid"),
            ("pi_ntargets", "pi_ntargets"),
            ("pi_nnodes", "pi_nnodes"),
            ("pi_ndisabled", "pi_ndisabled"),
            ("pi_map_ver", "pi_map_ver"),
            ("pi_leader", "pi_leader"),
            ("pi_bits", "pi_bits"),
            ("pi_space", "pi_space"),
            ("pi_rebuild_st", "pi_rebuild_st"),
        ],
    ),
    # libc, not DAOS, but our DFS I/O depends on it exactly as much.
    (
        b.struct_stat,
        "struct stat",
        False,
        [
            ("st_dev", "st_dev"),
            ("st_ino", "st_ino"),
            ("st_nlink", "st_nlink"),
            ("st_mode", "st_mode"),
            ("st_uid", "st_uid"),
            ("st_gid", "st_gid"),
            ("st_rdev", "st_rdev"),
            ("st_size", "st_size"),
            ("st_blksize", "st_blksize"),
            ("st_blocks", "st_blocks"),
        ],
    ),
    (
        b.dirent,
        "struct dirent",
        False,
        [
            ("d_ino", "d_ino"),
            ("d_off", "d_off"),
            ("d_reclen", "d_reclen"),
            ("d_type", "d_type"),
            ("d_name", "d_name"),
        ],
    ),
]

C_HEAD = r"""
#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <sys/stat.h>
#include <dirent.h>
#include <daos.h>
#include <daos_fs.h>
#include <daos_types.h>
#include <daos_obj_class.h>
#include <daos_pool.h>
#include <gurt/types.h>

int main(void) {
"""


def gen_c():
    lines = [C_HEAD]
    for _py, cname, _td, fields in SPECS:
        # Key must be whitespace-free: "struct daos_pool_cont_info" would
        # otherwise split across two fields and break the parse below.
        key = cname.replace(" ", "_")
        lines.append(f'    printf("SIZE %s %zu\\n", "{key}", sizeof({cname}));')
        for pyf, cexpr in fields:
            lines.append(
                f'    printf("OFF %s %s %zu\\n", "{key}", "{pyf}", offsetof({cname}, {cexpr}));'
            )
    # Scalar typedef widths that caused real bugs (oclass truncation).
    for t in ("daos_oclass_id_t", "daos_size_t", "mode_t", "daos_obj_id_t"):
        lines.append(f'    printf("SCALAR %s %zu\\n", "{t}", sizeof({t}));')
    lines.append("    return 0;\n}")
    return "\n".join(lines)


def main():
    src = "/tmp/abi_probe.c"
    exe = "/tmp/abi_probe"
    with open(src, "w") as f:
        f.write(gen_c())

    cc = subprocess.run(
        ["gcc", "-O0", "-o", exe, src, "-I/usr/include"],
        capture_output=True,
        text=True,
    )
    if cc.returncode != 0:
        print("COMPILE FAILED:\n" + cc.stderr[-4000:])
        return 2

    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        print("PROBE FAILED:\n" + out.stderr)
        return 2

    c_sizes, c_offs, c_scalars = {}, {}, {}
    for line in out.stdout.splitlines():
        p = line.split()
        if p[0] == "SIZE":
            c_sizes[p[1]] = int(p[2])
        elif p[0] == "OFF":
            c_offs[(p[1], p[2])] = int(p[3])
        elif p[0] == "SCALAR":
            c_scalars[p[1]] = int(p[2])

    daos_version = os.environ.get("DAOS_VERSION", "unknown")
    print(f"=== ABI verification against DAOS {daos_version} headers ===\n")

    print("scalar typedef widths:")
    for t, sz in sorted(c_scalars.items()):
        print(f"  {t:20s} = {sz}")
    print()

    failures = []
    for py, cname, _td, fields in SPECS:
        key = cname.replace(" ", "_")
        py_size = ctypes.sizeof(py)
        c_size = c_sizes[key]
        size_ok = py_size == c_size
        status = "OK " if size_ok else "BAD"
        print(f"[{status}] {cname}: sizeof ctypes={py_size} c={c_size}")
        if not size_ok:
            failures.append(f"{cname}: sizeof ctypes={py_size} != c={c_size}")

        for pyf, cexpr in fields:
            py_off = getattr(py, pyf).offset
            c_off = c_offs[(key, pyf)]
            if py_off != c_off:
                failures.append(f"{cname}.{pyf} (C: {cexpr}): offset ctypes={py_off} != c={c_off}")
                print(f"       BAD {pyf:22s} ctypes={py_off:4d} c={c_off:4d}  <-- MISMATCH")

    print()
    if failures:
        print(f"!!! {len(failures)} MISMATCH(ES) against DAOS {daos_version} headers:")
        for f in failures:
            print(f"  - {f}")
        return 1
    total = sum(len(s[3]) for s in SPECS)
    print(f"ALL CLEAR: {len(SPECS)} structs, {total} field offsets, all sizes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
