# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""ctypes bindings to libdaos.so and libdfs.so.

Wraps the DAOS / DFS C API needed for fsspec:
pool connect/disconnect, container open/close/create/destroy/list,
DFS mount/umount, file open/close/read/write, stat, setattr,
readdirplus, mkdir, remove, move, symlinks.

Requires DAOS 2.6+ libraries on LD_LIBRARY_PATH or in /usr/lib64.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
import platform as _platform
import stat as stat_mod
import threading
import uuid
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int64,
    c_size_t,
    c_uint,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
    sizeof,
)
from dataclasses import dataclass
from typing import Any

from daos_fsspec._runtime import Runtime as _Runtime

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


def _load_lib(name: str) -> ctypes.CDLL:
    """Load a shared library by name, raising a clear error if missing."""
    path = ctypes.util.find_library(name)
    if path is None:
        for prefix in ("/usr/lib64", "/usr/lib", "/usr/local/lib64", "/usr/local/lib"):
            candidate = os.path.join(prefix, f"lib{name}.so")
            if os.path.exists(candidate):
                path = candidate
                break
    if path is None:
        raise OSError(
            f"Cannot find lib{name}.so. Install DAOS 2.6+ and ensure "
            f"the library is on LD_LIBRARY_PATH or in /usr/lib64."
        )
    return ctypes.CDLL(path, use_errno=True)


_libdaos: ctypes.CDLL | None = None
_libdfs: ctypes.CDLL | None = None
_lib_lock = threading.Lock()
# Set last, after both handles exist *and* signatures are applied. The
# unlocked fast path in _ensure_libs() tests this and nothing else: testing
# _libdaos instead was a broken double-checked lock, because _libdaos is
# assigned before _libdfs and before _setup_signatures. A second thread
# arriving in that window got (_libdaos, None); AttributeError on the first
# libdfs call; or a CDLL with no argtypes, which is an ABI hazard rather
# than an error. Two threads constructing a DFSFileSystem concurrently was
# enough.
_libs_ready = False


def _ensure_libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Load libdaos/libdfs once, applying signatures before publishing them."""
    global _libdaos, _libdfs, _libs_ready
    if _libs_ready:
        return _libdaos, _libdfs  # type: ignore[return-value]
    with _lib_lock:
        if _libs_ready:
            return _libdaos, _libdfs  # type: ignore[return-value]
        try:
            _load_lib("gurt")
        except OSError:
            _logger.debug("libgurt.so not found (will be auto-loaded by libdaos)")
        libdaos = _load_lib("daos")
        libdfs = _load_lib("dfs")
        # Signatures first, then publish. Assigning the globals before
        # _setup_signatures let another thread see argtype-less CDLLs.
        _setup_signatures(libdaos, libdfs)
        _libdaos = libdaos
        _libdfs = libdfs
        _libs_ready = True
    return _libdaos, _libdfs


# ---------------------------------------------------------------------------
# C struct definitions
# ---------------------------------------------------------------------------


class daos_handle_t(Structure):
    _fields_ = [("cookie", c_uint64)]


class d_iov_t(Structure):
    _fields_ = [
        ("iov_buf", c_void_p),
        ("iov_buf_len", c_size_t),
        ("iov_len", c_size_t),
    ]


class d_sg_list_t(Structure):
    _fields_ = [
        ("sg_nr", c_uint32),
        ("sg_nr_out", c_uint32),
        ("sg_iovs", POINTER(d_iov_t)),
    ]


DAOS_ANCHOR_BUF_MAX = 104


class daos_anchor_t(Structure):
    _fields_ = [
        ("da_type", c_uint16),
        ("da_shard", c_uint16),
        ("da_flags", c_uint32),
        ("da_sub_anchors", c_uint64),
        ("da_buf", c_char * DAOS_ANCHOR_BUF_MAX),
    ]


class dirent(Structure):
    _fields_ = [
        ("d_ino", c_uint64),
        ("d_off", c_int64),
        ("d_reclen", c_uint16),
        ("d_type", ctypes.c_ubyte),
        ("d_name", c_char * 256),
    ]


class struct_stat(Structure):
    """struct stat (Linux x86_64 layout)."""

    _fields_ = [
        ("st_dev", c_uint64),
        ("st_ino", c_uint64),
        ("st_nlink", c_uint64),
        ("st_mode", c_uint32),
        ("st_uid", c_uint32),
        ("st_gid", c_uint32),
        ("__pad0", c_uint32),
        ("st_rdev", c_uint64),
        ("st_size", c_int64),
        ("st_blksize", c_int64),
        ("st_blocks", c_int64),
        ("st_atime_sec", c_int64),
        ("st_atime_nsec", c_int64),
        ("st_mtime_sec", c_int64),
        ("st_mtime_nsec", c_int64),
        ("st_ctime_sec", c_int64),
        ("st_ctime_nsec", c_int64),
        ("__reserved", c_int64 * 3),
    ]


_STAT_LAYOUT_X86_64: dict[str, int] = {
    "st_dev": 0,
    "st_ino": 8,
    "st_nlink": 16,
    "st_mode": 24,
    "st_uid": 28,
    "st_gid": 32,
    "st_size": 48,
}
_STAT_SIZE_X86_64 = 144

_actual = sizeof(struct_stat)
_drift = [
    f"{name}@{getattr(struct_stat, name).offset} (expected {offset})"
    for name, offset in _STAT_LAYOUT_X86_64.items()
    if getattr(struct_stat, name).offset != offset
]
if _actual != _STAT_SIZE_X86_64 or _drift:
    raise OSError(
        "struct_stat no longer matches glibc's x86_64 struct stat: "
        f"size {_actual} (expected {_STAT_SIZE_X86_64})"
        + (f", fields {', '.join(_drift)}" if _drift else "")
        + ". This is a daos-fsspec bug; the declaration was edited without "
        "updating the recorded layout."
    )

# Architectures whose glibc struct stat is laid out differently from the above.
# Importing would otherwise succeed and every stat would be silently wrong.
_INCOMPATIBLE_STAT_ARCHES = frozenset({"aarch64", "arm64", "armv7l", "riscv64"})
_arch = _platform.machine()

if _arch in _INCOMPATIBLE_STAT_ARCHES:
    raise OSError(
        f"daos-fsspec cannot run on {_arch}: its C struct stat uses glibc's "
        "generic layout (128 bytes, st_mode and st_nlink swapped relative to "
        "x86_64), and the ctypes declaration here mirrors the x86_64 layout. "
        "Every stat would be silently wrong rather than raise. Porting means "
        "selecting the struct layout per architecture and re-running "
        "docker/abi_probe.py there."
    )
if _arch != "x86_64":
    _logger.warning(
        "Architecture %s is unverified: struct_stat mirrors glibc's x86_64 "
        "layout and nothing here has checked that %s matches. If stat results "
        "look wrong, that is the first thing to suspect.",
        _arch,
        _arch,
    )


# uuid_t is 16 bytes
uuid_t = c_char * 16


# daos_pool_cont_info for pool_list_cont
# DAOS_PROP_LABEL_MAX_LEN = 127, pci_label is 128 bytes (127 + NUL)
DAOS_PROP_LABEL_MAX_LEN = 127


class daos_pool_cont_info(Structure):
    _fields_ = [("pci_uuid", uuid_t), ("pci_label", c_char * (DAOS_PROP_LABEL_MAX_LEN + 1))]


mode_t = c_uint32
dfs_obj_p = c_void_p
dfs_p = c_void_p
daos_size_t = c_uint64


# dfs_attr_t (DFS mount attributes: chunk_size, oclass, flags)
# Layout verified field-by-field with offsetof() against the DAOS 2.6.4 headers
# (daos_fs.h); sizeof(dfs_attr_t) == 176 on x86_64.
#   uint64_t         da_id;             // off 0
#   daos_size_t      da_chunk_size;     // off 8
#   daos_oclass_id_t da_oclass_id;      // off 16
#   daos_prop_t     *da_props;          // off 24
#   uint32_t         da_mode;           // off 32
#   daos_oclass_id_t da_dir_oclass_id;  // off 36
#   daos_oclass_id_t da_file_oclass_id; // off 40
#   char             da_hints[DAOS_CONT_HINT_MAX_LEN]; // off 44, LEN = 128
#
# daos_oclass_id_t is uint32 (NOT uint16: see daos_obj_class.h), daos_size_t is
# uint64, daos_prop_t* is a pointer.
class dfs_attr_t(Structure):
    _fields_ = [
        ("da_id", c_uint64),  # off 0
        ("da_chunk_size", daos_size_t),  # off 8
        ("da_oclass_id", c_uint32),  # off 16
        # 4 bytes padding to align da_props to 8 bytes
        ("da_props", c_void_p),  # off 24
        ("da_mode", c_uint32),  # off 32
        ("da_dir_oclass_id", c_uint32),  # off 36
        ("da_file_oclass_id", c_uint32),  # off 40
        ("da_hints", c_char * 128),  # off 44, DAOS_CONT_HINT_MAX_LEN = 128
    ]


# dfs_obj_info_t (queried per-object: oclass, chunk_size, OID)
# Layout verified field-by-field with offsetof() against the DAOS 2.6.4 headers
# (daos_fs.h); sizeof(dfs_obj_info_t) == 40 on x86_64.
#   daos_oclass_id_t doi_oclass_id;       // off 0
#   daos_size_t      doi_chunk_size;      // off 8
#   daos_oclass_id_t doi_dir_oclass_id;   // off 16
#   daos_oclass_id_t doi_file_oclass_id;  // off 20
#   daos_obj_id_t    doi_oid;             // off 24; struct { uint64 lo; uint64 hi; }
#
# daos_oclass_id_t is uint32 (NOT uint16), and daos_obj_id_t orders lo BEFORE hi.
class dfs_obj_info_t(Structure):
    _fields_ = [
        ("doi_oclass_id", c_uint32),  # off 0
        # 4 bytes padding to align doi_chunk_size to 8 bytes
        ("doi_chunk_size", daos_size_t),  # off 8
        ("doi_dir_oclass_id", c_uint32),  # off 16
        ("doi_file_oclass_id", c_uint32),  # off 20
        # doi_oid is daos_obj_id_t { uint64 lo; uint64 hi; } starting at off 24
        ("doi_oid_lo", c_uint64),  # off 24
        ("doi_oid_hi", c_uint64),  # off 32
    ]


@dataclass
class ObjInfo:
    """Object info from dfs_obj_get_info."""

    oclass_id: int
    dir_oclass_id: int
    file_oclass_id: int
    chunk_size: int
    oid_hi: int
    oid_lo: int


# ---------------------------------------------------------------------------
# DAOS flags and constants
# ---------------------------------------------------------------------------

DAOS_PC_RO = 1 << 0
DAOS_PC_RW = 1 << 1
DAOS_COO_RO = 1 << 0
DAOS_COO_RW = 1 << 1

# dfs_osetattr flags
DFS_SET_ATTR_MODE = 1 << 0
DFS_SET_ATTR_ATIME = 1 << 1
DFS_SET_ATTR_MTIME = 1 << 2
DFS_SET_ATTR_SIZE = 1 << 3
DFS_SET_ATTR_UID = 1 << 4
DFS_SET_ATTR_GID = 1 << 5

# xattr flags
XATTR_CREATE = 1
XATTR_REPLACE = 2

# DFS xattr limits (from daos_fs.h)
DFS_MAX_XATTR_NAME = 255
DFS_MAX_XATTR_LEN = 65536

# Chunk I/O size for large reads/writes
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB
_CHUNK_SIZE = CHUNK_SIZE  # backward-compatible alias

# Feature availability flags (set by _setup_signatures)
_HAS_DFS_CONNECT = False
_HAS_SNAP = False
_HAS_LISTIO = False
_HAS_PUNCH = False
_HAS_SET_PREFIX = False
_HAS_SUGGEST_OCLASS = False
_HAS_DUP = False
_HAS_CONT_CHECK = False
_HAS_DIR_ANCHOR_SET = False
_HAS_OBJ_ANCHOR_SET = False
_HAS_ITERATE = False


# ---------------------------------------------------------------------------
# Function signature setup
# ---------------------------------------------------------------------------


def _setup_signatures(libdaos: ctypes.CDLL, libdfs: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety and correct calling."""

    global _HAS_DFS_CONNECT, _HAS_SNAP, _HAS_LISTIO, _HAS_PUNCH
    global _HAS_SET_PREFIX, _HAS_SUGGEST_OCLASS, _HAS_DUP, _HAS_CONT_CHECK
    global _HAS_DIR_ANCHOR_SET, _HAS_OBJ_ANCHOR_SET, _HAS_ITERATE

    # daos_init / daos_fini
    libdaos.daos_init.restype = c_int
    libdaos.daos_init.argtypes = []
    libdaos.daos_fini.restype = c_int
    libdaos.daos_fini.argtypes = []

    # Pool connect/disconnect
    libdaos.daos_pool_connect.restype = c_int
    libdaos.daos_pool_connect.argtypes = [
        c_char_p,
        c_char_p,
        c_uint,
        POINTER(daos_handle_t),
        c_void_p,
        c_void_p,
    ]
    libdaos.daos_pool_disconnect.restype = c_int
    libdaos.daos_pool_disconnect.argtypes = [daos_handle_t, c_void_p]

    # Container open/close
    libdaos.daos_cont_open.restype = c_int
    libdaos.daos_cont_open.argtypes = [
        daos_handle_t,
        c_char_p,
        c_uint,
        POINTER(daos_handle_t),
        c_void_p,
        c_void_p,
    ]
    libdaos.daos_cont_close.restype = c_int
    libdaos.daos_cont_close.argtypes = [daos_handle_t, c_void_p]

    # Container create (poh, *uuid, *prop, *ev)
    libdaos.daos_cont_create.restype = c_int
    libdaos.daos_cont_create.argtypes = [
        daos_handle_t,
        POINTER(uuid_t),
        c_void_p,
        c_void_p,
    ]

    # Container create with label (poh, label, *prop, *uuid, *ev)
    try:
        libdaos.daos_cont_create_with_label.restype = c_int
        libdaos.daos_cont_create_with_label.argtypes = [
            daos_handle_t,
            c_char_p,
            c_void_p,
            POINTER(uuid_t),
            c_void_p,
        ]
    except AttributeError:
        pass  # Older DAOS versions

    # Container destroy (poh, cont_label, force, *ev)
    libdaos.daos_cont_destroy.restype = c_int
    libdaos.daos_cont_destroy.argtypes = [
        daos_handle_t,
        c_char_p,
        c_int,
        c_void_p,
    ]

    # Pool list containers (poh, *ncont, *cbuf, *ev)
    libdaos.daos_pool_list_cont.restype = c_int
    libdaos.daos_pool_list_cont.argtypes = [
        daos_handle_t,
        POINTER(daos_size_t),
        POINTER(daos_pool_cont_info),
        c_void_p,
    ]

    # DFS mount/umount
    libdfs.dfs_mount.restype = c_int
    libdfs.dfs_mount.argtypes = [daos_handle_t, daos_handle_t, c_int, POINTER(dfs_p)]
    libdfs.dfs_umount.restype = c_int
    libdfs.dfs_umount.argtypes = [dfs_p]

    # DFS lookup
    libdfs.dfs_lookup.restype = c_int
    libdfs.dfs_lookup.argtypes = [
        dfs_p,
        c_char_p,
        c_int,
        POINTER(dfs_obj_p),
        POINTER(mode_t),
        POINTER(struct_stat),
    ]
    libdfs.dfs_lookup_rel.restype = c_int
    libdfs.dfs_lookup_rel.argtypes = [
        dfs_p,
        dfs_obj_p,
        c_char_p,
        c_int,
        POINTER(dfs_obj_p),
        POINTER(mode_t),
        POINTER(struct_stat),
    ]

    # DFS release
    libdfs.dfs_release.restype = c_int
    libdfs.dfs_release.argtypes = [dfs_obj_p]

    # DFS open (also used for symlink creation with value param)
    libdfs.dfs_open.restype = c_int
    libdfs.dfs_open.argtypes = [
        dfs_p,
        dfs_obj_p,
        c_char_p,
        mode_t,
        c_int,
        c_uint32,  # daos_oclass_id_t cid: uint32, not uint16
        daos_size_t,
        c_char_p,
        POINTER(dfs_obj_p),
    ]

    # DFS read/write
    libdfs.dfs_read.restype = c_int
    libdfs.dfs_read.argtypes = [
        dfs_p,
        dfs_obj_p,
        POINTER(d_sg_list_t),
        daos_size_t,
        POINTER(daos_size_t),
        c_void_p,
    ]
    libdfs.dfs_write.restype = c_int
    libdfs.dfs_write.argtypes = [
        dfs_p,
        dfs_obj_p,
        POINTER(d_sg_list_t),
        daos_size_t,
        c_void_p,
    ]

    # DFS readdirplus (batch stat)
    libdfs.dfs_readdirplus.restype = c_int
    libdfs.dfs_readdirplus.argtypes = [
        dfs_p,
        dfs_obj_p,
        POINTER(daos_anchor_t),
        POINTER(c_uint32),
        POINTER(dirent),
        POINTER(struct_stat),
    ]

    # DFS readdir (names only, no batch stat). Needed as the fallback when
    # readdirplus aborts because an entry vanished mid-scan: see
    # dfs_readdirplus() for why that happens and why this is immune.
    libdfs.dfs_readdir.restype = c_int
    libdfs.dfs_readdir.argtypes = [
        dfs_p,
        dfs_obj_p,
        POINTER(daos_anchor_t),
        POINTER(c_uint32),
        POINTER(dirent),
    ]

    # DFS stat
    libdfs.dfs_ostat.restype = c_int
    libdfs.dfs_ostat.argtypes = [dfs_p, dfs_obj_p, POINTER(struct_stat)]

    # DFS stat by name, relative to an open parent (dfs_ostat needs an open
    # handle; this avoids opening one per entry in the fallback path).
    libdfs.dfs_stat.restype = c_int
    libdfs.dfs_stat.argtypes = [dfs_p, dfs_obj_p, c_char_p, POINTER(struct_stat)]

    # DFS setattr (for touch)
    libdfs.dfs_osetattr.restype = c_int
    libdfs.dfs_osetattr.argtypes = [dfs_p, dfs_obj_p, POINTER(struct_stat), c_int]

    # DFS get_symlink_value
    libdfs.dfs_get_symlink_value.restype = c_int
    libdfs.dfs_get_symlink_value.argtypes = [dfs_obj_p, c_char_p, POINTER(daos_size_t)]

    # DFS mkdir
    libdfs.dfs_mkdir.restype = c_int
    libdfs.dfs_mkdir.argtypes = [dfs_p, dfs_obj_p, c_char_p, mode_t, c_uint32]  # cid is uint32

    # DFS remove
    libdfs.dfs_remove.restype = c_int
    libdfs.dfs_remove.argtypes = [dfs_p, dfs_obj_p, c_char_p, c_bool, c_void_p]

    # DFS move
    libdfs.dfs_move.restype = c_int
    libdfs.dfs_move.argtypes = [dfs_p, dfs_obj_p, c_char_p, dfs_obj_p, c_char_p, c_void_p]

    # DFS exchange (atomic swap of two entries)
    libdfs.dfs_exchange.restype = c_int
    libdfs.dfs_exchange.argtypes = [dfs_p, dfs_obj_p, c_char_p, dfs_obj_p, c_char_p]

    # DFS chmod / chown / access
    libdfs.dfs_chmod.restype = c_int
    libdfs.dfs_chmod.argtypes = [dfs_p, dfs_obj_p, c_char_p, mode_t]
    libdfs.dfs_chown.restype = c_int
    libdfs.dfs_chown.argtypes = [dfs_p, dfs_obj_p, c_char_p, c_uint32, c_uint32, c_int]
    libdfs.dfs_access.restype = c_int
    libdfs.dfs_access.argtypes = [dfs_p, dfs_obj_p, c_char_p, c_int]

    # DFS get_mode
    libdfs.dfs_get_mode.restype = c_int
    libdfs.dfs_get_mode.argtypes = [dfs_obj_p, POINTER(mode_t)]

    # DFS get_chunk_size
    libdfs.dfs_get_chunk_size.restype = c_int
    libdfs.dfs_get_chunk_size.argtypes = [dfs_p, dfs_obj_p, POINTER(daos_size_t)]

    # DFS get_size
    libdfs.dfs_get_size.restype = c_int
    libdfs.dfs_get_size.argtypes = [dfs_p, dfs_obj_p, POINTER(daos_size_t)]

    # DFS xattr operations
    libdfs.dfs_setxattr.restype = c_int
    libdfs.dfs_setxattr.argtypes = [dfs_p, dfs_obj_p, c_char_p, c_void_p, daos_size_t, c_int]
    libdfs.dfs_getxattr.restype = c_int
    libdfs.dfs_getxattr.argtypes = [dfs_p, dfs_obj_p, c_char_p, c_void_p, POINTER(daos_size_t)]
    libdfs.dfs_listxattr.restype = c_int
    libdfs.dfs_listxattr.argtypes = [dfs_p, dfs_obj_p, c_char_p, POINTER(daos_size_t)]
    libdfs.dfs_removexattr.restype = c_int
    libdfs.dfs_removexattr.argtypes = [dfs_p, dfs_obj_p, c_char_p]

    # DFS query (mount attributes)
    libdfs.dfs_query.restype = c_int
    libdfs.dfs_query.argtypes = [dfs_p, c_void_p]  # dfs_attr_t* (opaque for now)

    # DFS obj_get_info (object class, chunk size, OID)
    libdfs.dfs_obj_get_info.restype = c_int
    libdfs.dfs_obj_get_info.argtypes = [dfs_p, dfs_obj_p, c_void_p]  # dfs_obj_info_t*

    # DFS obj_anchor_split (parallel directory iteration)
    libdfs.dfs_obj_anchor_split.restype = c_int
    libdfs.dfs_obj_anchor_split.argtypes = [dfs_obj_p, POINTER(c_uint32), POINTER(daos_anchor_t)]

    # DFS pool_get / pool_put (refcounted pool handle extraction)
    libdfs.dfs_pool_get.restype = c_int
    libdfs.dfs_pool_get.argtypes = [dfs_p, POINTER(daos_handle_t)]
    libdfs.dfs_pool_put.restype = c_int
    libdfs.dfs_pool_put.argtypes = [dfs_p, daos_handle_t]

    # daos_pool_query (pool space usage)
    libdaos.daos_pool_query.restype = c_int
    libdaos.daos_pool_query.argtypes = [
        daos_handle_t,
        POINTER(c_uint32),
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    # DFS connect/disconnect (high-level API)
    try:
        libdfs.dfs_init.restype = c_int
        libdfs.dfs_init.argtypes = []
        libdfs.dfs_fini.restype = c_int
        libdfs.dfs_fini.argtypes = []
        libdfs.dfs_connect.restype = c_int
        libdfs.dfs_connect.argtypes = [
            c_char_p,  # pool
            c_char_p,  # sys (NULL for default)
            c_char_p,  # container
            c_int,  # flags
            c_void_p,  # dfs_attr_t*
            POINTER(dfs_p),  # *dfs
        ]
        libdfs.dfs_disconnect.restype = c_int
        libdfs.dfs_disconnect.argtypes = [dfs_p]
        _HAS_DFS_CONNECT = True
    except AttributeError:
        _HAS_DFS_CONNECT = False  # Older DAOS

    # DFS connect_snap / mount_snap (snapshot mounts)
    try:
        libdfs.dfs_connect_snap.restype = c_int
        libdfs.dfs_connect_snap.argtypes = [
            c_char_p,
            c_char_p,
            c_char_p,
            c_int,
            c_void_p,
            c_uint64,  # epoch
            c_char_p,  # snap name
            POINTER(dfs_p),
        ]
        libdfs.dfs_mount_snap.restype = c_int
        libdfs.dfs_mount_snap.argtypes = [
            daos_handle_t,
            daos_handle_t,
            c_int,
            c_uint64,  # epoch
            c_char_p,  # snap name
            POINTER(dfs_p),
        ]
        _HAS_SNAP = True
    except AttributeError:
        _HAS_SNAP = False

    # DFS readx/writex (non-contiguous / list-IO)
    try:
        libdfs.dfs_readx.restype = c_int
        libdfs.dfs_readx.argtypes = [
            dfs_p,
            dfs_obj_p,
            c_void_p,
            POINTER(d_sg_list_t),
            POINTER(daos_size_t),
            c_void_p,
        ]
        libdfs.dfs_writex.restype = c_int
        libdfs.dfs_writex.argtypes = [dfs_p, dfs_obj_p, c_void_p, POINTER(d_sg_list_t), c_void_p]
        _HAS_LISTIO = True
    except AttributeError:
        _HAS_LISTIO = False

    # DFS punch (truncate)
    try:
        libdfs.dfs_punch.restype = c_int
        libdfs.dfs_punch.argtypes = [dfs_p, dfs_obj_p, daos_size_t, daos_size_t]
        _HAS_PUNCH = True
    except AttributeError:
        _HAS_PUNCH = False

    # DFS set_prefix
    try:
        libdfs.dfs_set_prefix.restype = c_int
        libdfs.dfs_set_prefix.argtypes = [dfs_p, c_char_p]
        _HAS_SET_PREFIX = True
    except AttributeError:
        _HAS_SET_PREFIX = False

    # DFS suggest_oclass
    try:
        libdfs.dfs_suggest_oclass.restype = c_int
        libdfs.dfs_suggest_oclass.argtypes = [dfs_p, c_char_p, POINTER(c_uint32)]
        _HAS_SUGGEST_OCLASS = True
    except AttributeError:
        _HAS_SUGGEST_OCLASS = False

    # DFS dup
    try:
        libdfs.dfs_dup.restype = c_int
        libdfs.dfs_dup.argtypes = [dfs_p, dfs_obj_p, c_int, POINTER(dfs_obj_p)]
        _HAS_DUP = True
    except AttributeError:
        _HAS_DUP = False

    # DFS cont_check / cont_scan
    try:
        libdfs.dfs_cont_check.restype = c_int
        libdfs.dfs_cont_check.argtypes = [daos_handle_t, c_char_p, c_uint64, c_char_p]
        libdfs.dfs_cont_scan.restype = c_int
        libdfs.dfs_cont_scan.argtypes = [daos_handle_t, c_char_p, c_uint64, c_char_p]
        _HAS_CONT_CHECK = True
    except AttributeError:
        _HAS_CONT_CHECK = False

    # DFS dir_anchor_set (set anchor to a specific entry name)
    try:
        libdfs.dfs_dir_anchor_set.restype = c_int
        libdfs.dfs_dir_anchor_set.argtypes = [dfs_obj_p, c_char_p, POINTER(daos_anchor_t)]
        _HAS_DIR_ANCHOR_SET = True
    except AttributeError:
        _HAS_DIR_ANCHOR_SET = False

    # DFS obj_anchor_set (set anchor by index from a split)
    try:
        libdfs.dfs_obj_anchor_set.restype = c_int
        libdfs.dfs_obj_anchor_set.argtypes = [dfs_obj_p, c_uint32, POINTER(daos_anchor_t)]
        _HAS_OBJ_ANCHOR_SET = True
    except AttributeError:
        _HAS_OBJ_ANCHOR_SET = False

    # DFS iterate (readdir with buffer size limit + per-entry callback)
    try:
        libdfs.dfs_iterate.restype = c_int
        libdfs.dfs_iterate.argtypes = [
            dfs_p,
            dfs_obj_p,
            POINTER(daos_anchor_t),
            POINTER(c_uint32),
            c_size_t,
            c_void_p,
            c_void_p,
        ]
        _HAS_ITERATE = True
    except AttributeError:
        _HAS_ITERATE = False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

# DAOS error codes (negative = DAOS-specific, positive = POSIX errno)
DER_SUCCESS = 0
DER_NO_PERM = -1001
DER_NO_HDL = -1002
DER_INVAL = -1003
DER_EXIST = -1004
DER_NONEXIST = -1005
DER_UNREACH = -1006
DER_NOSPACE = -1007
DER_ALREADY = -1010
DER_NOMEM = -1011
DER_TIMEDOUT = -1012
DER_NOSYS = -1018
DER_NO_ENTRY = -1039
DER_BUSY = -1013
DER_HG = -1020

# Errors that are safe to retry (transient network/system issues)
RETRYABLE_ERRORS: frozenset[int] = frozenset(
    {
        DER_UNREACH,
        DER_HG,
        DER_BUSY,
        DER_TIMEDOUT,
        errno.ETIMEDOUT,
        errno.EAGAIN,
        errno.EINTR,
    }
)

_DER_NAMES: dict[int, str] = {
    DER_SUCCESS: "DER_SUCCESS",
    DER_NO_PERM: "DER_NO_PERM (no permission)",
    DER_NO_HDL: "DER_NO_HDL (invalid handle)",
    DER_INVAL: "DER_INVAL (invalid parameter)",
    DER_EXIST: "DER_EXIST (already exists)",
    DER_NONEXIST: "DER_NONEXIST (does not exist)",
    DER_UNREACH: "DER_UNREACH (unreachable)",
    DER_NOSPACE: "DER_NOSPACE (no space)",
    DER_ALREADY: "DER_ALREADY (already initialized)",
    DER_NOMEM: "DER_NOMEM (out of memory)",
    DER_NOSYS: "DER_NOSYS (not supported)",
    DER_NO_ENTRY: "DER_NO_ENTRY (no entry)",
    DER_BUSY: "DER_BUSY (busy)",
    DER_TIMEDOUT: "DER_TIMEDOUT (timed out)",
    DER_HG: "DER_HG (Mercury transport error)",
}

# Map DAOS/errno codes to Python exceptions
_ERRNO_NAMES: dict[int, str] = {
    errno.ENOENT: "ENOENT (no such file or directory)",
    errno.EEXIST: "EEXIST (file exists)",
    errno.EACCES: "EACCES (permission denied)",
    errno.ENOTDIR: "ENOTDIR (not a directory)",
    errno.EISDIR: "EISDIR (is a directory)",
    errno.ENOTEMPTY: "ENOTEMPTY (directory not empty)",
    errno.EINVAL: "EINVAL (invalid argument)",
    errno.ENOMEM: "ENOMEM (out of memory)",
    errno.ENOSPC: "ENOSPC (no space left)",
    errno.ENOSYS: "ENOSYS (not implemented)",
}


def _error_name(rc: int) -> str:
    """Human-readable name for a DAOS/errno return code."""
    if rc in _DER_NAMES:
        return _DER_NAMES[rc]
    if rc in _ERRNO_NAMES:
        return _ERRNO_NAMES[rc]
    if rc > 0:
        try:
            return os.strerror(rc)
        except ValueError:
            pass
    return f"error code {rc}"


class DaosError(Exception):
    """Error from a DAOS/DFS C library call."""

    def __init__(self, rc: int, msg: str = ""):
        self.rc = rc
        name = _error_name(rc)
        detail = f"{msg}: {name}" if msg else name
        super().__init__(detail)

    @property
    def is_retryable(self) -> bool:
        """Return True if this error is transient and the operation can be retried."""
        return self.rc in RETRYABLE_ERRORS


def _check(rc: int, msg: str = "") -> None:
    """Raise appropriate Python exception for non-zero DAOS return code."""
    if rc == 0:
        return
    # Map to natural Python exceptions
    if rc == errno.ENOENT or rc == DER_NONEXIST or rc == DER_NO_ENTRY:
        raise FileNotFoundError(_error_detail(rc, msg))
    if rc == errno.EEXIST or rc == DER_EXIST:
        raise FileExistsError(_error_detail(rc, msg))
    if rc == errno.EACCES or rc == DER_NO_PERM:
        raise PermissionError(_error_detail(rc, msg))
    if rc == errno.ENOTDIR:
        raise NotADirectoryError(_error_detail(rc, msg))
    if rc == errno.EISDIR:
        raise IsADirectoryError(_error_detail(rc, msg))
    if rc == errno.ENOTEMPTY:
        raise OSError(errno.ENOTEMPTY, _error_detail(rc, msg))
    raise DaosError(rc, msg)


def _error_detail(rc: int, msg: str) -> str:
    name = _error_name(rc)
    return f"{msg}: {name}" if msg else name


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StatResult:
    """Parsed result of dfs_ostat / dfs_lookup."""

    size: int
    mode: int
    uid: int
    gid: int
    nlink: int
    mtime: float
    atime: float
    ctime: float

    @property
    def is_dir(self) -> bool:
        return stat_mod.S_ISDIR(self.mode)

    @property
    def is_file(self) -> bool:
        return stat_mod.S_ISREG(self.mode)

    @property
    def is_link(self) -> bool:
        return stat_mod.S_ISLNK(self.mode)

    @property
    def type_name(self) -> str:
        if self.is_dir:
            return "directory"
        if self.is_link:
            return "other"  # fsspec convention
        return "file"


def _parse_stat(st: struct_stat) -> StatResult:
    return StatResult(
        size=st.st_size,
        mode=st.st_mode,
        uid=st.st_uid,
        gid=st.st_gid,
        nlink=st.st_nlink,
        mtime=st.st_mtime_sec + st.st_mtime_nsec / 1e9,
        atime=st.st_atime_sec + st.st_atime_nsec / 1e9,
        ctime=st.st_ctime_sec + st.st_ctime_nsec / 1e9,
    )


@dataclass
class DirEntry:
    name: str
    stat: StatResult | None


@dataclass
class ContainerInfo:
    uuid: str
    label: str


# ---------------------------------------------------------------------------
# Process-level DAOS initialization
# ---------------------------------------------------------------------------

_runtime = _Runtime()


def daos_init() -> None:
    """Own one DAOS initialization reference for this process."""
    _runtime.initialize()


def _dfs_init() -> None:
    """Own one DFS reference, independent of connection/reconnect count."""
    _runtime.initialize(dfs=True)


def daos_fini() -> None:
    """Finalize our references after work has stopped and resources are closed."""
    _runtime.finish()


# ---------------------------------------------------------------------------
# Pool / container connection
# ---------------------------------------------------------------------------


def pool_connect(pool_label: str, flags: int = DAOS_PC_RW) -> daos_handle_t:
    libdaos, _ = _ensure_libs()
    poh = daos_handle_t()
    rc = libdaos.daos_pool_connect(pool_label.encode(), None, flags, byref(poh), None, None)
    _check(rc, f"pool_connect({pool_label})")
    return poh


def pool_disconnect(poh: daos_handle_t) -> None:
    libdaos, _ = _ensure_libs()
    rc = libdaos.daos_pool_disconnect(poh, None)
    _check(rc, "pool_disconnect")


def cont_open(poh: daos_handle_t, cont_label: str, flags: int = DAOS_COO_RW) -> daos_handle_t:
    libdaos, _ = _ensure_libs()
    coh = daos_handle_t()
    rc = libdaos.daos_cont_open(poh, cont_label.encode(), flags, byref(coh), None, None)
    _check(rc, f"cont_open({cont_label})")
    return coh


def cont_close(coh: daos_handle_t) -> None:
    libdaos, _ = _ensure_libs()
    rc = libdaos.daos_cont_close(coh, None)
    _check(rc, "cont_close")


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------


def cont_create(poh: daos_handle_t, label: str) -> str:
    """Create a POSIX container with the given label. Returns UUID string.

    Requires DAOS 2.6+ (daos_cont_create_with_label).
    """
    libdaos, _ = _ensure_libs()
    cont_uuid = uuid_t()
    if not hasattr(libdaos, "daos_cont_create_with_label"):
        raise DaosError(
            0,
            "daos_cont_create_with_label not found in libdaos. "
            "DAOS 2.6+ is required for labeled container creation.",
        )
    rc = libdaos.daos_cont_create_with_label(poh, label.encode(), None, byref(cont_uuid), None)
    _check(rc, f"cont_create({label})")
    return str(uuid.UUID(bytes=bytes(cont_uuid)))


def cont_destroy(poh: daos_handle_t, label: str, force: bool = False) -> None:
    """Destroy a container by label."""
    libdaos, _ = _ensure_libs()
    rc = libdaos.daos_cont_destroy(poh, label.encode(), 1 if force else 0, None)
    _check(rc, f"cont_destroy({label})")


def pool_list_cont(poh: daos_handle_t) -> list[ContainerInfo]:
    """List all containers in a pool."""
    libdaos, _ = _ensure_libs()

    # First call: get count
    ncont = daos_size_t(0)
    rc = libdaos.daos_pool_list_cont(poh, byref(ncont), None, None)
    _check(rc, "pool_list_cont (count)")

    if ncont.value == 0:
        return []

    # Second call: get info
    buf = (daos_pool_cont_info * ncont.value)()
    rc = libdaos.daos_pool_list_cont(poh, byref(ncont), buf, None)
    _check(rc, "pool_list_cont (list)")

    result = []
    for i in range(ncont.value):
        try:
            u = str(uuid.UUID(bytes=bytes(buf[i].pci_uuid)))
        except Exception:
            u = ""
        label = buf[i].pci_label.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        result.append(ContainerInfo(uuid=u, label=label))
    return result


# ---------------------------------------------------------------------------
# DFS mount
# ---------------------------------------------------------------------------


def dfs_mount_fs(poh: daos_handle_t, coh: daos_handle_t, readonly: bool = False) -> dfs_p:
    _, libdfs = _ensure_libs()
    dfs = dfs_p()
    flags = os.O_RDONLY if readonly else os.O_RDWR
    rc = libdfs.dfs_mount(poh, coh, flags, byref(dfs))
    _check(rc, "dfs_mount")
    return dfs


def dfs_umount(dfs: dfs_p) -> None:
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_umount(dfs)
    _check(rc, "dfs_umount")


# ---------------------------------------------------------------------------
# Path operations
# ---------------------------------------------------------------------------


def dfs_lookup_path(dfs: dfs_p, path: str, flags: int = os.O_RDWR) -> tuple[dfs_obj_p, StatResult]:
    """Lookup a path and return (obj_handle, stat)."""
    _, libdfs = _ensure_libs()
    obj = dfs_obj_p()
    m = mode_t()
    stbuf = struct_stat()
    rc = libdfs.dfs_lookup(dfs, path.encode(), flags, byref(obj), byref(m), byref(stbuf))
    _check(rc, f"lookup({path})")
    return obj, _parse_stat(stbuf)


def dfs_lookup_rel_path(
    dfs: dfs_p, parent: dfs_obj_p, name: str, flags: int = os.O_RDWR
) -> tuple[dfs_obj_p, StatResult]:
    """Lookup relative to a parent directory."""
    _, libdfs = _ensure_libs()
    obj = dfs_obj_p()
    m = mode_t()
    stbuf = struct_stat()
    rc = libdfs.dfs_lookup_rel(
        dfs, parent, name.encode(), flags, byref(obj), byref(m), byref(stbuf)
    )
    _check(rc, f"lookup_rel({name})")
    return obj, _parse_stat(stbuf)


def same_handle(a: Any, b: Any) -> bool:
    """True if two handle wrappers refer to the same DFS object.

    ``dfs_obj_p`` is ``c_void_p``, and two c_void_p wrapping the *same*
    address compare unequal; ctypes defines no __eq__, so == falls back to
    object identity. Any check written as ``handle != other`` is therefore
    asking "is this the same Python wrapper", not "is this the same object in
    DAOS". That distinction decides whether a release is correct or a double
    free, so compare the addresses explicitly.

    Mocks have no usable ``.value``, so they fall back to identity, which is
    what the test suite wants.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False

    missing = object()
    addr_a = getattr(a, "value", missing)
    addr_b = getattr(b, "value", missing)
    if addr_a is missing or addr_b is missing:
        return False
    return bool(addr_a == addr_b)


def dfs_release(obj: dfs_obj_p) -> None:
    """Release a DFS object handle; report uncertain cleanup and prevent fini."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_release(obj)
    if rc != 0:
        _runtime.failure = f"dfs_release failed: {_error_name(rc)}; native cleanup is uncertain"
        _check(rc, "dfs_release")


def dfs_stat(dfs: dfs_p, obj: dfs_obj_p) -> StatResult:
    _, libdfs = _ensure_libs()
    stbuf = struct_stat()
    rc = libdfs.dfs_ostat(dfs, obj, byref(stbuf))
    _check(rc, "ostat")
    return _parse_stat(stbuf)


def dfs_setattr(
    dfs: dfs_p,
    obj: dfs_obj_p,
    mtime: float | None = None,
    atime: float | None = None,
    file_size: int | None = None,
) -> None:
    """Set file attributes (mtime, atime, size)."""
    _, libdfs = _ensure_libs()
    stbuf = struct_stat()
    ctypes.memset(byref(stbuf), 0, sizeof(stbuf))
    flags = 0
    if mtime is not None:
        stbuf.st_mtime_sec = int(mtime)
        stbuf.st_mtime_nsec = max(0, min(999_999_999, int((mtime - int(mtime)) * 1e9)))
        flags |= DFS_SET_ATTR_MTIME
    if atime is not None:
        stbuf.st_atime_sec = int(atime)
        stbuf.st_atime_nsec = max(0, min(999_999_999, int((atime - int(atime)) * 1e9)))
        flags |= DFS_SET_ATTR_ATIME
    if file_size is not None:
        stbuf.st_size = file_size
        flags |= DFS_SET_ATTR_SIZE
    if flags == 0:
        return
    rc = libdfs.dfs_osetattr(dfs, obj, byref(stbuf), flags)
    _check(rc, "osetattr")


# ---------------------------------------------------------------------------
# Symlink operations
# ---------------------------------------------------------------------------


def dfs_create_symlink(dfs: dfs_p, parent: dfs_obj_p, name: str, target: str) -> dfs_obj_p:
    """Create a symbolic link. Returns the symlink's dfs_obj handle."""
    _, libdfs = _ensure_libs()
    obj = dfs_obj_p()
    rc = libdfs.dfs_open(
        dfs,
        parent,
        name.encode(),
        mode_t(stat_mod.S_IFLNK | 0o777),
        os.O_CREAT | os.O_EXCL,
        0,
        daos_size_t(0),
        target.encode(),
        byref(obj),
    )
    _check(rc, f"symlink({name} -> {target})")
    return obj


def dfs_read_symlink(obj: dfs_obj_p) -> str:
    """Read the target of a symbolic link."""
    _, libdfs = _ensure_libs()
    # First call: get required size (including NUL)
    sz = daos_size_t(0)
    rc = libdfs.dfs_get_symlink_value(obj, None, byref(sz))
    _check(rc, "get_symlink_value (size)")
    # Second call: read the actual value
    buf = ctypes.create_string_buffer(sz.value)
    rc = libdfs.dfs_get_symlink_value(obj, buf, byref(sz))
    _check(rc, "get_symlink_value")
    return buf.value.decode()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def dfs_open_file(
    dfs: dfs_p,
    parent: dfs_obj_p,
    name: str,
    mode: int,
    flags: int,
    oclass_id: int = 0,
    chunk_size: int = 0,
) -> dfs_obj_p:
    """Open or create a file. Returns dfs_obj handle.

    Parameters
    ----------
    oclass_id : int
        DAOS object class ID (0 = use directory/container default).
    chunk_size : int
        Stripe/chunk size in bytes (0 = use default 1MB).
    """
    _, libdfs = _ensure_libs()
    obj = dfs_obj_p()
    rc = libdfs.dfs_open(
        dfs,
        parent,
        name.encode(),
        mode_t(mode),
        flags,
        c_uint32(oclass_id),  # daos_oclass_id_t is uint32
        daos_size_t(chunk_size),
        None,
        byref(obj),
    )
    _check(rc, f"open({name})")
    return obj


def dfs_read(dfs: dfs_p, obj: dfs_obj_p, offset: int, size: int) -> bytes:
    """Read up to `size` bytes at `offset`, chunking large reads."""
    _, libdfs = _ensure_libs()
    if size <= _CHUNK_SIZE:
        return _dfs_read_single(libdfs, dfs, obj, offset, size)
    # Chunked read for large sizes
    parts = []
    remaining = size
    pos = offset
    while remaining > 0:
        chunk = min(remaining, _CHUNK_SIZE)
        data = _dfs_read_single(libdfs, dfs, obj, pos, chunk)
        if not data:
            # read_size == 0 is the only reliable EOF signal. A short
            # (but non-empty) read is NOT EOF: dfs_read may return fewer
            # bytes than requested at a chunk/stripe boundary, so we must
            # keep reading until we hit a genuinely empty read. Breaking on
            # a short read here would silently truncate valid data.
            break
        parts.append(data)
        pos += len(data)
        remaining -= len(data)
    return b"".join(parts)


def _dfs_read_single(
    libdfs: ctypes.CDLL, dfs: dfs_p, obj: dfs_obj_p, offset: int, size: int
) -> bytes:
    buf = ctypes.create_string_buffer(size)
    iov = d_iov_t(ctypes.cast(buf, c_void_p), size, size)
    sgl = d_sg_list_t(1, 0, ctypes.pointer(iov))
    read_size = daos_size_t(size)
    rc = libdfs.dfs_read(dfs, obj, byref(sgl), daos_size_t(offset), byref(read_size), None)
    _check(rc, f"read(off={offset}, sz={size})")
    return buf.raw[: read_size.value]


def dfs_write(dfs: dfs_p, obj: dfs_obj_p, offset: int, data: bytes) -> int:
    """Write `data` at `offset`, chunking large writes. Returns bytes written."""
    _, libdfs = _ensure_libs()
    total = len(data)
    if total <= _CHUNK_SIZE:
        _dfs_write_single(libdfs, dfs, obj, offset, data)
        return total
    written = 0
    while written < total:
        chunk = data[written : written + _CHUNK_SIZE]
        _dfs_write_single(libdfs, dfs, obj, offset + written, chunk)
        written += len(chunk)
    return written


def _dfs_write_single(
    libdfs: ctypes.CDLL, dfs: dfs_p, obj: dfs_obj_p, offset: int, data: bytes
) -> None:
    buf = ctypes.create_string_buffer(data, len(data))
    iov = d_iov_t(ctypes.cast(buf, c_void_p), len(data), len(data))
    sgl = d_sg_list_t(1, 0, ctypes.pointer(iov))
    rc = libdfs.dfs_write(dfs, obj, byref(sgl), daos_size_t(offset), None)
    _check(rc, f"write(off={offset}, sz={len(data)})")


def dfs_close(obj: dfs_obj_p) -> None:
    """Alias for dfs_release: close a DFS object handle."""
    dfs_release(obj)


# ---------------------------------------------------------------------------
# Directory operations
# ---------------------------------------------------------------------------

DAOS_ANCHOR_TYPE_ZERO = 0
DAOS_ANCHOR_TYPE_EOF = 3
_READDIR_BATCH = 128


def _anchor_is_eof(anchor: daos_anchor_t) -> bool:
    return anchor.da_type == DAOS_ANCHOR_TYPE_EOF


def _entry_name(d: dirent) -> str:
    return d.d_name.split(b"\x00", 1)[0].decode("utf-8", errors="surrogateescape")


def dfs_readdir(dfs: dfs_p, obj: dfs_obj_p) -> list[str]:
    """Read entry names from a directory, without stat info.

    Immune to the concurrent-unlink abort that affects readdirplus: names come
    from ``daos_obj_list_dkey`` alone, with no per-entry stat that can fail.
    """
    _, libdfs = _ensure_libs()
    names: list[str] = []
    anchor = daos_anchor_t()
    ctypes.memset(byref(anchor), 0, sizeof(anchor))
    dirs_array = (dirent * _READDIR_BATCH)()

    while not _anchor_is_eof(anchor):
        nr = c_uint32(_READDIR_BATCH)
        rc = libdfs.dfs_readdir(dfs, obj, byref(anchor), byref(nr), dirs_array)
        _check(rc, "readdir")
        names.extend(_entry_name(dirs_array[i]) for i in range(nr.value))
    return names


def dfs_stat_rel(dfs: dfs_p, parent: dfs_obj_p, name: str) -> StatResult | None:
    """Stat one entry by name. Returns None if it no longer exists."""
    _, libdfs = _ensure_libs()
    stbuf = struct_stat()
    rc = libdfs.dfs_stat(dfs, parent, name.encode(), byref(stbuf))
    if rc in (errno.ENOENT, -DER_NONEXIST, DER_NONEXIST, DER_NO_ENTRY):
        return None
    _check(rc, f"stat({name})")
    return _parse_stat(stbuf)


def dfs_readdirplus(dfs: dfs_p, obj: dfs_obj_p) -> list[DirEntry]:
    """Read all entries from a directory with stat info in one call per batch."""
    _, libdfs = _ensure_libs()
    entries: list[DirEntry] = []
    anchor = daos_anchor_t()
    ctypes.memset(byref(anchor), 0, sizeof(anchor))
    dirs_array = (dirent * _READDIR_BATCH)()
    stats_array = (struct_stat * _READDIR_BATCH)()

    while not _anchor_is_eof(anchor):
        nr = c_uint32(_READDIR_BATCH)
        rc = libdfs.dfs_readdirplus(dfs, obj, byref(anchor), byref(nr), dirs_array, stats_array)
        if rc in (errno.ENOENT, -DER_NONEXIST, DER_NONEXIST, DER_NO_ENTRY):
            return _readdir_tolerant(dfs, obj)
        _check(rc, "readdirplus")
        for i in range(nr.value):
            entries.append(
                DirEntry(name=_entry_name(dirs_array[i]), stat=_parse_stat(stats_array[i]))
            )
    return entries


def _readdir_tolerant(dfs: dfs_p, obj: dfs_obj_p) -> list[DirEntry]:
    """List a directory that is being mutated concurrently."""
    entries: list[DirEntry] = []
    for name in dfs_readdir(dfs, obj):
        st = dfs_stat_rel(dfs, obj, name)
        if st is None:
            continue  # unlinked while we scanned; omit rather than fail
        entries.append(DirEntry(name=name, stat=st))
    return entries


def dfs_mkdir(
    dfs: dfs_p, parent: dfs_obj_p, name: str, mode: int = 0o755, oclass_id: int = 0
) -> None:
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_mkdir(dfs, parent, name.encode(), mode_t(mode), c_uint32(oclass_id))
    _check(rc, f"mkdir({name})")


def dfs_remove(dfs: dfs_p, parent: dfs_obj_p, name: str, force: bool = False) -> None:
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_remove(dfs, parent, name.encode(), 1 if force else 0, None)
    _check(rc, f"remove({name})")


def dfs_move(
    dfs: dfs_p, src_parent: dfs_obj_p, src_name: str, dst_parent: dfs_obj_p, dst_name: str
) -> None:
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_move(dfs, src_parent, src_name.encode(), dst_parent, dst_name.encode(), None)
    _check(rc, f"move({src_name} -> {dst_name})")


def dfs_exchange(
    dfs: dfs_p, parent1: dfs_obj_p, name1: str, parent2: dfs_obj_p, name2: str
) -> None:
    """Atomically exchange two directory entries."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_exchange(dfs, parent1, name1.encode(), parent2, name2.encode())
    _check(rc, f"exchange({name1} <-> {name2})")


# ---------------------------------------------------------------------------
# Extended attributes
# ---------------------------------------------------------------------------


def dfs_setxattr(dfs: dfs_p, obj: dfs_obj_p, name: str, value: bytes, flags: int = 0) -> None:
    """Set an extended attribute on a DFS object."""
    _, libdfs = _ensure_libs()
    if len(name) > DFS_MAX_XATTR_NAME:
        raise ValueError(f"xattr name too long ({len(name)} > {DFS_MAX_XATTR_NAME})")
    if len(value) > DFS_MAX_XATTR_LEN:
        raise ValueError(f"xattr value too long ({len(value)} > {DFS_MAX_XATTR_LEN})")
    buf = ctypes.create_string_buffer(value, len(value))
    rc = libdfs.dfs_setxattr(dfs, obj, name.encode(), buf, daos_size_t(len(value)), flags)
    _check(rc, f"setxattr({name})")


def dfs_getxattr(dfs: dfs_p, obj: dfs_obj_p, name: str) -> bytes:
    """Get an extended attribute value from a DFS object."""
    _, libdfs = _ensure_libs()
    # First call: get required size
    sz = daos_size_t(0)
    rc = libdfs.dfs_getxattr(dfs, obj, name.encode(), None, byref(sz))
    if rc != 0:
        _check(rc, f"getxattr({name})")
    if sz.value == 0:
        return b""
    # Second call: read the value
    buf = ctypes.create_string_buffer(sz.value)
    rc = libdfs.dfs_getxattr(dfs, obj, name.encode(), buf, byref(sz))
    _check(rc, f"getxattr({name})")
    return buf.raw[: sz.value]


def dfs_listxattr(dfs: dfs_p, obj: dfs_obj_p) -> list[str]:
    """List all extended attribute names on a DFS object."""
    _, libdfs = _ensure_libs()
    # First call: get required buffer size
    sz = daos_size_t(0)
    rc = libdfs.dfs_listxattr(dfs, obj, None, byref(sz))
    if rc != 0:
        _check(rc, "listxattr(size)")
    if sz.value == 0:
        return []
    # Second call: read the names (NUL-separated)
    buf = ctypes.create_string_buffer(sz.value)
    rc = libdfs.dfs_listxattr(dfs, obj, buf, byref(sz))
    _check(rc, "listxattr")
    # Parse NUL-separated list
    raw = buf.raw[: sz.value]
    if not raw:
        return []
    return [name.decode("utf-8", errors="replace") for name in raw.split(b"\x00") if name]


def dfs_removexattr(dfs: dfs_p, obj: dfs_obj_p, name: str) -> None:
    """Remove an extended attribute from a DFS object."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_removexattr(dfs, obj, name.encode())
    _check(rc, f"removexattr({name})")


# ---------------------------------------------------------------------------
# Attribute operations (chmod, chown, access, get_size, get_chunk_size)
# ---------------------------------------------------------------------------


def dfs_chmod(dfs: dfs_p, parent: dfs_obj_p, name: str, mode: int) -> None:
    """Change permissions on a DFS entry."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_chmod(dfs, parent, name.encode(), mode_t(mode))
    _check(rc, f"chmod({name})")


def dfs_chown(dfs: dfs_p, parent: dfs_obj_p, name: str, uid: int, gid: int, flags: int = 0) -> None:
    """Change owner/group on a DFS entry."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_chown(dfs, parent, name.encode(), c_uint32(uid), c_uint32(gid), flags)
    _check(rc, f"chown({name})")


def dfs_access(dfs: dfs_p, parent: dfs_obj_p, name: str, mask: int) -> bool:
    """Check access permissions. Returns True if access is granted."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_access(dfs, parent, name.encode(), c_int(mask))
    if rc == 0:
        return True
    if rc == errno.EACCES or rc == DER_NO_PERM:
        return False
    _check(rc, f"access({name})")
    return True  # unreachable but satisfies type checker


def dfs_get_size(dfs: dfs_p, obj: dfs_obj_p) -> int:
    """Get the size of a DFS file without full stat."""
    _, libdfs = _ensure_libs()
    sz = daos_size_t(0)
    rc = libdfs.dfs_get_size(dfs, obj, byref(sz))
    _check(rc, "get_size")
    return sz.value


def dfs_get_chunk_size(dfs: dfs_p, obj: dfs_obj_p) -> int:
    """Get the chunk size of a DFS object."""
    _, libdfs = _ensure_libs()
    sz = daos_size_t(0)
    rc = libdfs.dfs_get_chunk_size(dfs, obj, byref(sz))
    _check(rc, "get_chunk_size")
    return sz.value


# ---------------------------------------------------------------------------
# Pool space query
# ---------------------------------------------------------------------------


@dataclass
class PoolSpace:
    """Pool storage space information."""

    scm_total: int
    scm_free: int
    nvme_total: int
    nvme_free: int


# Pool space / info structs. Layout verified with offsetof() against the DAOS
# 2.6.4 headers (daos_pool.h): sizeof(daos_pool_info_t) == 192, pi_bits @40,
# pi_space @48, pi_rebuild_st @136. The struct MUST be the full 192 bytes: DAOS
# writes the whole daos_pool_info_t through the pointer, so an undersized buffer
# corrupts the heap.
DAOS_MEDIA_SCM = 0
DAOS_MEDIA_NVMe = 1


DPI_SPACE = 1 << 0


class _daos_space(Structure):
    # struct daos_space { uint64_t s_total[2]; uint64_t s_free[2]; }
    _fields_ = [
        ("s_total", c_uint64 * 2),
        ("s_free", c_uint64 * 2),
    ]


class _daos_pool_space(Structure):
    # struct daos_pool_space: sizeof 88
    _fields_ = [
        ("ps_space", _daos_space),  # off 0  (s_total @0, s_free @16)
        ("ps_free_min", c_uint64 * 2),  # off 32
        ("ps_free_max", c_uint64 * 2),  # off 48
        ("ps_free_mean", c_uint64 * 2),  # off 64
        ("ps_ntargets", c_uint32),  # off 80
        ("ps_padding", c_uint32),  # off 84
    ]


class _daos_rebuild_status(Structure):
    # struct daos_rebuild_status: sizeof 56; opaque tail, reserved so the
    # overall daos_pool_info_t buffer is the full 192 bytes DAOS writes into.
    _fields_ = [("_reserved", c_char * 56)]


class _daos_pool_info(Structure):
    _fields_ = [
        ("pi_uuid", uuid_t),  # off 0
        ("pi_ntargets", c_uint32),  # off 16
        ("pi_nnodes", c_uint32),  # off 20
        ("pi_ndisabled", c_uint32),  # off 24
        ("pi_map_ver", c_uint32),  # off 28
        ("pi_leader", c_uint32),  # off 32
        # 4 bytes padding (ctypes auto-aligns pi_bits to 8)
        ("pi_bits", c_uint64),  # off 40
        ("pi_space", _daos_pool_space),  # off 48
        ("pi_rebuild_st", _daos_rebuild_status),  # off 136
    ]


def pool_query_space(poh: daos_handle_t) -> PoolSpace:
    """Query pool storage space usage.

    Returns a PoolSpace with SCM and NVMe total/free in bytes.
    """
    libdaos, _ = _ensure_libs()
    info = _daos_pool_info()
    # Request that DAOS populate pi_space; without DPI_SPACE it stays zeroed.
    info.pi_bits = DPI_SPACE
    rc = libdaos.daos_pool_query(poh, None, byref(info), None, None)
    _check(rc, "pool_query")
    space = info.pi_space.ps_space
    return PoolSpace(
        scm_total=space.s_total[DAOS_MEDIA_SCM],
        scm_free=space.s_free[DAOS_MEDIA_SCM],
        nvme_total=space.s_total[DAOS_MEDIA_NVMe],
        nvme_free=space.s_free[DAOS_MEDIA_NVMe],
    )


def dfs_pool_get(dfs: dfs_p) -> daos_handle_t:
    """Get the pool handle from a DFS mount (refcounted)."""
    _, libdfs = _ensure_libs()
    poh = daos_handle_t()
    rc = libdfs.dfs_pool_get(dfs, byref(poh))
    _check(rc, "dfs_pool_get")
    return poh


def dfs_pool_put(dfs: dfs_p, poh: daos_handle_t) -> None:
    """Release a pool handle obtained from dfs_pool_get."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_pool_put(dfs, poh)
    if rc != 0:
        _logger.warning("dfs_pool_put failed: %s", _error_name(rc))


# ---------------------------------------------------------------------------
# DFS query (mount attributes)
# ---------------------------------------------------------------------------


@dataclass
class MountAttr:
    """DFS mount attributes from dfs_query."""

    oclass_id: int
    dir_oclass_id: int
    file_oclass_id: int
    chunk_size: int
    mode: int


def dfs_query_attr(dfs: dfs_p) -> MountAttr:
    """Query the DFS mount attributes (default oclass, chunk_size, etc.)."""
    _, libdfs = _ensure_libs()
    attr = dfs_attr_t()
    rc = libdfs.dfs_query(dfs, byref(attr))
    _check(rc, "dfs_query")
    return MountAttr(
        oclass_id=attr.da_oclass_id,
        dir_oclass_id=attr.da_dir_oclass_id,
        file_oclass_id=attr.da_file_oclass_id,
        chunk_size=attr.da_chunk_size,
        mode=attr.da_mode,
    )


def dfs_obj_get_info(dfs: dfs_p, obj: dfs_obj_p) -> ObjInfo:
    """Query object info (oclass, chunk_size, OID) for an open DFS object."""
    _, libdfs = _ensure_libs()
    info = dfs_obj_info_t()
    rc = libdfs.dfs_obj_get_info(dfs, obj, byref(info))
    _check(rc, "dfs_obj_get_info")
    return ObjInfo(
        oclass_id=info.doi_oclass_id,
        dir_oclass_id=info.doi_dir_oclass_id,
        file_oclass_id=info.doi_file_oclass_id,
        chunk_size=info.doi_chunk_size,
        oid_hi=info.doi_oid_hi,
        oid_lo=info.doi_oid_lo,
    )


# ---------------------------------------------------------------------------
# Parallel directory iteration (dfs_obj_anchor_split)
# ---------------------------------------------------------------------------


def dfs_obj_anchor_split_optimal(obj: dfs_obj_p) -> int:
    """Return how many parallel iteration anchors DAOS will allow for ``obj``."""
    _, libdfs = _ensure_libs()
    nr = c_uint32(0)
    rc = libdfs.dfs_obj_anchor_split(obj, byref(nr), None)
    _check(rc, "obj_anchor_split(query)")
    return int(nr.value)


def dfs_obj_anchor_split(obj: dfs_obj_p, nr_splits: int) -> list[daos_anchor_t]:
    """Split a directory into ``nr_splits`` parallel iteration anchors.

    ``nr_splits`` must equal :func:`dfs_obj_anchor_split_optimal` for this
    object; any other value returns EINVAL. See that function for why.
    """
    _, libdfs = _ensure_libs()
    nr = c_uint32(nr_splits)
    anchors = (daos_anchor_t * nr_splits)()
    for i in range(nr_splits):
        ctypes.memset(byref(anchors[i]), 0, sizeof(daos_anchor_t))
    rc = libdfs.dfs_obj_anchor_split(obj, byref(nr), anchors)
    _check(rc, "obj_anchor_split")
    return list(anchors[: nr.value])


# ---------------------------------------------------------------------------
# readdirplus with explicit anchor (for parallel iteration)
# ---------------------------------------------------------------------------


def dfs_readdirplus_at(
    dfs: dfs_p, obj: dfs_obj_p, anchor: daos_anchor_t, batch_size: int = 128
) -> list[DirEntry]:
    """Read one batch of directory entries from a specific anchor position.

    Caller manages the anchor across calls. When anchor is EOF, iteration
    for that partition is complete.
    """
    _, libdfs = _ensure_libs()
    dirs_array = (dirent * batch_size)()
    stats_array = (struct_stat * batch_size)()
    nr = c_uint32(batch_size)
    rc = libdfs.dfs_readdirplus(dfs, obj, byref(anchor), byref(nr), dirs_array, stats_array)
    _check(rc, "readdirplus_at")
    entries: list[DirEntry] = []
    for i in range(nr.value):
        name = dirs_array[i].d_name.split(b"\x00", 1)[0].decode("utf-8", errors="surrogateescape")
        st = _parse_stat(stats_array[i])
        entries.append(DirEntry(name=name, stat=st))
    return entries


# ---------------------------------------------------------------------------
# DFS punch (truncate / hole-punch)
# ---------------------------------------------------------------------------

# DFS_MAX_FSIZE: use as length to truncate to offset
DFS_MAX_FSIZE = (1 << 64) - 1


def dfs_punch(dfs: dfs_p, obj: dfs_obj_p, offset: int, length: int) -> None:
    """Punch a hole or truncate a DFS file.

    offset: start offset.
    length: number of bytes to punch. Use DFS_MAX_FSIZE to truncate file to offset.
    """
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_punch(dfs, obj, daos_size_t(offset), daos_size_t(length))
    _check(rc, "punch")


# ---------------------------------------------------------------------------
# DFS suggest_oclass
# ---------------------------------------------------------------------------


def dfs_suggest_oclass(dfs: dfs_p, hint: str) -> int:
    """Suggest an object class from a hint string (e.g. 'file:max', 'dir:single').

    Returns the oclass ID.
    """
    _, libdfs = _ensure_libs()
    cid = c_uint32(0)  # daos_oclass_id_t is uint32: uint16 truncates real classes
    rc = libdfs.dfs_suggest_oclass(dfs, hint.encode(), byref(cid))
    _check(rc, f"suggest_oclass({hint})")
    return cid.value


# ---------------------------------------------------------------------------
# DFS set_prefix
# ---------------------------------------------------------------------------


def dfs_set_prefix(dfs: dfs_p, prefix: str) -> None:
    """Set a prefix to trim from all paths."""
    _, libdfs = _ensure_libs()
    rc = libdfs.dfs_set_prefix(dfs, prefix.encode())
    _check(rc, "set_prefix")


# ---------------------------------------------------------------------------
# DFS dup
# ---------------------------------------------------------------------------


def dfs_dup(dfs: dfs_p, obj: dfs_obj_p, flags: int = 0) -> dfs_obj_p:
    """Duplicate a DFS object handle locally (no RPC)."""
    _, libdfs = _ensure_libs()
    new_obj = dfs_obj_p()
    rc = libdfs.dfs_dup(dfs, obj, flags, byref(new_obj))
    _check(rc, "dup")
    return new_obj
