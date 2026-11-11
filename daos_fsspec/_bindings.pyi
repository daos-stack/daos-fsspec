"""Type stubs for daos_fsspec._bindings (ctypes FFI to libdaos.so + libdfs.so).

These stubs enable IDE autocompletion and mypy type checking for code
that uses the bindings module. The actual implementation is in _bindings.py
using ctypes.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from daos_fsspec._runtime import Runtime as _Runtime

_runtime: _Runtime

def _dfs_init() -> None: ...

# ---------------------------------------------------------------------------
# Opaque handle types (ctypes wrappers, not usable as plain ints)
# ---------------------------------------------------------------------------

class daos_handle_t(ctypes.Structure):
    cookie: int
    ...

class dfs_p: ...  # c_void_p alias
class dfs_obj_p: ...  # c_void_p alias

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAOS_PC_RO: int
DAOS_PC_RW: int
DAOS_COO_RO: int
DAOS_COO_RW: int

DFS_SET_ATTR_MODE: int
DFS_SET_ATTR_ATIME: int
DFS_SET_ATTR_MTIME: int
DFS_SET_ATTR_SIZE: int
DFS_SET_ATTR_UID: int
DFS_SET_ATTR_GID: int

DAOS_PROP_LABEL_MAX_LEN: int
DAOS_ANCHOR_BUF_MAX: int
DAOS_ANCHOR_TYPE_ZERO: int
DAOS_ANCHOR_TYPE_EOF: int

# Error codes (negative = DAOS-specific, positive = POSIX errno)
DER_SUCCESS: int
DER_NO_PERM: int
DER_NO_HDL: int
DER_INVAL: int
DER_EXIST: int
DER_NONEXIST: int
DER_UNREACH: int
DER_NOSPACE: int
DER_ALREADY: int
DER_NOMEM: int
DER_TIMEDOUT: int
DER_NOSYS: int
DER_NO_ENTRY: int
DER_BUSY: int
DER_HG: int

RETRYABLE_ERRORS: frozenset[int]
CHUNK_SIZE: int

# xattr flags and limits
XATTR_CREATE: int
XATTR_REPLACE: int
DFS_MAX_XATTR_NAME: int
DFS_MAX_XATTR_LEN: int

# Punch truncate sentinel
DFS_MAX_FSIZE: int

# Feature availability flags
_HAS_DFS_CONNECT: bool
_HAS_SNAP: bool
_HAS_LISTIO: bool
_HAS_PUNCH: bool
_HAS_SET_PREFIX: bool
_HAS_SUGGEST_OCLASS: bool
_HAS_DUP: bool
_HAS_CONT_CHECK: bool
_HAS_DIR_ANCHOR_SET: bool
_HAS_OBJ_ANCHOR_SET: bool
_HAS_ITERATE: bool

# Internal helpers (not part of public API, but used by core.py)
_CHUNK_SIZE: int

def _ensure_libs() -> tuple[ctypes.CDLL, ctypes.CDLL]: ...
def _check(rc: int, msg: str = ...) -> None: ...
def _anchor_is_eof(anchor: Any) -> bool: ...

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StatResult:
    size: int
    mode: int
    uid: int
    gid: int
    nlink: int
    mtime: float
    atime: float
    ctime: float
    @property
    def is_dir(self) -> bool: ...
    @property
    def is_file(self) -> bool: ...
    @property
    def is_link(self) -> bool: ...
    @property
    def type_name(self) -> str: ...

@dataclass
class DirEntry:
    name: str
    stat: StatResult | None

@dataclass
class ContainerInfo:
    uuid: str
    label: str

@dataclass
class PoolSpace:
    scm_total: int
    scm_free: int
    nvme_total: int
    nvme_free: int

@dataclass
class MountAttr:
    oclass_id: int
    dir_oclass_id: int
    file_oclass_id: int
    chunk_size: int
    mode: int

@dataclass
class ObjInfo:
    oclass_id: int
    dir_oclass_id: int
    file_oclass_id: int
    chunk_size: int
    oid_hi: int
    oid_lo: int

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DaosError(Exception):
    rc: int
    def __init__(self, rc: int, msg: str = ...) -> None: ...
    @property
    def is_retryable(self) -> bool: ...

# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------

def daos_init() -> None: ...
def daos_fini() -> None: ...

# ---------------------------------------------------------------------------
# Pool / container connection
# ---------------------------------------------------------------------------

def pool_connect(pool_label: str, flags: int = ...) -> daos_handle_t: ...
def pool_disconnect(poh: daos_handle_t) -> None: ...
def cont_open(poh: daos_handle_t, cont_label: str, flags: int = ...) -> daos_handle_t: ...
def cont_close(coh: daos_handle_t) -> None: ...

# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

def cont_create(poh: daos_handle_t, label: str) -> str: ...
def cont_destroy(poh: daos_handle_t, label: str, force: bool = ...) -> None: ...
def pool_list_cont(poh: daos_handle_t) -> list[ContainerInfo]: ...

# ---------------------------------------------------------------------------
# DFS mount
# ---------------------------------------------------------------------------

def dfs_mount_fs(poh: daos_handle_t, coh: daos_handle_t, readonly: bool = ...) -> dfs_p: ...
def dfs_umount(dfs: dfs_p) -> None: ...

# ---------------------------------------------------------------------------
# Path operations
# ---------------------------------------------------------------------------

def dfs_lookup_path(dfs: dfs_p, path: str, flags: int = ...) -> tuple[dfs_obj_p, StatResult]: ...
def dfs_lookup_rel_path(
    dfs: dfs_p, parent: dfs_obj_p, name: str, flags: int = ...
) -> tuple[dfs_obj_p, StatResult]: ...
def same_handle(a: Any, b: Any) -> bool: ...
def dfs_readdir(dfs: dfs_p, obj: dfs_obj_p) -> list[str]: ...
def dfs_stat_rel(dfs: dfs_p, parent: dfs_obj_p, name: str) -> StatResult | None: ...
def dfs_release(obj: dfs_obj_p) -> None: ...
def dfs_stat(dfs: dfs_p, obj: dfs_obj_p) -> StatResult: ...
def dfs_setattr(
    dfs: dfs_p,
    obj: dfs_obj_p,
    mtime: float | None = ...,
    atime: float | None = ...,
    file_size: int | None = ...,
) -> None: ...

# ---------------------------------------------------------------------------
# Symlink operations
# ---------------------------------------------------------------------------

def dfs_create_symlink(dfs: dfs_p, parent: dfs_obj_p, name: str, target: str) -> dfs_obj_p: ...
def dfs_read_symlink(obj: dfs_obj_p) -> str: ...

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def dfs_open_file(
    dfs: dfs_p,
    parent: dfs_obj_p,
    name: str,
    mode: int,
    flags: int,
    oclass_id: int = ...,
    chunk_size: int = ...,
) -> dfs_obj_p: ...
def dfs_read(dfs: dfs_p, obj: dfs_obj_p, offset: int, size: int) -> bytes: ...
def dfs_write(dfs: dfs_p, obj: dfs_obj_p, offset: int, data: bytes) -> int: ...
def dfs_close(obj: dfs_obj_p) -> None: ...

# ---------------------------------------------------------------------------
# Directory operations
# ---------------------------------------------------------------------------

def dfs_readdirplus(dfs: dfs_p, obj: dfs_obj_p) -> list[DirEntry]: ...
def dfs_mkdir(
    dfs: dfs_p, parent: dfs_obj_p, name: str, mode: int = ..., oclass_id: int = ...
) -> None: ...
def dfs_remove(dfs: dfs_p, parent: dfs_obj_p, name: str, force: bool = ...) -> None: ...
def dfs_move(
    dfs: dfs_p,
    src_parent: dfs_obj_p,
    src_name: str,
    dst_parent: dfs_obj_p,
    dst_name: str,
) -> None: ...
def dfs_exchange(
    dfs: dfs_p,
    parent1: dfs_obj_p,
    name1: str,
    parent2: dfs_obj_p,
    name2: str,
) -> None: ...

# ---------------------------------------------------------------------------
# Extended attributes
# ---------------------------------------------------------------------------

def dfs_setxattr(dfs: dfs_p, obj: dfs_obj_p, name: str, value: bytes, flags: int = ...) -> None: ...
def dfs_getxattr(dfs: dfs_p, obj: dfs_obj_p, name: str) -> bytes: ...
def dfs_listxattr(dfs: dfs_p, obj: dfs_obj_p) -> list[str]: ...
def dfs_removexattr(dfs: dfs_p, obj: dfs_obj_p, name: str) -> None: ...

# ---------------------------------------------------------------------------
# Attribute operations
# ---------------------------------------------------------------------------

def dfs_chmod(dfs: dfs_p, parent: dfs_obj_p, name: str, mode: int) -> None: ...
def dfs_chown(
    dfs: dfs_p, parent: dfs_obj_p, name: str, uid: int, gid: int, flags: int = ...
) -> None: ...
def dfs_access(dfs: dfs_p, parent: dfs_obj_p, name: str, mask: int) -> bool: ...
def dfs_get_size(dfs: dfs_p, obj: dfs_obj_p) -> int: ...
def dfs_get_chunk_size(dfs: dfs_p, obj: dfs_obj_p) -> int: ...

# ---------------------------------------------------------------------------
# Pool space query
# ---------------------------------------------------------------------------

def pool_query_space(poh: daos_handle_t) -> PoolSpace: ...
def dfs_pool_get(dfs: dfs_p) -> daos_handle_t: ...
def dfs_pool_put(dfs: dfs_p, poh: daos_handle_t) -> None: ...

# ---------------------------------------------------------------------------
# Mount attributes / object info
# ---------------------------------------------------------------------------

def dfs_query_attr(dfs: dfs_p) -> MountAttr: ...
def dfs_obj_get_info(dfs: dfs_p, obj: dfs_obj_p) -> ObjInfo: ...

# ---------------------------------------------------------------------------
# Parallel directory iteration
# ---------------------------------------------------------------------------

def dfs_obj_anchor_split_optimal(obj: dfs_obj_p) -> int: ...
def dfs_obj_anchor_split(obj: dfs_obj_p, nr_splits: int) -> list: ...
def dfs_readdirplus_at(
    dfs: dfs_p, obj: dfs_obj_p, anchor: Any, batch_size: int = ...
) -> list[DirEntry]: ...

# ---------------------------------------------------------------------------
# Punch / truncate
# ---------------------------------------------------------------------------

def dfs_punch(dfs: dfs_p, obj: dfs_obj_p, offset: int, length: int) -> None: ...

# ---------------------------------------------------------------------------
# Object class suggestion
# ---------------------------------------------------------------------------

def dfs_suggest_oclass(dfs: dfs_p, hint: str) -> int: ...

# ---------------------------------------------------------------------------
# Path prefix
# ---------------------------------------------------------------------------

def dfs_set_prefix(dfs: dfs_p, prefix: str) -> None: ...

# ---------------------------------------------------------------------------
# Dup
# ---------------------------------------------------------------------------

def dfs_dup(dfs: dfs_p, obj: dfs_obj_p, flags: int = ...) -> dfs_obj_p: ...
