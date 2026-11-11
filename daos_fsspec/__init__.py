# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""fsspec filesystem interface for DAOS DFS containers.

Provides ``dfs://pool/container/path`` URIs for the PyData ecosystem.
Requires ``libdaos.so`` and ``libdfs.so`` (DAOS 2.6+) on the system.

Usage::

    import fsspec
    fs = fsspec.filesystem("dfs", pool="mypool", container="mycont")
    data = fs.cat("/path/to/file")

    # Or via URI:
    df = pd.read_parquet("dfs://mypool/mycont/data.parquet")

    # Async:
    from daos_fsspec.async_core import DFSAsyncFileSystem
    fs = DFSAsyncFileSystem(pool="mypool", container="mycont")
    data = await fs._cat("/path/to/file")

    # Per-process cached filesystem: the supported pattern for spawn-based
    # DataLoader workers, since DFSFileSystem deliberately refuses to pickle:
    import daos_fsspec
    fs = daos_fsspec.get_fs("mypool", "mycont")

    # Connection pool:
    from daos_fsspec.pool import DFSFileSystemPool
    pool = DFSFileSystemPool(pool="mypool", container="mycont", size=4)
    with pool.connection() as fs:
        data = fs.cat("/path/to/file")
"""

from __future__ import annotations

from daos_fsspec._bindings import MountAttr, ObjInfo, PoolSpace
from daos_fsspec._fs_cache import cache_info, clear_fs_cache, get_fs
from daos_fsspec.core import DFSFileSystem
from daos_fsspec.dfs_file import DFSFile
from daos_fsspec.metrics import DFSFileSystemWithMetrics, DFSMetrics, OperationMetrics
from daos_fsspec.pool import DFSFileSystemPool

__all__ = [
    "DFSFile",
    "DFSFileSystem",
    "DFSFileSystemPool",
    "DFSFileSystemWithMetrics",
    "DFSMetrics",
    "MountAttr",
    "ObjInfo",
    "OperationMetrics",
    "PoolSpace",
    "cache_info",
    "clear_fs_cache",
    "get_fs",
]
__version__ = "1.4.1"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Register the dfs:// protocol with fsspec on import.
# This ensures the protocol is discoverable even without pip install
# (editable installs, sys.path hacks, frozen environments).
# Entry points in pyproject.toml handle the normal install case.
try:
    from fsspec import register_implementation

    register_implementation("dfs", "daos_fsspec:DFSFileSystem", clobber=True)
except Exception:
    import logging

    logging.getLogger(__name__).debug(
        "Could not register dfs:// protocol with fsspec", exc_info=True
    )
