# Live DAOS integration test results

Standing record of `daos-fsspec` validated against a **real DAOS cluster** (not
mocks). The unit suite on GitHub Actions runs `-m "not live"` and never loads
`libdfs.so`; this file is the proof that the ctypes FFI actually works end to end
against real DAOS. Regenerate it after ABI-affecting changes.

## How to reproduce

The run uses a throwaway Ubuntu VM on the Core42 lab Proxmox host (DAOS needs
hugepages + privileged containers, which GitHub-hosted runners can't provide).

```bash
# On the Proxmox host (pve):
qm clone 902 300 --name daos-fsspec-test
qm set 300 --cores 32 --memory 98304 --cpu host --ipconfig0 ip=dhcp
qm start 300                                  # Docker preinstalled in template 902
ssh ubuntu@<vm-ip> 'sudo sysctl vm.nr_hugepages=4096'

# On the VM, with the repo shipped to ~/daos-fsspec:
cd ~/daos-fsspec && bash docker/test.sh --live
```

> **Host-side prerequisite:** `docker/test.sh` runs a unit-test preflight on the
> host (outside the container) before bringing DAOS up, so the host needs the
> test deps: `pip install fsspec pytest pytest-asyncio pytest-cov pytest-timeout
> hypothesis` (add `--break-system-packages` on a PEP 668 base image). The DAOS
> libraries themselves are only needed inside the containers, not on the host.

`docker/test.sh --live` builds the DAOS 2.6 server + client images, brings up a
RAM-backed single-node cluster, creates a POSIX container, and runs the full
suite (unit + live) inside the client container with `DAOS_FSSPEC_TEST_POOL` /
`DAOS_FSSPEC_TEST_CONTAINER` set.

## Latest run

| | |
|---|---|
| Date | 2026-06-25 |
| Code | HEAD `c296e20` + ABI fix (this change) |
| DAOS | `daos-2.6.4-7.suse.lp155.x86_64` (openSUSE Leap 15.6, RAM-backed) |
| Runner | Proxmox VM 300, 32 vCPU / 96 GB, Ubuntu 26.04, Docker 29.6.0 |
| Result | **472 passed in 10.11s**, 0 skipped, 0 failed |

```
collecting ... collected 472 items
============================= 472 passed in 10.11s =============================
[test] All tests passed.
```

All `@pytest.mark.live` tests executed against real DFS I/O — 64 MB files,
1000-file trees, 16-thread concurrent read/write, symlinks, xattr, chmod,
pool-space query, `dfs_obj_get_info`, and the benchmark suite (now collected) —
with **zero skips**.

> Two earlier rounds informed this result. The first live run (456 passed) only
> reached green after fixing the `dfs_obj_info_t`/`dfs_attr_t` structs below. A
> second round — a full audit of *every* test for assertions that pass without
> verifying the contract — found more real bugs (oclass uint16 truncation in
> `dfs_open`/`dfs_mkdir`/`suggest_oclass`; `pool_query_space` returning zeros;
> 10 benchmark tests that never ran). The live suite caught an incomplete fix of
> mine (a `c_uint16` value passed into the now-`c_uint32` `dfs_mkdir` slot,
> `ctypes.ArgumentError`) — now guarded in CI by
> `TestArgtypeValueConsistency`.

## ABI verification (DAOS 2.6.4 headers)

Every ctypes `Structure` in `_bindings.py` was checked field-by-field with a
compiled `offsetof()` probe against the `daos-devel-2.6.4` headers. The
data-path structs (`d_iov_t`, `d_sg_list_t`, `daos_anchor_t`, `struct_stat`,
`daos_handle_t`, `dirent`) match the C ABI exactly — which is why real I/O works.

Bugs found and fixed, all from typing `daos_oclass_id_t` as `uint16` when it is
`uint32` (sizeof 4):

- **`dfs_obj_info_t`** — `doi_file_oclass_id` at the wrong offset, oclass fields
  truncated, and `doi_oid` (`daos_obj_id_t {lo, hi}`) had `hi`/`lo` swapped. On a
  real file: returned `oclass_id=1` (real `16777217`), OID words reversed.
- **`dfs_attr_t`** — same oclass truncation, `da_file_oclass_id` misaligned,
  `da_hints` sized 256 vs the real 128.
- **`dfs_open` / `dfs_mkdir` / `dfs_suggest_oclass`** — the oclass argument/return
  was `c_uint16`, truncating real class ids (`suggest_oclass` returned `1`
  instead of `16777217`).
- **`pool_query_space`** — `_daos_pool_info` was 112 bytes with `pi_bits` at the
  wrong offset (real layout is 192 bytes, `pi_bits @40`, `pi_space @48`); it
  also never set `DPI_SPACE`, so it returned all zeros and risked a heap
  overflow from `daos_pool_query` writing past the short buffer. Rewritten to the
  compiler-verified 192-byte layout, `DPI_SPACE` set.

These slipped through the live suite because the relevant tests asserted only
`hasattr(...)` / `>= 0` / length, never the real values. Struct layouts are now
locked by `tests/test_bindings.py::TestStructABI` (incl. the pool structs), the
oclass argtypes by `TestArgtypeValueConsistency`, and the live tests assert real
values (object class `> 0xFFFF`, non-zero OID words, `0 <= free <= total`,
content round-trips). All pure-ctypes locks run in plain CI with no DAOS.
