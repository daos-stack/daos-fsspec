/* Copyright 2026 Enakta Labs Ltd
 * SPDX-License-Identifier: Apache-2.0
 *
 * Controllable lifecycle ABI, not a substitute for a real DAOS live test.
 * Models the reference ownership in DAOS 2.6.5 and 2.8.0. The Python tests
 * call these symbols through ctypes, including its configured return types.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

enum { INIT = 1, DFS_INIT, DFS_FINI, FINI };
static int daos_refs, dfs_refs;
static int calls[5], failures[5], fail_after[5];
static pid_t creator_pid;

__attribute__((constructor)) static void record_creator(void) {
    creator_pid = getpid();
}

/* Observe after Python's exit callbacks and module cleanup, from outside the
 * Python runtime. Tests read this file only after subprocess termination. */
__attribute__((destructor)) static void report_exit(void) {
    const char *path = getenv("DAOS_FSSPEC_NATIVE_EXIT_TRACE");
    FILE *out;
    if (path == NULL || *path == '\0') return;
    out = fopen(path, "a");
    if (out == NULL) return;
    fprintf(out,
            "{\"pid\":%ld,\"creator_pid\":%ld,\"daos_refs\":%d,\"dfs_refs\":%d,"
            "\"init\":%d,\"dfs_init\":%d,\"dfs_fini\":%d,\"fini\":%d}\n",
            (long)getpid(), (long)creator_pid, daos_refs, dfs_refs,
            calls[INIT], calls[DFS_INIT], calls[DFS_FINI], calls[FINI]);
    fclose(out);
}

int test_daos_refs(void) { return daos_refs; }
int test_dfs_refs(void) { return dfs_refs; }
int test_calls(int which) { return calls[which]; }
void test_fail(int which, int rc, int after) {
    failures[which] = rc;
    fail_after[which] = after;
}

int daos_init(void) {
    calls[INIT]++;
    if (failures[INIT]) return failures[INIT];
    daos_refs++;
    return 0;
}

int daos_fini(void) {
    calls[FINI]++;
    if (failures[FINI] && !fail_after[FINI]) return failures[FINI];
    if (!daos_refs) return -1015;
    daos_refs--;
    /* A late error can accompany a runtime that has already been torn down. */
    return failures[FINI];
}

int dfs_init(void) {
    int rc;
    calls[DFS_INIT]++;
    if (failures[DFS_INIT]) return failures[DFS_INIT];
    if (!dfs_refs) {
        rc = daos_init();
        if (rc) return EIO;
    }
    dfs_refs++;
    return 0;
}

int dfs_fini(void) {
    int rc;
    calls[DFS_FINI]++;
    if (failures[DFS_FINI] && !fail_after[DFS_FINI]) return failures[DFS_FINI];
    if (!dfs_refs) return EINVAL;
    if (--dfs_refs == 0) {
        rc = daos_fini();
        if (rc) return EIO;
    }
    return failures[DFS_FINI];
}
