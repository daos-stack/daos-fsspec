# Docker-based DAOS test environment

This directory contains everything needed to run daos-fsspec tests against a real DAOS cluster in Docker containers.

## Contents

| File | Purpose |
|------|---------|
| `Dockerfile.daos-server` | DAOS control plane + engine (openSUSE Leap 15.6, DAOS 2.6) |
| `Dockerfile.daos-client` | DAOS agent + Python test runner (openSUSE Leap 15.6, Python 3.11) |
| `docker-compose.test.yml` | Orchestrates server and client containers |
| `daos_server.yml` | DAOS server configuration (RAM-backed storage, insecure mode) |
| `daos_agent.yml` | DAOS agent configuration |
| `dmg.yml` | DAOS management tool configuration |
| `entrypoint-server.sh` | Server startup: format storage, create pool |
| `entrypoint-client.sh` | Client startup: start agent, wait for pool, create container |
| `test.sh` | Test runner script (unit tests, live tests, or both) |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  daos-test-server (privileged, host network)    │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ daos_server  │  │ daos_engine (RAM-backed) │  │
│  │   :10001     │  │   /mnt/daos              │  │
│  └─────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────┘
         ↕ shared memory / hugepages
┌─────────────────────────────────────────────────┐
│  daos-test-client (privileged, host network)    │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ daos_agent   │  │ pytest (Python 3.11)     │  │
│  │              │  │ daos-fsspec installed     │  │
│  └─────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

Both containers run in **privileged mode** with **host networking** (required by DAOS for shared memory and hugepages access). The server uses RAM-backed storage (`class: ram`) so no NVMe or SCM hardware is needed.

## Quick start

```bash
# 0. Host needs the Python test deps for the pre-container unit preflight
#    (DAOS libs are only required inside the containers, not on the host):
pip install fsspec pytest pytest-asyncio pytest-cov pytest-timeout hypothesis

# 1. Allocate hugepages (one-time per host boot)
sudo sysctl vm.nr_hugepages=4096

# 2. Run all tests (builds, starts DAOS, runs tests, tears down)
docker/test.sh --live

# 3. Or keep containers running for development
docker/test.sh --live --keep
```

## Manual usage

```bash
# Start the cluster
docker compose -f docker/docker-compose.test.yml up -d --build

# Wait ~15s for server to format storage and create pool, then:
docker exec daos-test-client daos container create test-pool test-cont --type=POSIX

# Run tests
docker exec \
  -e DAOS_FSSPEC_TEST_POOL=test-pool \
  -e DAOS_FSSPEC_TEST_CONTAINER=test-cont \
  daos-test-client \
  python3 -m pytest /tmp/daos-fsspec/tests/ -v

# Tear down
docker compose -f docker/docker-compose.test.yml down
```

## Development workflow

With `--keep`, the containers stay running. You can iterate on code by copying it into the client:

```bash
# Copy updated source
docker cp daos_fsspec/. daos-test-client:/tmp/daos-fsspec/daos_fsspec/

# Re-run a single test
docker exec \
  -e DAOS_FSSPEC_TEST_POOL=test-pool \
  -e DAOS_FSSPEC_TEST_CONTAINER=test-cont \
  daos-test-client \
  python3 -m pytest /tmp/daos-fsspec/tests/test_filesystem.py::TestBasicOps::test_mkdir_ls -v
```

## Configuration

The server config (`daos_server.yml`) uses:

- **RAM-backed storage** (`class: ram`, 16 GB) — no hardware requirements
- **Insecure mode** (`allow_insecure: true`) — no certificates needed
- **ofi+tcp provider** — works in any Docker/network environment
- **1 engine, 1 target** — minimal for testing

Override pool size via environment:

```bash
DAOS_POOL_SIZE=32G docker compose -f docker/docker-compose.test.yml up -d --build
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hugepages too low` | Host has no hugepages allocated | `sudo sysctl vm.nr_hugepages=4096` |
| Server won't start | Not running privileged | Ensure `privileged: true` in compose |
| Agent can't connect | Server not ready yet | Wait 15s; entrypoint retries for 30s |
| Pool creation fails | Not enough memory for RAM storage | Reduce `DAOS_POOL_SIZE` or add host RAM |
| Container already exists | Idempotent — safe to ignore | Expected on re-runs |
