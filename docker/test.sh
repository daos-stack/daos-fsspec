#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"
POOL_NAME="test-pool"
CONTAINER_NAME="fsspec-live-test"

DAOS_VERSION="${DAOS_VERSION:-2.6.5}"
export DAOS_VERSION

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[test]${NC} $*"; }
warn() { echo -e "${YELLOW}[test]${NC} $*"; }
err()  { echo -e "${RED}[test]${NC} $*" >&2; }

COMPOSE_COMMAND=()
detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_COMMAND=(docker compose)
    elif docker-compose version >/dev/null 2>&1; then
        COMPOSE_COMMAND=(docker-compose)
    else
        err "Docker Compose v2 is required (neither 'docker compose' nor 'docker-compose' found)"
        exit 1
    fi
    log "Using compose command: ${COMPOSE_COMMAND[*]}"
}

compose() { "${COMPOSE_COMMAND[@]}" -f "$COMPOSE_FILE" "$@"; }

pytest_timeout_args() {
    local seconds="$1"
    if python3 -c "import pytest_timeout" 2>/dev/null; then
        printf '%s' "--timeout=$seconds"
    else
        warn "pytest-timeout not installed; running with no per-test timeout" >&2
    fi
}

run_unit_tests() {
    log "Running unit tests (no DAOS required)..."
    cd "$PROJECT_DIR"
    local t
    t="$(pytest_timeout_args 300)"
    python3 -m pytest tests/ -v -m "not live" --tb=short ${t:+"$t"}
}



LIVE=false
SKIP_UNIT=false
KEEP_CONTAINERS=false
for arg in "$@"; do
    case "$arg" in
        --live)  LIVE=true ;;
        --keep)  KEEP_CONTAINERS=true ;;
        --skip-unit) SKIP_UNIT=true ;;
        *)       err "Unknown option: $arg"; exit 1 ;;
    esac
done

if [ "$SKIP_UNIT" = false ]; then
    run_unit_tests
else
    log "Skipping host-side unit tests (--skip-unit); the container runs the full suite."
fi

if [ "$LIVE" = false ]; then
    log "Done. Use --live to also run integration tests against DAOS."
    exit 0
fi


detect_compose

log "DAOS version under test: $DAOS_VERSION"

stale="$(docker ps -a --filter 'name=^daos-test-server$' --format '{{.Names}} {{.Status}}' || true)"
if [ -n "$stale" ] && ! docker ps --format '{{.Names}}' | grep -q '^daos-test-server$'; then
    owner="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' daos-test-server 2>/dev/null || echo unknown)"
    err "A stopped container named 'daos-test-server' exists (compose project: ${owner:-none}): $stale"
    err "Compose cannot reuse the name. Remove it if it is not needed:  docker rm daos-test-server daos-test-client"
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q '^daos-test-server$'; then
    running_version="$(docker exec daos-test-server printenv DAOS_VERSION 2>/dev/null || echo unknown)"
    if [ "$running_version" != "$DAOS_VERSION" ]; then
        err "A DAOS cluster is already running, but it is version '$running_version', not '$DAOS_VERSION'."
        err "Tear it down first:  DAOS_VERSION=$running_version docker/test.sh --live  (or: compose down)"
        exit 1
    fi
    log "DAOS containers already running at $DAOS_VERSION."
else
    log "Building and starting DAOS containers (DAOS $DAOS_VERSION)..."
    cd "$PROJECT_DIR"
    compose up -d --build
fi

log "Waiting for DAOS pool '$POOL_NAME' to be ready..."
pool_deadline=$((SECONDS + 180))
while true; do
    if docker exec daos-test-client timeout 5s daos pool query "$POOL_NAME" >/dev/null 2>&1; then
        log "Pool '$POOL_NAME' is ready."
        break
    fi
    server_running="$(docker inspect -f '{{.State.Running}}' daos-test-server 2>/dev/null || echo false)"
    if [ "$SECONDS" -ge "$pool_deadline" ] || [ "$server_running" != true ]; then
        err "Pool '$POOL_NAME' is unavailable; server running: $server_running."
        docker logs daos-test-server 2>&1
        docker logs daos-test-client 2>&1
        docker exec daos-test-server cat /tmp/daos_engine.log 2>&1 | tail -80 || true
        exit 1
    fi
    sleep 2
done

log "Creating test container $CONTAINER_NAME on pool $POOL_NAME..."
docker exec daos-test-client daos container create "$POOL_NAME" "$CONTAINER_NAME" --type=POSIX 2>/dev/null || true

log "Copying source into daos-test-client..."
docker cp "$PROJECT_DIR/." daos-test-client:/tmp/daos-fsspec/

log "Reinstalling daos-fsspec in the client from the copied source..."
docker exec daos-test-client python3 -m pip install --quiet --no-deps \
    --force-reinstall /tmp/daos-fsspec
docker exec daos-test-client python3 -c "
import daos_fsspec, pathlib
print('  daos_fsspec', daos_fsspec.__version__, 'from', pathlib.Path(daos_fsspec.__file__).parent)
"

log "Verifying ctypes ABI against DAOS $DAOS_VERSION headers..."
ABI_EXIT=0
docker exec \
    -e "DAOS_VERSION=$DAOS_VERSION" \
    daos-test-client \
    python3 /tmp/daos-fsspec/docker/abi_probe.py || ABI_EXIT=$?
if [ "$ABI_EXIT" -ne 0 ]; then
    err "ctypes ABI verification failed (exit code $ABI_EXIT); refusing to run against mismatched layouts."
    if [ "$KEEP_CONTAINERS" = false ]; then
        log "Tearing down DAOS containers..."
        cd "$PROJECT_DIR"
        compose down
    fi
    exit 1
fi

docker exec daos-test-client python3 -c "import pandas" 2>/dev/null || {
    log "Installing pandas + pyarrow in container..."
    docker exec daos-test-client python3 -m pip install --quiet pandas pyarrow
}

LIVE_TIMEOUT=""
if docker exec daos-test-client python3 -c "import pytest_timeout" 2>/dev/null; then
    LIVE_TIMEOUT="--timeout=600"
else
    warn "pytest-timeout missing in the client image; running with no per-test timeout"
fi

log "Running live tests against DAOS $DAOS_VERSION..."
LIVE_EXIT=0
docker exec \
    -e "DAOS_FSSPEC_TEST_POOL=$POOL_NAME" \
    -e "DAOS_FSSPEC_TEST_CONTAINER=$CONTAINER_NAME" \
    -e "DAOS_VERSION=$DAOS_VERSION" \
    daos-test-client \
    python3 -m pytest /tmp/daos-fsspec/tests/ -v --tb=short ${LIVE_TIMEOUT:+$LIVE_TIMEOUT} || LIVE_EXIT=$?

if [ "$KEEP_CONTAINERS" = false ]; then
    log "Tearing down DAOS containers..."
    cd "$PROJECT_DIR"
    compose down
fi

if [ "$LIVE_EXIT" -eq 0 ]; then
    log "All tests passed."
else
    err "Live tests failed (exit code $LIVE_EXIT)."
    exit 1
fi
