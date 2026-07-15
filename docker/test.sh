#!/usr/bin/env bash
# docker/test.sh — Full daos-fsspec test cycle with Docker DAOS.
#
# Usage:
#   docker/test.sh              # unit tests only (no DAOS)
#   docker/test.sh --live       # build containers, start DAOS, run all tests, tear down
#   docker/test.sh --live --keep  # keep containers running after tests
#
# Prerequisites:
#   - Docker and Docker Compose
#   - sudo sysctl vm.nr_hugepages=4096 (for DAOS engine)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"
POOL_NAME="test-pool"
CONTAINER_NAME="fsspec-live-test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[test]${NC} $*"; }
warn() { echo -e "${YELLOW}[test]${NC} $*"; }
err()  { echo -e "${RED}[test]${NC} $*" >&2; }

run_unit_tests() {
    log "Running unit tests (no DAOS required)..."
    cd "$PROJECT_DIR"
    python3 -m pytest tests/ -v -m "not live" --tb=short
}

check_hugepages() {
    local hp
    hp=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo "0")
    if [ "$hp" -lt 2048 ]; then
        warn "Hugepages too low ($hp). DAOS engine needs at least 4096."
        warn "Run: sudo sysctl vm.nr_hugepages=4096"
    fi
}

# ── Parse args ──────────────────────────────────────────────────────

LIVE=false
KEEP_CONTAINERS=false
for arg in "$@"; do
    case "$arg" in
        --live)  LIVE=true ;;
        --keep)  KEEP_CONTAINERS=true ;;
        *)       err "Unknown option: $arg"; exit 1 ;;
    esac
done

# Always run unit tests
run_unit_tests

if [ "$LIVE" = false ]; then
    log "Done. Use --live to also run integration tests against DAOS."
    exit 0
fi

# ── Live tests ──────────────────────────────────────────────────────

check_hugepages

# Check if containers are already running
if docker ps --format '{{.Names}}' | grep -q '^daos-test-server$'; then
    log "DAOS containers already running."
else
    log "Building and starting DAOS containers..."
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" up -d --build
fi

# Wait for the DAOS pool to be provisioned by the server entrypoint (async).
# A fixed sleep races format+pool-create (~30-40s), so poll until the client can
# actually query the pool before creating the test container / running tests.
log "Waiting for DAOS pool '$POOL_NAME' to be ready..."
for i in $(seq 1 60); do
    if docker exec daos-test-client daos pool query "$POOL_NAME" >/dev/null 2>&1; then
        log "Pool '$POOL_NAME' is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        err "Pool '$POOL_NAME' did not become ready in time."
        docker logs daos-test-server 2>&1 | tail -20
        exit 1
    fi
    sleep 2
done

# Create test container (idempotent)
log "Creating test container $CONTAINER_NAME on pool $POOL_NAME..."
docker exec daos-test-client daos container create "$POOL_NAME" "$CONTAINER_NAME" --type=POSIX 2>/dev/null || true

# Copy source into container
log "Copying source into daos-test-client..."
docker cp "$PROJECT_DIR/." daos-test-client:/tmp/daos-fsspec/

# Install pandas+pyarrow if missing
docker exec daos-test-client python3 -c "import pandas" 2>/dev/null || {
    log "Installing pandas + pyarrow in container..."
    docker exec daos-test-client python3 -m pip install --quiet pandas pyarrow
}

# Run live tests
log "Running live tests against DAOS..."
docker exec \
    -e "DAOS_FSSPEC_TEST_POOL=$POOL_NAME" \
    -e "DAOS_FSSPEC_TEST_CONTAINER=$CONTAINER_NAME" \
    daos-test-client \
    python3 -m pytest /tmp/daos-fsspec/tests/ -v --tb=short
LIVE_EXIT=$?

if [ "$KEEP_CONTAINERS" = false ]; then
    log "Tearing down DAOS containers..."
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down
fi

if [ "$LIVE_EXIT" -eq 0 ]; then
    log "All tests passed."
else
    err "Live tests failed (exit code $LIVE_EXIT)."
    exit 1
fi
