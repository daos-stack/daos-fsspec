#!/bin/bash
set -e

echo "==> Starting DAOS agent..."
daos_agent -o /etc/daos/daos_agent.yml &
AGENT_PID=$!

echo "==> Waiting for DAOS agent..."
for i in $(seq 1 30); do
    if daos_agent -o /etc/daos/daos_agent.yml dump-attachinfo 2>&1 | grep -q "ofi+tcp"; then
        echo "==> DAOS agent is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> WARNING: DAOS agent attachinfo check timed out (continuing anyway)."
    fi
    sleep 1
done

POOL_NAME="${DAOS_POOL_NAME:-test-pool}"
CONT_NAME="${DAOS_CONTAINER_NAME:-test-cont}"
echo "==> Waiting for pool '${POOL_NAME}' to be available..."
for i in $(seq 1 30); do
    if daos pool query "${POOL_NAME}" 2>&1 | grep -q "Pool"; then
        echo "==> Pool '${POOL_NAME}' is available."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> WARNING: Pool '${POOL_NAME}' not available after 30s."
    fi
    sleep 1
done

echo "==> Creating container '${CONT_NAME}' in pool '${POOL_NAME}'..."
daos cont create "${POOL_NAME}" "${CONT_NAME}" --type=POSIX 2>&1 || \
    echo "==> Container '${CONT_NAME}' already exists or creation failed (continuing)."

echo "==> DAOS client setup complete."

if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "==> Sleeping forever (use docker exec to run tests)..."
    wait $AGENT_PID
fi
