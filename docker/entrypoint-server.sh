#!/bin/bash
set -e

# Auto-detect the fabric interface: the static daos_server.yml ships a default
# (e.g. em1) that won't exist on every host (cloud VMs use ens18/eth0/etc.).
# Pure-bash detection so it works in minimal images without iproute2 — read the
# default route from /proc/net/route, then fall back to the first "up" non-virtual
# interface. Override with DAOS_FABRIC_IFACE if auto-detection picks wrong.
IFACE="${DAOS_FABRIC_IFACE:-}"
if [ -z "$IFACE" ] && [ -r /proc/net/route ]; then
    while read -r ifc dest _; do
        if [ "$dest" = "00000000" ] && [ "$ifc" != "Iface" ]; then IFACE="$ifc"; break; fi
    done < /proc/net/route
fi
if [ -z "$IFACE" ]; then
    for d in /sys/class/net/*; do
        n=$(basename "$d")
        case "$n" in lo|docker*|veth*|br-*|virbr*) continue;; esac
        if [ "$(cat "$d/operstate" 2>/dev/null)" = "up" ]; then IFACE="$n"; break; fi
    done
fi
if [ -n "$IFACE" ] && [ -e "/sys/class/net/$IFACE/type" ]; then
    echo "==> Using fabric interface: $IFACE"
    sed -i "s/^\( *fabric_iface:\).*/\1 ${IFACE}/" /etc/daos/daos_server.yml
else
    echo "==> WARNING: could not auto-detect a fabric interface (got '${IFACE:-none}'); using config as-is."
fi

echo "==> Starting DAOS server..."
daos_server start --config=/etc/daos/daos_server.yml &
SERVER_PID=$!

# Wait for server to respond to dmg commands (any output means it's up)
echo "==> Waiting for DAOS server to respond..."
for i in $(seq 1 30); do
    if dmg -o /etc/daos/dmg.yml system query 2>&1 | grep -qE "Pool|Rank|joined|uninitialized"; then
        echo "==> DAOS server is responding."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> ERROR: DAOS server did not start in time."
        exit 1
    fi
    sleep 1
done

# Format storage (idempotent — skips if already formatted)
echo "==> Formatting storage..."
dmg -o /etc/daos/dmg.yml storage format --force 2>&1 || true
sleep 5

# Wait for storage to be ready
echo "==> Waiting for storage..."
for i in $(seq 1 20); do
    if dmg -o /etc/daos/dmg.yml storage query usage 2>&1 | grep -q "smd"; then
        echo "==> Storage is ready."
        break
    fi
    sleep 1
done

# Create pool if it doesn't exist
POOL_NAME="${DAOS_POOL_NAME:-test-pool}"
POOL_SIZE="${DAOS_POOL_SIZE:-1G}"
echo "==> Creating pool '${POOL_NAME}' (${POOL_SIZE})..."
dmg -o /etc/daos/dmg.yml pool create --size="${POOL_SIZE}" "${POOL_NAME}" 2>&1 || \
    echo "==> Pool '${POOL_NAME}' already exists or creation failed (continuing)."

echo "==> DAOS server setup complete."
wait $SERVER_PID
