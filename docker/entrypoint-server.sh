#!/bin/bash
set -e

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
daos_usable_ifaces() {
    daos_server network scan --config=/etc/daos/daos_server.yml 2>/dev/null \
        | awk '/^[[:space:]]*ofi\+/ {print $NF}' | sort -u
}

if [ -n "${DAOS_FABRIC_IFACE:-}" ]; then
    echo "==> Fabric interface pinned by DAOS_FABRIC_IFACE: $IFACE (skipping scan validation)"
else
    USABLE="$(daos_usable_ifaces || true)"
    if [ -n "$USABLE" ]; then
        echo "==> Fabric interfaces DAOS reports as usable: $(echo "$USABLE" | tr '\n' ' ')"
        if [ -n "$IFACE" ] && ! echo "$USABLE" | grep -qx -- "$IFACE"; then
            FALLBACK="$(echo "$USABLE" | head -1)"
            echo "==> WARNING: detected interface '$IFACE' is not usable by DAOS; falling back to '$FALLBACK'."
            IFACE="$FALLBACK"
        fi
    else
        echo "==> WARNING: 'daos_server network scan' returned no usable interfaces; proceeding with '${IFACE:-none}'."
    fi
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

echo "==> Waiting for DAOS server to respond..."
for i in $(seq 1 30); do
    if timeout 5s dmg -o /etc/daos/dmg.yml system query 2>&1 | grep -qE "Pool|Rank|joined|uninitialized"; then
        echo "==> DAOS server is responding."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> ERROR: DAOS server did not start in time."
        exit 1
    fi
    sleep 1
done

echo "==> Formatting storage..."
timeout 60s dmg -o /etc/daos/dmg.yml storage format --force 2>&1 || true
sleep 5

echo "==> Waiting for the engine to join..."
for i in $(seq 1 30); do
    if timeout 5s dmg -o /etc/daos/dmg.yml system query 2>&1 | grep -qi "joined"; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "==> ERROR: engine did not join in time."
        exit 1
    fi
    sleep 1
done

POOL_NAME="${DAOS_POOL_NAME:-test-pool}"
POOL_SIZE="${DAOS_POOL_SIZE:-2G}"
if timeout 10s dmg -o /etc/daos/dmg.yml pool query "$POOL_NAME"; then
    echo "==> Using existing pool '${POOL_NAME}'."
else
    echo "==> Creating pool '${POOL_NAME}' with ${POOL_SIZE} of SCM..."
    timeout 90s dmg -o /etc/daos/dmg.yml pool create --scm-size="$POOL_SIZE" "$POOL_NAME"
fi

echo "==> DAOS server setup complete."
wait $SERVER_PID
