#!/usr/bin/env bash
# Run every example against a real DAOS container and tally pass/fail.
#
#   export DAOS_POOL=mypool DAOS_CONTAINER=mycont
#   bash examples/run_all.sh
set -uo pipefail
cd "$(dirname "$0")"

: "${DAOS_POOL:?set DAOS_POOL first}"
: "${DAOS_CONTAINER:?set DAOS_CONTAINER first}"

echo "daos-fsspec examples: pool=$DAOS_POOL container=$DAOS_CONTAINER"
echo "version: $(python3 -c 'import daos_fsspec; print(daos_fsspec.__version__)' 2>&1)"

examples=(dataset_staging.py checkpoint_store.py concurrent_pool.py capacity_report.py)
declare -A result
fail=0
for ex in "${examples[@]}"; do
  echo
  echo "==================== $ex ===================="
  if python3 "$ex"; then result[$ex]="PASS"; else result[$ex]="FAIL"; fail=1; fi
done

echo
echo "===================== SUMMARY ====================="
for ex in "${examples[@]}"; do
  printf "  %-24s %s\n" "$ex" "${result[$ex]}"
done
[ "$fail" -eq 0 ] && echo "All examples passed." || echo "Some examples FAILED."
exit $fail
