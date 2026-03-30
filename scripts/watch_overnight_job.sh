#!/usr/bin/env bash
set -euo pipefail

MAX_HOURS="${1:-12}"
CHECK_INTERVAL_SEC="${2:-90}"

OVERNIGHT_CMD="bash /home/howard/Splat-Nav/scripts/run_overnight_12h.sh old_union /data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml 12 180 sfc-1,sfc-2,sfc-3,sfc-4,splatplan 8"
PATTERN="run_overnight_12h.sh old_union /data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml 12 180 sfc-1,sfc-2,sfc-3,sfc-4,splatplan 8"

START_TS="$(date +%s)"
MAX_SECONDS="$((MAX_HOURS * 3600))"

echo "[watchdog] max_hours=$MAX_HOURS check_interval_sec=$CHECK_INTERVAL_SEC"
echo "[watchdog] pattern=$PATTERN"

while true; do
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED >= MAX_SECONDS )); then
    echo "[watchdog] reached max runtime, stop."
    break
  fi

  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "[watchdog] alive elapsed_sec=$ELAPSED"
  else
    echo "[watchdog] target not found, restarting..."
    bash -lc "$OVERNIGHT_CMD" >> /home/howard/Splat-Nav/runs/watchdog_restart.log 2>&1 &
    sleep 8
    if pgrep -f "$PATTERN" >/dev/null 2>&1; then
      echo "[watchdog] restart ok"
    else
      echo "[watchdog] restart failed, will retry next cycle"
    fi
  fi

  sleep "$CHECK_INTERVAL_SEC"
done

echo "[watchdog] completed"
