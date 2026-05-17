#!/bin/bash
# Poll sibling NOTES.md files. Exit 0 when all 4 ready or 4h elapsed.
set -u
WORKDIR=/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v5-overnight
LOG=$WORKDIR/loop-runner/poll.log
READY_MARKER=$WORKDIR/loop-runner/SIBLINGS_READY
SIBS=(snic-real mediapipe-spatial alpha-proof-dumper mokuhanga-pigments)
MAX_WAIT_S=14400   # 4 hours
POLL_S=120
START_TS=$(date +%s)
echo "[$(date)] poll_siblings start. max_wait=${MAX_WAIT_S}s poll=${POLL_S}s" >> "$LOG"
while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TS))
  READY=0
  STATUS=""
  for s in "${SIBS[@]}"; do
    if [ -f "$WORKDIR/$s/NOTES.md" ]; then
      READY=$((READY+1)); STATUS="$STATUS $s:READY"
    else
      STATUS="$STATUS $s:WAIT"
    fi
  done
  echo "[$(date)] elapsed=${ELAPSED}s ready=${READY}/4${STATUS}" >> "$LOG"
  if [ "$READY" -eq 4 ]; then
    echo "ALL_READY" > "$READY_MARKER"
    echo "[$(date)] all 4 siblings READY -- exiting poll" >> "$LOG"
    exit 0
  fi
  if [ "$ELAPSED" -ge "$MAX_WAIT_S" ]; then
    echo "TIMEOUT $READY/4 ready" > "$READY_MARKER"
    echo "[$(date)] 4h timeout reached -- exiting poll with $READY/4 ready" >> "$LOG"
    exit 0
  fi
  sleep "$POLL_S"
done
