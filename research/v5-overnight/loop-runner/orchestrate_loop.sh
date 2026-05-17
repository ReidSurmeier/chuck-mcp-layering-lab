#!/bin/bash
# Orchestrate up to 12 iterations after SIBLINGS_READY marker exists.
# Runs each iter, checks acceptance, applies adaptive nudges, commits per iter.
set -u
REPO=/home/reidsurmeier/src/chuck-mcp-layering-lab
WORKDIR=$REPO/research/v5-overnight/loop-runner
LOG=$WORKDIR/orchestrate.log
exec >>"$LOG" 2>&1

echo "==== orchestrate start $(date) ===="

# defaults
PROFILE="thorough"
M_PRIOR=26
MAX_ITERS=12
ACCEPT_DE=8.0
ACCEPT_UNDERLAYER=85.0
ACCEPT_VALIDATORS=6

# track best-so-far and convergence
BEST_DE=99999
BEST_VAL=0
BEST_ULM=0
LAST_DE=""
LAST_LAST_DE=""
LAST_LAST_LAST_DE=""
NUDGE_COUNT=0

# pre-loop catch-up: pull sibling commits
cd "$REPO"
git pull --rebase origin main 2>&1 | tail -5

for i in $(seq 1 "$MAX_ITERS"); do
  ITER=$(printf '%02d' "$i")
  echo "---- iter $ITER start $(date) profile=$PROFILE m_prior=$M_PRIOR ----"

  # pull again to catch concurrent fixes
  git pull --rebase origin main 2>&1 | tail -3

  bash "$WORKDIR/run_iter.sh" "$ITER" "$PROFILE" "$M_PRIOR"
  RC=$?
  echo "run_iter rc=$RC"

  # parse last row of csv
  ROW=$(tail -n1 "$WORKDIR/iterations.csv")
  echo "row: $ROW"
  # fields: iter_n,wall_s,plate_count,plates_pass_pnc,dE_mean,dE_p95,validators_passed,underlayer_match_pct,sheet_path,notes
  DE=$(echo "$ROW" | awk -F, '{print $5}')
  VAL=$(echo "$ROW" | awk -F, '{print $7}')
  ULM=$(echo "$ROW" | awk -F, '{print $8}')

  # track best
  python -c "import sys; de=float(sys.argv[1] or 99999); best=float(sys.argv[2]); sys.exit(0 if de<best else 1)" "$DE" "$BEST_DE" \
    && BEST_DE="$DE"
  python -c "import sys; v=int(sys.argv[1] or 0); best=int(sys.argv[2]); sys.exit(0 if v>best else 1)" "$VAL" "$BEST_VAL" \
    && BEST_VAL="$VAL"
  python -c "import sys; u=float(sys.argv[1] or 0); best=float(sys.argv[2]); sys.exit(0 if u>best else 1)" "$ULM" "$BEST_ULM" \
    && BEST_ULM="$ULM"

  # commit progress
  cd "$REPO"
  git add research/v5-overnight/loop-runner/iterations.csv research/v5-overnight/loop-runner/*.log 2>/dev/null
  git commit -m "v5 overnight iter $ITER: dE $DE, validators $VAL/6, underlayer ${ULM}%" 2>&1 | tail -3 || true
  git push origin main 2>&1 | tail -3 || true

  # acceptance check
  ACCEPT=$(python -c "
import sys
de=float(sys.argv[1] or 99999); v=int(sys.argv[2] or 0); u=float(sys.argv[3] or 0)
ok = (v >= int(sys.argv[4])) and (de < float(sys.argv[5])) and (u >= float(sys.argv[6]))
print('YES' if ok else 'NO')
" "$DE" "$VAL" "$ULM" "$ACCEPT_VALIDATORS" "$ACCEPT_DE" "$ACCEPT_UNDERLAYER")
  echo "accept_check: $ACCEPT (de=$DE val=$VAL ulm=$ULM)"

  if [ "$ACCEPT" = "YES" ]; then
    echo "==== ACCEPTED at iter $ITER ===="
    break
  fi

  # convergence stall detection -- 3 iters dE delta < 0.5
  if [ -n "$LAST_LAST_DE" ] && [ -n "$LAST_LAST_LAST_DE" ]; then
    STALL=$(python -c "
import sys
des = [float(x or 99999) for x in sys.argv[1:]]
deltas = [abs(des[i+1]-des[i]) for i in range(len(des)-1)]
print('YES' if all(d < 0.5 for d in deltas) else 'NO')
" "$LAST_LAST_LAST_DE" "$LAST_LAST_DE" "$LAST_DE" "$DE")
    if [ "$STALL" = "YES" ] && [ "$NUDGE_COUNT" -lt 3 ]; then
      NUDGE_COUNT=$((NUDGE_COUNT+1))
      # cycle: bump m_prior up 2 / swap profile / reset
      case $((NUDGE_COUNT % 3)) in
        1) M_PRIOR=$((M_PRIOR + 2)); echo "STALL nudge: m_prior -> $M_PRIOR" ;;
        2) [ "$PROFILE" = "thorough" ] && PROFILE="fast" || PROFILE="thorough"
           echo "STALL nudge: profile -> $PROFILE" ;;
        0) M_PRIOR=26; PROFILE="thorough"; echo "STALL nudge: reset to defaults" ;;
      esac
    fi
  fi
  LAST_LAST_LAST_DE="$LAST_LAST_DE"
  LAST_LAST_DE="$LAST_DE"
  LAST_DE="$DE"
done

echo "==== orchestrate end $(date) ===="
# stash final state for the report writer
cat > "$WORKDIR/FINAL_STATS.json" <<EOF
{
  "best_dE_mean": ${BEST_DE},
  "best_validators": ${BEST_VAL},
  "best_underlayer_match_pct": ${BEST_ULM},
  "iters_run": ${i:-0},
  "max_iters": ${MAX_ITERS}
}
EOF
echo "wrote FINAL_STATS.json"
