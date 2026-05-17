#!/bin/bash
# Single iteration of the v5 overnight loop.
# Arg 1: iteration number (0-padded 2 digits)
# Arg 2: --solve-profile (fast|thorough)  -> maps to --max-outer-iters/--max-inner-iters
# Arg 3: --m-prior int                    -> maps to --target-pulls
set -u
ITER="$1"
PROFILE="${2:-thorough}"
M_PRIOR="${3:-26}"

REPO=/home/reidsurmeier/src/chuck-mcp-layering-lab
WORKDIR=$REPO/research/v5-overnight/loop-runner
JOB_DIR=$HOME/cnc-carving-jobs/emma-overnight-iter-${ITER}
ARTIFACTS=$JOB_DIR/artifacts
SHARE_DIR=/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-${ITER}
INPUT=/srv/woodblock-share/input-images/close_emma_2002_2048.jpg
LOG=$WORKDIR/iter_${ITER}.log
exec >>"$LOG" 2>&1

echo "==== iter ${ITER} start $(date) ===="
mkdir -p "$JOB_DIR" "$ARTIFACTS" "$SHARE_DIR"

cd "$REPO"

# venv
if [ -d "$REPO/.venv-renderer" ]; then
  source "$REPO/.venv-renderer/bin/activate"
elif [ -d "$REPO/.venv-v23" ]; then
  source "$REPO/.venv-v23/bin/activate"
fi
echo "python: $(which python) -- $(python --version 2>&1)"

# solve profile -> outer/inner iter counts
case "$PROFILE" in
  fast)     OUTER=1; INNER=10 ;;
  thorough) OUTER=3; INNER=25 ;;
  *)        OUTER=2; INNER=15 ;;
esac
PLATE_COUNT=20  # held constant per design; can be tuned later

T0=$(date +%s)

# 1. Run plan_emma end-to-end
python -m chuck_mcp_v2.plan_emma "$INPUT" \
  --output "$JOB_DIR/hybrid_result.json" \
  --plan-output "$JOB_DIR/production_plan.json" \
  --artifacts-dir "$ARTIFACTS" \
  --target-pulls "$M_PRIOR" \
  --max-outer-iters "$OUTER" \
  --max-inner-iters "$INNER" \
  --plate-count "$PLATE_COUNT" \
  --size 256 \
  --cells 96
PLAN_RC=$?
echo "plan_emma rc=$PLAN_RC"

T1=$(date +%s)
PLAN_WALL=$((T1 - T0))

# 2. Acceptance sheet
SHEET="$JOB_DIR/sheet_iter_${ITER}.png"
PYTHONPATH="$REPO/research/v4-build/example-harness:$PYTHONPATH" \
  python -m acceptance_harness "$ARTIFACTS" --output "$SHEET" --json > "$JOB_DIR/sheet_result.json" 2>&1
SHEET_RC=$?
echo "acceptance_harness rc=$SHEET_RC"

# Also try plan_dir = JOB_DIR (in case artifacts/ wasn't populated)
if [ ! -s "$SHEET" ]; then
  PYTHONPATH="$REPO/research/v4-build/example-harness:$PYTHONPATH" \
    python -m acceptance_harness "$JOB_DIR" --output "$SHEET" --json > "$JOB_DIR/sheet_result.json" 2>&1
  SHEET_RC=$?
  echo "acceptance_harness (retry on JOB_DIR) rc=$SHEET_RC"
fi

# 3. Build plan-dict and run validators
python "$WORKDIR/build_validator_plan.py" \
  --hybrid-result "$JOB_DIR/hybrid_result.json" \
  --production-plan "$JOB_DIR/production_plan.json" \
  --artifacts-dir "$ARTIFACTS" \
  --job-dir "$JOB_DIR" \
  --input-image "$INPUT" \
  --output "$JOB_DIR/validator_plan.json"
BUILDPLAN_RC=$?
echo "build_validator_plan rc=$BUILDPLAN_RC"

PYTHONPATH="$REPO/research/v3-construction/validators-reconstruction:$REPO:$PYTHONPATH" \
  python -m run_all_validators "$JOB_DIR/validator_plan.json" \
  --output "$JOB_DIR/validator_report.json" > "$JOB_DIR/validator_stdout.json" 2>&1
VALID_RC=$?
echo "run_all_validators rc=$VALID_RC"

# 4. Underlayer match
python "$WORKDIR/underlayer_match.py" \
  --artifacts-dir "$ARTIFACTS" \
  --job-dir "$JOB_DIR" \
  --reference /srv/woodblock-share/chuck-mcp-iterations/references/2026-05-16_user-annotated-emma-underlayer-methodology.png \
  --output "$JOB_DIR/underlayer_match.json"
ULM_RC=$?
echo "underlayer_match rc=$ULM_RC"

# 5. Compute metrics + append CSV row
python "$WORKDIR/append_row.py" \
  --iter "$ITER" \
  --wall "$PLAN_WALL" \
  --hybrid-result "$JOB_DIR/hybrid_result.json" \
  --validator-report "$JOB_DIR/validator_report.json" \
  --underlayer-match "$JOB_DIR/underlayer_match.json" \
  --sheet "$SHEET" \
  --csv "$WORKDIR/iterations.csv" \
  --notes "profile=$PROFILE m_prior=$M_PRIOR plan_rc=$PLAN_RC sheet_rc=$SHEET_RC valid_rc=$VALID_RC"
ROW_RC=$?
echo "append_row rc=$ROW_RC"

# 6. Mirror sheet to share dir
if [ -f "$SHEET" ]; then
  cp "$SHEET" "$SHARE_DIR/sheet_iter_${ITER}.png"
  echo "copied sheet to $SHARE_DIR/sheet_iter_${ITER}.png"
fi

T2=$(date +%s)
echo "==== iter ${ITER} done in $((T2 - T0))s ===="
exit 0
