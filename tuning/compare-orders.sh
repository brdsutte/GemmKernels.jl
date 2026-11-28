#!/usr/bin/env bash
# Compares search orders (GK_STRATEGY) on the GPU with equal budgets: every TC is tuned once
# per strategy and repetition, for the same time, in alternating order (so that drift, e.g.
# of the GPU's temperature, affects both alike), each into its own results file. Then
# tuning/compare-orders.jl compares them.
#
# Usage (from the repository): bash tuning/compare-orders.sh
#   GK_COMPARE_TCS         TCs to tune (default: 1,5,8,16,18,33,41,47, all held out while
#                          designing the cost model)
#   GK_COMPARE_STRATEGIES  default: random,model
#   GK_COMPARE_BUDGET      seconds of sweeping per TC and run (default 1800)
#   GK_COMPARE_REPS        repetitions (default 1)
#   GK_COMPARE_DIR         results directory (default tuning/orders)
#   GK_JULIA               julia command (default "julia +1.11")
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TCS="${GK_COMPARE_TCS:-1,5,8,16,18,33,41,47}"
STRATEGIES="${GK_COMPARE_STRATEGIES:-random,model}"
BUDGET="${GK_COMPARE_BUDGET:-1800}"
REPS="${GK_COMPARE_REPS:-1}"
DIR="$(realpath -m "${GK_COMPARE_DIR:-orders}")"
JULIA="${GK_JULIA:-julia +1.11}"
mkdir -p "$DIR"

IFS=, read -r -a tcs <<< "$TCS"
IFS=, read -r -a strategies <<< "$STRATEGIES"
step() { echo "== $(date '+%Y-%m-%d %H:%M:%S') $*"; }

n=0
for rep in $(seq 1 "$REPS"); do
    for tc in "${tcs[@]}"; do
        # alternate which strategy goes first
        order=("${strategies[@]}")
        if (( n % 2 == 1 )); then order=($(printf '%s\n' "${strategies[@]}" | tac)); fi
        n=$((n + 1))
        for s in "${order[@]}"; do
            out="$DIR/$s-tc$tc-r$rep.arrow"
            if [ -f "$out" ] && [ -f "$out.done" ]; then step "TC$tc $s r$rep: already done"; continue; fi
            rm -f "$out" "$out.version"
            step "TC$tc $s r$rep: tuning for $BUDGET s"
            GK_PROBLEMS=$tc GK_STRATEGY=$s GK_CONFIGS="$out" GK_EXHAUSTIVE=1 \
            GK_PROBLEM_TIME_LIMIT=$BUDGET GK_SWEEP_ONLY=1 GK_NO_SYSTEMD=1 \
                $JULIA --project --heap-size-hint=5G tune.jl > "$DIR/$s-tc$tc-r$rep.log" 2>&1 \
                || { step "TC$tc $s r$rep: tune.jl failed, see $DIR/$s-tc$tc-r$rep.log"; tail -20 "$DIR/$s-tc$tc-r$rep.log"; continue; }
            touch "$out.done"
            grep -E "best time|reached time limit|Cost model" "$DIR/$s-tc$tc-r$rep.log" | tail -3 || true
        done
    done
done

step "comparing"
$JULIA --project compare-orders.jl "$DIR" "$DIR/compare-orders.csv" | tee "$DIR/compare-orders.txt"
