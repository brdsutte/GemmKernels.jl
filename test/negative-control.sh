#!/usr/bin/env bash
# Negative control: run the new tests against the code *without* their fix, to show that
# the tests catch the bugs. For each fix, this reverses only its source change (on top of
# the current branch, which has the working test setup), runs the matching test file,
# saves the log in negative-control/<name>.log, and restores src/ and configs/.
#
#     bash test/negative-control.sh                    all of them
#     bash test/negative-control.sh gett alpha         only the named ones
#
# Environment: JULIA_CMD (default "julia +1.11"), JULIA_CPU_THREADS (default 4),
# TIMEOUT per run in seconds (default 3600).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet -- src configs; then
    echo "src/ or configs/ has local changes; commit or stash them first."
    exit 1
fi

JULIA_CMD=${JULIA_CMD:-julia +1.11}
THREADS=${JULIA_CPU_THREADS:-4}
TIMEOUT=${TIMEOUT:-3600}
mkdir -p negative-control
trap 'git checkout -- src configs' EXIT

# name | commit to reverse, or "sed:<expression>" | file | test file | what should go wrong
CASES=(
    "gett|7428d7e|src/tensors/gettlayout.jl|gett-layout|misaligned-address crash or wrong results for PERM_M = [2, 1]"
    "swizzle|7ee2ea4|src/kernel.jl|contraction-configs|pipelined_ng fails with all 4 non-identity swizzles"
    "alpha|4017834|src/tensors/contraction.jl|contraction|the alpha test fails and the tropical test throws a MethodError"
    "defaults|65eb44a|src/tensors/gett.jl|contraction|every contraction with default parameters errors; the 3 descriptor cases fail"
    "rowmajor|09b5784|src/layout.jl|rowmajor-d|128x64 crashes (illegal address) and 64x128 gives wrong results, for all 3 kernels; 64x64 and the col-major size check pass"
    "cachekey|bb7599e|src/matmul.jl|kernel-cache|the three kernels load the same cached code"
    "overrides|e39b4f2|configs/configs.jl|tuning-overrides|the overrides stay switched on after planning"
    "leaky|0c6b262|configs/configs.jl|baseline|the baseline output lacks the final Leaky ReLU"
    "descriptors|sed:/^    check_gett_support(desc, β)\$/d|src/tensors/gett.jl|contraction|only the 3 descriptor cases fail, with wrong results (the default-parameter fix stays in place)"
)

selected=("$@")
for case in "${CASES[@]}"; do
    IFS='|' read -r name commit file tests expect <<< "$case"
    if [ ${#selected[@]} -gt 0 ] && [[ ! " ${selected[*]} " == *" $name "* ]]; then
        continue
    fi

    echo "== $name: reverting $file ($commit), running $tests"
    echo "   expected: $expect"
    if [[ $commit == sed:* ]]; then
        before=$(git hash-object "$file")
        sed -i "${commit#sed:}" "$file"
        if [ "$(git hash-object "$file")" = "$before" ]; then
            echo "   the sed expression changed nothing; skipping"
            continue
        fi
    elif ! git show "$commit" -- "$file" | git apply -R; then
        echo "   could not revert $commit; skipping"
        continue
    fi

    log="negative-control/$name.log"
    GEMMKERNELS_TESTS=$tests JULIA_CPU_THREADS=$THREADS \
        timeout "$TIMEOUT" $JULIA_CMD --project -e 'using Pkg; Pkg.test()' > "$log" 2>&1
    status=$?
    echo "   exit status $status (0 = all tests passed, which would mean the tests miss the bug)"
    grep -E "Test Summary|Error During Test|Test Failed|misaligned|MethodError|Some tests did not pass" "$log" \
        | head -n 15 | sed 's/^/   /'
    echo "   full log: $log"

    git checkout -- src configs
done
