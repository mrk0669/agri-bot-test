#!/usr/bin/env bash
# ============================================================================
#  Run every check in docs/VERIFICATION.md, in order, and stop on the first
#  failure. This is the command to run before a competition, after any change,
#  and in CI.
#
#      bash verify.sh
#
#  Exit code 0 means every claim in the verification document still holds.
# ============================================================================
set -uo pipefail

PYTHON="${PYTHON:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")"

PASSED=0
FAILED=0
FAILED_NAMES=()

step() {
    local name="$1"; shift
    printf '\n\033[1;34m==> %s\033[0m\n' "${name}"
    if "$@"; then
        PASSED=$((PASSED + 1))
        printf '\033[1;32m    PASS\033[0m  %s\n' "${name}"
    else
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("${name}")
        printf '\033[1;31m    FAIL\033[0m  %s\n' "${name}"
    fi
}

printf '\033[1m AgriBot verification \033[0m\n'
printf ' python: %s\n' "$(${PYTHON} --version 2>&1)"

step "Preflight configuration checks" \
    ${PYTHON} -m agribot.app.preflight --skip-hardware

step "Test suite" \
    ${PYTHON} -m pytest -q

step "Sensor-fusion study (Section 5.3) across 8 seeds" \
    ${PYTHON} tools/kalman_sim.py --sweep 8 --no-figure

step "Navigation controller settles below 1 mm" \
    ${PYTHON} tools/tune_pid.py --check

step "Perception throughput has margin at cruise speed" \
    ${PYTHON} tools/bench_perception.py --frames 120

# The simulator exits non-zero if any crop was sprayed.
step "Software-in-the-loop mission, one row" \
    ${PYTHON} -m agribot.app.simulate --seconds 45 --rows 1 --quiet --report

step "Software-in-the-loop mission, two rows with a turn" \
    ${PYTHON} -m agribot.app.simulate --seconds 120 --rows 2 --quiet --report

printf '\n============================================================\n'
if [[ ${FAILED} -eq 0 ]]; then
    printf '\033[1;32m  ALL %d CHECKS PASSED\033[0m\n' "${PASSED}"
    printf '============================================================\n'
    exit 0
fi
printf '\033[1;31m  %d of %d CHECKS FAILED\033[0m\n' "${FAILED}" "$((PASSED + FAILED))"
for name in "${FAILED_NAMES[@]}"; do
    printf '    - %s\n' "${name}"
done
printf '============================================================\n'
exit 1
