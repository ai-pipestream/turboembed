#!/usr/bin/env bash
set -euo pipefail
main_binary=${1:?baseline test executable}
head_binary=${2:?candidate test executable}
cpu=${CPU:?set the CPU to pin both executables to}
rounds=${ROUNDS:-12}
printf 'baseline_sha=%s\ncandidate_sha=%s\n' "${MAIN_SHA:?baseline commit}" "${HEAD_SHA:?candidate commit}"
printf 'cpu=%s\nrustc=%s\n' "$cpu" "$(rustc --version)"
for tree in main head; do
    if [[ $tree == main ]]; then binary=$main_binary; else binary=$head_binary; fi
    read -r digest _ < <(sha256sum "$binary")
    printf '%s_binary_sha256=%s\n' "$tree" "$digest"
done
for corpus in ascii non_latin non_latin_nd; do
    for ((round=1;round<=rounds;round++)); do
        if ((round % 2)); then order=(main head); else order=(head main); fi
        for tree in "${order[@]}"; do
            if [[ $tree == main ]]; then binary=$main_binary; else binary=$head_binary; fi
            printf 'sample corpus=%s round=%s tree=%s load=%s\n' "$corpus" "$round" "$tree" "$(cat /proc/loadavg)"
            TURBO_BENCH_CORPUS=$corpus taskset -c "$cpu" "$binary" --ignored --exact paired_release --nocapture --test-threads=1
        done
    done
done
