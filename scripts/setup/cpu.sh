#!/usr/bin/env bash
# The CPU reference backend: always built, nothing beyond Rust and a C
# compiler (docs/setup/cpu.md).
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
case "${TURBO_SETUP_ARGS[0]:-}" in -h|--help) usage_common ""; exit 0 ;; "") ;; *) usage_common ""; exit 2 ;; esac

echo "TurboEmbed setup: cpu ($(os_id) $(os_version), $(uname -m))"
check_rust
check_c_compiler

section "Processor"
if [ -r /proc/cpuinfo ]; then
    flags=$(grep -m1 '^flags' /proc/cpuinfo)
    case "$flags" in
        *avx512f*avx512vl*|*avx512vl*avx512f*) ok "AVX-512 F and VL: the AVX-512 kernels run" ;;
        *avx2*fma*|*fma*avx2*) ok "AVX2 and FMA: the AVX2 kernels run" ;;
        *) ok "no AVX2: the portable kernels run (correct, slower)" ;;
    esac
elif [ "$(uname -s)" = Darwin ]; then
    ok "$(sysctl -n machdep.cpu.brand_string 2>/dev/null): the portable kernels run on arm64"
else
    ok "$(uname -m): the portable kernels run"
fi
if have nproc; then note "threads a session starts: $(nproc) (TURBO_CPU_THREADS sets another count)"; fi

finish
