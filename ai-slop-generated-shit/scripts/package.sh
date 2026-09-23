#!/usr/bin/env bash
# Build a Turbo distribution archive for the current machine's target triple:
# dist/turbo-<version>-<target>.tar.gz, containing libturbo, the headers,
# turbo-bundle, and every provider library this machine can produce.
#
# Usage: scripts/package.sh [--no-cuda] [--no-openvino] [--no-hailo] [--no-ggml]
#                           [--use-prebuilt-openvino] [--use-prebuilt-hailo]
#
# A provider whose toolchain is not installed is recorded as absent and the
# archive is still built. A provider whose toolchain IS installed and whose
# build then fails stops the script: the caller asked for that provider by
# not passing its --no-* flag.
#
# Everything the script writes lives under dist/ (the archive) and target/
# (build output and a staging tree); nothing else in the repository is
# touched. See docs/packaging.md for the archive layout and how to consume
# it.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

usage="usage: scripts/package.sh [--no-cuda] [--no-openvino] [--no-hailo] [--no-ggml] [--use-prebuilt-openvino] [--use-prebuilt-hailo]"
no_cuda=0
no_openvino=0
no_hailo=0
no_ggml=0
use_prebuilt_openvino=0
use_prebuilt_hailo=0
cuda_ok=0
ov_ok=0
hailo_ok=0
ggml_ok=0
for arg in "$@"; do
    case "$arg" in
        --no-cuda) no_cuda=1 ;;
        --no-openvino) no_openvino=1 ;;
        --no-hailo) no_hailo=1 ;;
        --no-ggml) no_ggml=1 ;;
        --use-prebuilt-openvino) use_prebuilt_openvino=1 ;;
        --use-prebuilt-hailo) use_prebuilt_hailo=1 ;;
        *)
            echo "package.sh: unknown argument: $arg" >&2
            echo "$usage" >&2
            exit 2
            ;;
    esac
done
[[ $no_openvino -eq 1 && $use_prebuilt_openvino -eq 1 ]] \
    && { echo "package.sh: --no-openvino and --use-prebuilt-openvino contradict each other" >&2; exit 2; }
[[ $no_hailo -eq 1 && $use_prebuilt_hailo -eq 1 ]] \
    && { echo "package.sh: --no-hailo and --use-prebuilt-hailo contradict each other" >&2; exit 2; }

log()  { printf '==> %s\n' "$*" >&2; }
warn() { printf 'package.sh: warning: %s\n' "$*" >&2; }
die()  { printf 'package.sh: error: %s\n' "$*" >&2; exit 1; }

# --- version and target triple -----------------------------------------
version="$(awk '
    /^\[workspace\.package\]/ { inpkg = 1; next }
    /^\[/ { inpkg = 0 }
    inpkg && /^version[[:space:]]*=/ {
        match($0, /"[^"]*"/)
        print substr($0, RSTART + 1, RLENGTH - 2)
        exit
    }
' Cargo.toml)"
[[ -n "$version" ]] || die "could not read workspace.package.version from Cargo.toml"

target="$(rustc -vV | awk '/^host:/ { print $2 }')"
[[ -n "$target" ]] || die "could not determine the target triple from 'rustc -vV'"

# Everything below assumes the ELF/.so layout and the GNU loader: the library
# names, the providers/*.so glob, and the ldd gate. Say so here rather than
# reporting a missing libturbo.so on a machine that built libturbo.dylib.
case "$target" in
    *-linux-*) ;;
    *) die "package.sh builds Linux archives only today; this machine's target is $target. The script looks for .so files and gates them with ldd, neither of which applies here (see docs/packaging.md)." ;;
esac

name="turbo-${version}-${target}"
dist_dir="$root/dist"
stage="$root/target/package/$name"
relq="$root/target/release"

log "packaging $name"
rm -rf "$stage"
mkdir -p "$stage/lib" "$stage/include/turbo" "$stage/bin" "$stage/providers" "$dist_dir"

present=()
absent=()

# --- required: libturbo, headers, turbo-bundle --------------------------
log "building libturbo (turbo-shared), turbo-bundle, turbo-bench, and the mock/static providers"
cargo build --release -p turbo-shared -p turbo-provider-mock -p turbo-provider-static -p turbo-bundle -p turbo-bench

[[ -f "$relq/libturbo.so" ]] || die "missing $relq/libturbo.so (required); cargo build --release -p turbo-shared failed"
[[ -f "$root/include/turbo/turbo.h" && -f "$root/include/turbo/turbo_types.h" && -f "$root/include/turbo/turbo_provider.h" ]] \
    || die "missing include/turbo/*.h (required); run scripts/gen-header.sh"
[[ -f "$relq/turbo-bundle" ]] || die "missing $relq/turbo-bundle (required); cargo build --release -p turbo-bundle failed"

cp "$relq/libturbo.so" "$stage/lib/"
present+=("lib/libturbo.so")
if [[ -f "$relq/libturbo.a" ]]; then
    cp "$relq/libturbo.a" "$stage/lib/"
    present+=("lib/libturbo.a")
else
    absent+=("lib/libturbo.a -- turbo-shared did not produce a staticlib on this build")
fi

cp "$root"/include/turbo/*.h "$stage/include/turbo/"
present+=("include/turbo/turbo.h" "include/turbo/turbo_types.h" "include/turbo/turbo_provider.h")

cp "$relq/turbo-bundle" "$stage/bin/"
present+=("bin/turbo-bundle")
[[ -f "$relq/turbo-bench" ]] || die "missing $relq/turbo-bench (required); cargo build --release -p turbo-bench failed"
cp -L "$relq/turbo-bench" "$stage/bin/"
present+=("bin/turbo-bench")

# --- required provider: mock --------------------------------------------
[[ -f "$relq/libturbo_provider_mock.so" ]] || die "missing $relq/libturbo_provider_mock.so (required)"
cp "$relq/libturbo_provider_mock.so" "$stage/providers/"
present+=("providers/libturbo_provider_mock.so")

# --- optional provider: static -------------------------------------------
if [[ -f "$relq/libturbo_provider_static.so" ]]; then
    cp "$relq/libturbo_provider_static.so" "$stage/providers/"
    present+=("providers/libturbo_provider_static.so")
else
    absent+=("providers/libturbo_provider_static.so -- not produced as a cdylib on this build")
fi

# --- optional provider: cuda ---------------------------------------------
# The same search providers/cuda/build.rs does: CUDA_PATH, then the two
# usual install prefixes. An absent toolkit is a legitimate reason to ship
# without the provider; a present toolkit plus a failing build is not.
cuda_home=""
if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    cuda_home="$CUDA_PATH"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    cuda_home="/usr/local/cuda"
elif [[ -x /usr/bin/nvcc ]]; then
    cuda_home="/usr"
fi
cuda_reason=""
if [[ $no_cuda -eq 1 ]]; then
    log "skipping the CUDA provider build (--no-cuda)"
    cuda_reason="skipped by --no-cuda"
elif [[ -z "$cuda_home" ]]; then
    log "no CUDA toolkit found (no nvcc in CUDA_PATH, /usr/local/cuda, or /usr); packaging without the CUDA provider"
    cuda_reason="no CUDA toolkit on this machine (no nvcc in CUDA_PATH, /usr/local/cuda, or /usr)"
else
    log "building the CUDA provider (turbo-provider-cuda) against the toolkit at $cuda_home"
    cargo build --release -p turbo-provider-cuda \
        || die "the CUDA toolkit at $cuda_home is installed but turbo-provider-cuda failed to build (see above); pass --no-cuda to package without it"
    cuda_ok=1
fi
if [[ $cuda_ok -eq 1 ]]; then
    [[ -f "$relq/libturbo_provider_cuda.so" ]] \
        || die "turbo-provider-cuda built but $relq/libturbo_provider_cuda.so is missing"
    cp -L "$relq/libturbo_provider_cuda.so" "$stage/providers/"
    present+=("providers/libturbo_provider_cuda.so")
    for extra in libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
        [[ -f "$relq/$extra" ]] \
            || die "turbo-provider-cuda built but $relq/$extra is missing; the ONNX Runtime CUDA execution provider is part of it"
        cp -L "$relq/$extra" "$stage/providers/"
        present+=("providers/$extra")
    done
else
    absent+=("providers/libturbo_provider_cuda.so -- $cuda_reason")
    absent+=("providers/libonnxruntime_providers_cuda.so -- $cuda_reason")
    absent+=("providers/libonnxruntime_providers_shared.so -- $cuda_reason")
fi

# --- optional provider: ggml ----------------------------------------------
# A plain workspace member that CI builds and unit-tests, so it is packaged
# like any other provider. llama.cpp is compiled from source, which needs
# cmake; without cmake it is named as absent rather than silently missing.
ggml_reason=""
if [[ $no_ggml -eq 1 ]]; then
    log "skipping the ggml provider build (--no-ggml)"
    ggml_reason="skipped by --no-ggml"
elif ! command -v cmake >/dev/null; then
    log "no cmake found; packaging without the ggml provider (llama.cpp is built from source with cmake)"
    ggml_reason="no cmake on this machine (llama.cpp is built from source with cmake)"
else
    log "building the ggml provider (turbo-provider-ggml; llama.cpp is compiled from source)"
    cargo build --release -p turbo-provider-ggml \
        || die "cmake is installed but turbo-provider-ggml failed to build (see above); pass --no-ggml to package without it"
    ggml_ok=1
fi
if [[ $ggml_ok -eq 1 ]]; then
    [[ -f "$relq/libturbo_provider_ggml.so" ]] \
        || die "turbo-provider-ggml built but $relq/libturbo_provider_ggml.so is missing"
    cp -L "$relq/libturbo_provider_ggml.so" "$stage/providers/"
    present+=("providers/libturbo_provider_ggml.so")
else
    absent+=("providers/libturbo_provider_ggml.so -- $ggml_reason")
fi

# --- optional provider: openvino ------------------------------------------
# An existing build/openvino tree is rebuilt, not trusted: cmake is
# incremental, so the rebuild is cheap, and a tree left over from an older
# providers/openvino/src still loads and would ship silently stale.
# --use-prebuilt-openvino is the explicit way to package it as it is.
ov_lib="$root/build/openvino/libturbo_provider_openvino.so"
ov_reason=""
if [[ $no_openvino -eq 1 ]]; then
    log "skipping the OpenVINO provider build (--no-openvino)"
    ov_reason="skipped by --no-openvino"
elif [[ $use_prebuilt_openvino -eq 1 ]]; then
    [[ -f "$ov_lib" ]] || die "--use-prebuilt-openvino was passed but there is no build at $ov_lib"
    log "packaging the existing OpenVINO provider build at build/openvino unchecked (--use-prebuilt-openvino)"
    ov_ok=1
else
    ov_dir="${TURBO_OPENVINO_DIR:-}"
    if [[ -z "$ov_dir" ]]; then
        for cand in "$HOME"/opt/openvino_genai_* "$HOME"/opt/openvino_*; do
            [[ -d "$cand/runtime/cmake" ]] && { ov_dir="$cand"; break; }
        done
    fi
    if [[ -n "$ov_dir" && -d "$ov_dir/runtime/cmake" ]]; then
        log "building the OpenVINO provider against $ov_dir"
        cmake -S "$root/providers/openvino" -B "$root/build/openvino" \
                -DOpenVINO_DIR="$ov_dir/runtime/cmake" -DCMAKE_BUILD_TYPE=Release >&2 \
            && cmake --build "$root/build/openvino" -j >&2 \
            || die "the OpenVINO SDK at $ov_dir is installed but the provider failed to build (see above); pass --no-openvino to package without it, or --use-prebuilt-openvino to package build/openvino as it is"
        ov_ok=1
    else
        log "no OpenVINO SDK found (set TURBO_OPENVINO_DIR); packaging without the OpenVINO provider"
        ov_reason="no OpenVINO SDK on this machine (set TURBO_OPENVINO_DIR)"
    fi
fi
if [[ $ov_ok -eq 1 ]]; then
    [[ -f "$ov_lib" ]] || die "the OpenVINO provider build reported success but $ov_lib is missing"
    cp -L "$ov_lib" "$stage/providers/"
    present+=("providers/libturbo_provider_openvino.so")
else
    absent+=("providers/libturbo_provider_openvino.so -- $ov_reason")
fi

# --- optional provider: hailo ---------------------------------------------
# Same rule as OpenVINO: rebuild when HailoRT is present, and require
# --use-prebuilt-hailo to package build/hailo without a rebuild.
hailo_lib="$root/build/hailo/libturbo_provider_hailo.so"
hailo_reason=""
if [[ $no_hailo -eq 1 ]]; then
    log "skipping the Hailo provider build (--no-hailo)"
    hailo_reason="skipped by --no-hailo"
elif [[ $use_prebuilt_hailo -eq 1 ]]; then
    [[ -f "$hailo_lib" ]] || die "--use-prebuilt-hailo was passed but there is no build at $hailo_lib"
    log "packaging the existing Hailo provider build at build/hailo unchecked (--use-prebuilt-hailo)"
    hailo_ok=1
elif [[ -f /usr/include/hailo/hailort.h || -n "${HAILORT_INCLUDE_DIR:-}" ]]; then
    log "building the Hailo provider"
    cmake -S "$root/providers/hailo" -B "$root/build/hailo" -DCMAKE_BUILD_TYPE=Release \
            ${HAILORT_INCLUDE_DIR:+-DHAILORT_INCLUDE_DIR="$HAILORT_INCLUDE_DIR"} \
            ${HAILORT_LIBRARY:+-DHAILORT_LIBRARY="$HAILORT_LIBRARY"} >&2 \
        && cmake --build "$root/build/hailo" -j >&2 \
        || die "HailoRT headers are installed but the Hailo provider failed to build (see above); pass --no-hailo to package without it, or --use-prebuilt-hailo to package build/hailo as it is"
    hailo_ok=1
else
    log "no HailoRT headers found; packaging without the Hailo provider"
    hailo_reason="no HailoRT headers on this machine (set HAILORT_INCLUDE_DIR)"
fi
if [[ $hailo_ok -eq 1 ]]; then
    [[ -f "$hailo_lib" ]] || die "the Hailo provider build reported success but $hailo_lib is missing"
    cp -L "$hailo_lib" "$stage/providers/"
    present+=("providers/libturbo_provider_hailo.so")
else
    absent+=("providers/libturbo_provider_hailo.so -- $hailo_reason")
fi

# --- license / notice -----------------------------------------------------
[[ -f "$root/LICENSE" ]] || die "missing LICENSE (required)"
cp "$root/LICENSE" "$stage/"
present+=("LICENSE")
if [[ -f "$root/NOTICE" ]]; then
    cp "$root/NOTICE" "$stage/"
    present+=("NOTICE")
fi

# --- ldd check --------------------------------------------------------
# Reports unresolved shared-library dependencies for every .so in the
# package. libturbo.so and the mock provider must additionally depend on
# nothing beyond the baseline C/C++ runtime; anything else (a stray dynamic
# dependency, or a dependency ldd cannot resolve) fails the build. Every
# other provider is reported only: providers such as cuda and openvino
# legitimately depend on vendor runtimes that are not part of this package
# (see the "not bundled" section of the generated package README).
baseline="libc.so.6 libm.so.6 libgcc_s.so.1 libstdc++.so.6 libpthread.so.0 libdl.so.2"
# The archive's glibc floor, when one is declared (the container build sets
# it to the manylinux_2_28 base). Sonames alone do not show a dependency on
# a newer symbol version, so each library's newest GLIBC_* symbol version is
# checked against it. Unset means "report the number, gate nothing", which
# is what a host build can honestly say.
glibc_floor="${TURBO_GLIBC_FLOOR:-}"
ldd_failed=0
# The newest GLIBC_<version> a library's dynamic symbol table references,
# reported for every packaged library and gated when a floor is declared.
check_symbol_versions() {
    local lib="$1" newest
    command -v objdump >/dev/null \
        || die "objdump is required to report the glibc symbol versions of $lib (install binutils)"
    newest="$(objdump -T "$lib" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -V | tail -1)"
    if [[ -z "$newest" ]]; then
        log "$(basename "$lib"): no versioned glibc symbols"
        return
    fi
    log "$(basename "$lib"): newest glibc symbol GLIBC_$newest${glibc_floor:+ (floor GLIBC_$glibc_floor)}"
    if [[ -n "$glibc_floor" && "$(printf '%s\n%s\n' "$glibc_floor" "$newest" | sort -V | tail -1)" != "$glibc_floor" ]]; then
        warn "$(basename "$lib") needs GLIBC_$newest, above this archive's floor of GLIBC_$glibc_floor"
        ldd_failed=1
    fi
}
check_lib() {
    local lib="$1" strict="$2"
    local out unresolved=() extra=() dep
    # A gate whose whole job is to fail must not pass because the tool it
    # runs failed: ldd's status is the first thing checked.
    if ! out="$(ldd "$lib" 2>&1)"; then
        die "ldd failed on $lib: $out"
    fi
    if [[ -z "${out//[[:space:]]/}" ]]; then
        die "ldd produced no output for $lib; the dependency gate cannot pass on nothing"
    fi
    check_symbol_versions "$lib"
    log "ldd $(basename "$lib"):"
    printf '%s\n' "$out" | sed 's/^/    /' >&2
    while read -r dep; do
        [[ -z "$dep" ]] && continue
        unresolved+=("$dep")
    done < <(printf '%s\n' "$out" | awk '/not found/ { print $1 }')
    if [[ "$strict" -eq 1 ]]; then
        while read -r dep; do
            [[ -z "$dep" ]] && continue
            local base
            base="$(basename "$dep")"
            case "$base" in
                ld-linux*.so*|linux-vdso.so*|linux-gate.so*) ;;
                *)
                    case " $baseline " in
                        *" $base "*) ;;
                        *) extra+=("$dep") ;;
                    esac
                    ;;
            esac
        done < <(printf '%s\n' "$out" | awk '{ print $1 }' | grep -v '^$')
        if [[ ${#unresolved[@]} -gt 0 || ${#extra[@]} -gt 0 ]]; then
            warn "$(basename "$lib") must depend on nothing beyond $baseline"
            [[ ${#unresolved[@]} -gt 0 ]] && warn "  unresolved: ${unresolved[*]}"
            [[ ${#extra[@]} -gt 0 ]] && warn "  unexpected dependency: ${extra[*]}"
            ldd_failed=1
        fi
    fi
}
check_lib "$stage/lib/libturbo.so" 1
check_lib "$stage/providers/libturbo_provider_mock.so" 1
for lib in "$stage"/lib/*.so "$stage"/providers/*.so; do
    case "$lib" in
        "$stage/lib/libturbo.so"|"$stage/providers/libturbo_provider_mock.so") continue ;;
    esac
    [[ -f "$lib" ]] && check_lib "$lib" 0
done
[[ $ldd_failed -eq 0 ]] || die "a packaged library has an unresolved or unexpected dependency, or a glibc symbol above this archive's floor (see above)"

# --- package README -------------------------------------------------------
list_or_none() {
    if [[ $# -eq 0 ]]; then
        printf '(none)\n'
    else
        local item
        for item in "$@"; do printf -- '- %s\n' "$item"; done
    fi
}

# Provenance: the one thing AGENTS.md's "name the machine" rule needs. A
# failed command substitution inside a here-document does not abort under
# set -e, so it is resolved and checked before the heredoc runs.
host="$(hostname 2>/dev/null || uname -n 2>/dev/null || true)"
[[ -n "$host" ]] || die "could not determine this machine's name (hostname and uname -n both failed); the archive README would have no provenance"

# The inventory below is the archive's own contents, so it has to include
# the two files written after the providers: this README and SHA256SUMS.
present+=("README.md" "SHA256SUMS")

# The vendor-runtime notes describe providers; they are emitted only when
# that provider is actually in this archive.
cuda_note=""
if [[ $cuda_ok -eq 1 ]]; then
    cuda_note="$(cat <<'CUDA_NOTE'


The CUDA user-space libraries the CUDA provider needs are never bundled: the
ONNX Runtime execution provider links the CUDA 13 set (cuBLAS, cuBLASLt,
cuDNN 9, NVRTC, the CUDA 13 runtime), and the provider's own kernels link the
`libcudart` of the toolkit that compiled them (see the `ldd` output in the
build log). They are large, GPU-driver-version-sensitive, and already managed
by whatever CUDA install or NVIDIA PyPI wheel set the target machine uses.
Point the provider at them with the `TURBO_CUDA_LIB_DIR` environment variable
(or the context option `cuda_lib_dir`); the provider preloads every `lib*.so*`
in that directory before creating a session. A missing library there is
`TURBO_E_DEVICE_UNAVAILABLE` at context creation, never a silent fallback to
CPU.
CUDA_NOTE
)"
fi

ov_note=""
if [[ $ov_ok -eq 1 ]]; then
    ov_note="$(cat <<'OV_NOTE'


The OpenVINO runtime (`libopenvino.so`, its plugins, and the bundled TBB) is
never bundled either: it is a multi-hundred-megabyte SDK with its own
versioned install layout, and the provider library's RUNPATH already points
at the OpenVINO tree it was built against. Set `LD_LIBRARY_PATH` to that
tree's `runtime/lib/intel64` (and its `runtime/3rdparty` TBB directory) if
the provider was moved off the machine it was built on.
OV_NOTE
)"
fi

env_notes="- \`LD_LIBRARY_PATH\`: must include this archive's \`lib/\` directory for any
  binary linked against \`libturbo.so\`."
if [[ $cuda_ok -eq 1 ]]; then
    env_notes="- \`TURBO_CUDA_LIB_DIR\`: directory holding the CUDA 13 user-space libraries
  (cuBLAS, cuBLASLt, cuDNN 9, NVRTC, the CUDA runtime) for the CUDA provider.
$env_notes"
fi
if [[ $ov_ok -eq 1 ]]; then
    env_notes="$env_notes It must also include the OpenVINO runtime's
  \`runtime/lib/intel64\` (plus its TBB directory) when using the OpenVINO
  provider away from its build machine."
fi

cat > "$stage/README.md" <<EOF
# Turbo $version ($target)

This archive is a Turbo distribution built on $host for $target. It is
consumed outside the TurboEmbed repository; see \`docs/packaging.md\` in the
repository for the full design. Load a provider with
\`turbo_runtime_load_provider\` (\`include/turbo/turbo_provider.h\`), passing
the path to one of the \`providers/*.so\` files below, or list every provider
directory in \`turbo_runtime_desc.provider_paths\` at \`turbo_runtime_create\`.

## What is in this archive

$(list_or_none "${present[@]}")

## What is not in this archive (and why)

$(list_or_none "${absent[@]}")$cuda_note$ov_note

## Environment variables a consumer may need

$env_notes

## Using it from C

\`\`\`c
#include "turbo/turbo.h"
/* cc ... -Iinclude -Llib -lturbo -Wl,-rpath,lib */
\`\`\`

See \`include/turbo/turbo.h\` for the full C ABI and \`bin/turbo-bundle\` for
importing, verifying, and inspecting model bundles.
EOF

# --- checksums --------------------------------------------------------
log "computing SHA256SUMS"
(
    cd "$stage"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | sed -z 's#^\./##' \
        | xargs -0 sha256sum > SHA256SUMS
)

# --- archive ------------------------------------------------------------
archive="$dist_dir/$name.tar.gz"
rm -f "$archive"
tar -czf "$archive" -C "$stage" .
log "wrote $archive"

# --- verify: extract, compile smoke.c, run it ----------------------------
verify_dir="$root/target/package/verify-$name"
rm -rf "$verify_dir"
mkdir -p "$verify_dir"
tar -xzf "$archive" -C "$verify_dir"

log "verifying the archive: compiling and running the C smoke test against it"
cc="${CC:-cc}"
smoke_bin="$root/target/package/turbo-c-smoke-$name"
"$cc" -std=c11 -Wall -Wextra -Wpedantic -Werror -I"$verify_dir/include" \
    "$root/crates/turbo-conformance/c/smoke.c" -L"$verify_dir/lib" -lturbo -lm \
    -Wl,-rpath,"$verify_dir/lib" -o "$smoke_bin"

ld_path="$verify_dir/lib"
[[ -n "${LD_LIBRARY_PATH:-}" ]] && ld_path="$ld_path:$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="$ld_path" "$smoke_bin" "$root/testdata/bundles/mock"

# Every packaged provider must load from the extracted archive: the survey
# dlopens each library, and --strict fails on any that does not load. The
# runtimes the providers link (OpenVINO, CUDA) come from LD_LIBRARY_PATH as
# they would on a consumer machine.
log "verifying the archive: loading every packaged provider"
LD_LIBRARY_PATH="$ld_path" "$relq/turbo-bench" discover --provider-dir "$verify_dir/providers" --strict >&2 \
    || die "a packaged provider does not load from the extracted archive (see above)"

size_bytes="$(stat -c '%s' "$archive" 2>/dev/null || stat -f '%z' "$archive")"
size_human="$(numfmt --to=iec --suffix=B "$size_bytes" 2>/dev/null || echo "${size_bytes} bytes")"
log "archive: $archive ($size_human, $size_bytes bytes)"
