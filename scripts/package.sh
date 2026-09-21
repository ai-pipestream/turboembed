#!/usr/bin/env bash
# Build a Turbo distribution archive for the current machine's target triple:
# dist/turbo-<version>-<target>.tar.gz, containing libturbo, the headers,
# turbo-bundle, and every provider library this machine can produce.
#
# Usage: scripts/package.sh [--no-cuda] [--no-openvino]
#
# Everything the script writes lives under dist/ (the archive) and target/
# (build output and a staging tree); nothing else in the repository is
# touched. See docs/packaging.md for the archive layout and how to consume
# it.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

no_cuda=0
no_openvino=0
for arg in "$@"; do
    case "$arg" in
        --no-cuda) no_cuda=1 ;;
        --no-openvino) no_openvino=1 ;;
        *)
            echo "package.sh: unknown argument: $arg" >&2
            echo "usage: scripts/package.sh [--no-cuda] [--no-openvino]" >&2
            exit 2
            ;;
    esac
done

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
log "building libturbo (turbo-shared), turbo-bundle, and the mock/static providers"
cargo build --release -p turbo-shared -p turbo-provider-mock -p turbo-provider-static -p turbo-bundle

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
if [[ $no_cuda -eq 1 ]]; then
    log "skipping the CUDA provider build (--no-cuda)"
else
    log "building the CUDA provider (turbo-provider-cuda)"
    if ! cargo build --release -p turbo-provider-cuda; then
        warn "turbo-provider-cuda build failed; packaging without it"
    fi
fi
if [[ -f "$relq/libturbo_provider_cuda.so" ]]; then
    cp -L "$relq/libturbo_provider_cuda.so" "$stage/providers/"
    present+=("providers/libturbo_provider_cuda.so")
    for extra in libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
        if [[ -f "$relq/$extra" ]]; then
            cp -L "$relq/$extra" "$stage/providers/"
            present+=("providers/$extra")
        else
            absent+=("providers/$extra -- not found next to the CUDA provider build in target/release")
        fi
    done
else
    if [[ $no_cuda -eq 1 ]]; then
        absent+=("providers/libturbo_provider_cuda.so -- skipped by --no-cuda")
        absent+=("providers/libonnxruntime_providers_cuda.so -- skipped by --no-cuda")
        absent+=("providers/libonnxruntime_providers_shared.so -- skipped by --no-cuda")
    else
        absent+=("providers/libturbo_provider_cuda.so -- not built on this machine (no CUDA toolkit, or the build failed; see the log above)")
        absent+=("providers/libonnxruntime_providers_cuda.so -- CUDA provider not built")
        absent+=("providers/libonnxruntime_providers_shared.so -- CUDA provider not built")
    fi
fi

# --- optional provider: openvino ------------------------------------------
ov_lib="$root/build/openvino/libturbo_provider_openvino.so"
if [[ $no_openvino -eq 1 ]]; then
    log "skipping the OpenVINO provider build (--no-openvino)"
elif [[ -f "$ov_lib" ]]; then
    log "using the existing OpenVINO provider build at build/openvino"
else
    ov_dir="${TURBO_OPENVINO_DIR:-}"
    if [[ -z "$ov_dir" ]]; then
        for cand in "$HOME"/opt/openvino_genai_* "$HOME"/opt/openvino_*; do
            [[ -d "$cand/runtime/cmake" ]] && { ov_dir="$cand"; break; }
        done
    fi
    if [[ -n "$ov_dir" && -d "$ov_dir/runtime/cmake" ]]; then
        log "building the OpenVINO provider against $ov_dir"
        if cmake -S "$root/providers/openvino" -B "$root/build/openvino" \
                -DOpenVINO_DIR="$ov_dir/runtime/cmake" -DCMAKE_BUILD_TYPE=Release >&2 \
            && cmake --build "$root/build/openvino" -j >&2; then
            :
        else
            warn "OpenVINO provider build failed; packaging without it"
        fi
    else
        warn "no OpenVINO SDK found (set TURBO_OPENVINO_DIR); packaging without the OpenVINO provider"
    fi
fi
if [[ -f "$ov_lib" ]]; then
    cp -L "$ov_lib" "$stage/providers/"
    present+=("providers/libturbo_provider_openvino.so")
else
    if [[ $no_openvino -eq 1 ]]; then
        absent+=("providers/libturbo_provider_openvino.so -- skipped by --no-openvino")
    else
        absent+=("providers/libturbo_provider_openvino.so -- not built on this machine (no OpenVINO SDK found, or the build failed; see the log above)")
    fi
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
ldd_failed=0
check_lib() {
    local lib="$1" strict="$2"
    local out unresolved=() extra=() dep
    out="$(ldd "$lib" 2>&1)" || true
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
[[ $ldd_failed -eq 0 ]] || die "libturbo.so or the mock provider has an unresolved or unexpected dependency (see above)"

# --- package README -------------------------------------------------------
list_or_none() {
    if [[ $# -eq 0 ]]; then
        printf '(none)\n'
    else
        local item
        for item in "$@"; do printf -- '- %s\n' "$item"; done
    fi
}

cat > "$stage/README.md" <<EOF
# Turbo $version ($target)

This archive is a Turbo distribution built on $(hostname) for $target. It is
consumed outside the TurboEmbed repository; see \`docs/packaging.md\` in the
repository for the full design. Load a provider with
\`turbo_runtime_load_provider\` (\`include/turbo/turbo_provider.h\`), passing
the path to one of the \`providers/*.so\` files below, or list every provider
directory in \`turbo_runtime_desc.provider_paths\` at \`turbo_runtime_create\`.

## What is in this archive

$(list_or_none "${present[@]}")

## What is not in this archive (and why)

$(list_or_none "${absent[@]}")

The CUDA 13 user-space libraries the CUDA provider's ONNX Runtime execution
provider links against (cuBLAS, cuBLASLt, cuDNN 9, NVRTC, the CUDA 13
runtime) are never bundled: they are large, GPU-driver-version-sensitive,
and already managed by whatever CUDA install or NVIDIA PyPI wheel set the
target machine uses. Point the provider at them with the \`TURBO_CUDA_LIB_DIR\`
environment variable (or the context option \`cuda_lib_dir\`); the provider
preloads every \`lib*.so*\` in that directory before creating a session. A
missing library there is \`TURBO_E_DEVICE_UNAVAILABLE\` at context creation,
never a silent fallback to CPU.

The OpenVINO runtime (\`libopenvino.so\`, its plugins, and the bundled TBB) is
never bundled either: it is a multi-hundred-megabyte SDK with its own
versioned install layout, and the provider library's RUNPATH already points
at the OpenVINO tree it was built against. Set \`LD_LIBRARY_PATH\` to that
tree's \`runtime/lib/intel64\` (and its \`runtime/3rdparty\` TBB directory) if
the provider was moved off the machine it was built on.

## Environment variables a consumer may need

- \`TURBO_CUDA_LIB_DIR\`: directory holding the CUDA 13 user-space libraries
  (cuBLAS, cuBLASLt, cuDNN 9, NVRTC, the CUDA runtime) for the CUDA provider.
- \`LD_LIBRARY_PATH\`: must include this archive's \`lib/\` directory for any
  binary linked against \`libturbo.so\`, and the OpenVINO runtime's
  \`runtime/lib/intel64\` (plus its TBB directory) when using the OpenVINO
  provider away from its build machine.

## Using it from C

\`\`\`c
#include "turbo/turbo.h"
/* cc ... -Iinclude -Llib -lturbo -Wl,-rpath,lib */
\`\`\`

See \`include/turbo/turbo.h\` for the full C ABI and \`bin/turbo-bundle\` for
importing, verifying, and inspecting model bundles.
EOF
present+=("README.md")

# --- checksums --------------------------------------------------------
log "computing SHA256SUMS"
(
    cd "$stage"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | sed -z 's#^\./##' \
        | xargs -0 sha256sum > SHA256SUMS
)
present+=("SHA256SUMS")

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

size_bytes="$(stat -c '%s' "$archive" 2>/dev/null || stat -f '%z' "$archive")"
size_human="$(numfmt --to=iec --suffix=B "$size_bytes" 2>/dev/null || echo "${size_bytes} bytes")"
log "archive: $archive ($size_human, $size_bytes bytes)"
