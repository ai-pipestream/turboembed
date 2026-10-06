#!/usr/bin/env bash
# Intel GPUs (Xe2, such as the Arc Pro B70) through Level Zero, the
# levelzero feature, and with --onednn the levelzero-onednn feature
# (docs/setup/intel-gpu.md).
#
#   intel-gpu.sh [--check|--install|--dry-run] [--onednn]
#
# --install on Ubuntu installs a clang with a SPIR-V backend, the Level
# Zero loader and Intel's GPU compute runtime, and with --onednn the oneAPI
# compiler and oneDNN from Intel's public apt repository. It never changes
# the kernel driver or group membership; the check says what is missing.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
onednn=0
set -- "${TURBO_SETUP_ARGS[@]+"${TURBO_SETUP_ARGS[@]}"}"
while [ $# -gt 0 ]; do
    case "$1" in
        --onednn) onednn=1 ;;
        -h|--help) usage_common "[--onednn]"; exit 0 ;;
        *) usage_common "[--onednn]"; exit 2 ;;
    esac
    shift
done

echo "TurboEmbed setup: intel-gpu$( [ $onednn = 1 ] && echo ' with oneDNN') ($(os_id) $(os_version), $(uname -m))"
if [ "$(uname -s)" != Linux ]; then
    missing "the levelzero backend is built and tested on Linux"
    finish; exit
fi
check_rust
check_c_compiler

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

section "A clang that compiles OpenCL C to SPIR-V (build time)"
# The same flags core/build.rs passes, on a kernel that uses the sub-group
# extension.
cat > "$scratch/probe.cl" <<'CL'
__attribute__((intel_reqd_sub_group_size(16)))
kernel void probe(global float *x) { x[get_global_id(0)] = sub_group_broadcast(x[0], 0); }
CL
probe_clang() {
    "$1" -cl-std=CL3.0 --target=spirv64 -O2 -mllvm --spirv-ext=+SPV_INTEL_subgroups -c "$scratch/probe.cl" -o "$scratch/probe.spv" >/dev/null 2>&1
}
wrapper="$TURBO_ROOT/scripts/setup/clang-spirv.sh"
if [ -n "${TURBO_CLANG:-}" ]; then
    if probe_clang "$TURBO_CLANG"; then ok "TURBO_CLANG=$TURBO_CLANG compiles the kernels"
    else missing "TURBO_CLANG=$TURBO_CLANG does not compile OpenCL C to SPIR-V with --spirv-ext"; fi
elif probe_clang clang; then
    ok "clang on the PATH compiles the kernels ($(clang -dumpversion))"
elif probe_clang "$wrapper"; then
    ok "scripts/setup/clang-spirv.sh compiles the kernels"
    note "set: export TURBO_CLANG=$wrapper"
else
    missing "a clang 20 or newer with LLVM's SPIR-V backend (clang $(clang -dumpversion 2>/dev/null || echo 'not found') on the PATH does not take --spirv-ext)"
    case "$(os_id)" in
        ubuntu|debian) apt_install clang-20 && note "then: export TURBO_CLANG=$wrapper" ;;
        fedora) install_step "$(as_root dnf install -y clang)" ;;
        *) note "install clang 20 or newer, then: export TURBO_CLANG=$wrapper" ;;
    esac
fi

section "Intel GPU and kernel driver (run time)"
found=0
for card in /sys/class/drm/card[0-9]*; do
    [ -e "$card/device/vendor" ] || continue
    case "$card" in *-*) continue ;; esac
    [ "$(cat "$card/device/vendor")" = 0x8086 ] || continue
    drv=$(basename "$(readlink -f "$card/device/driver" 2>/dev/null)" 2>/dev/null)
    dev=$(cat "$card/device/device")
    found=1
    case "$drv" in
        xe|i915) ok "$(basename "$card"): Intel GPU $dev, driver $drv" ;;
        *) missing "$(basename "$card"): Intel GPU $dev with no xe or i915 driver bound" ;;
    esac
done
[ $found = 1 ] || missing "no Intel GPU under /sys/class/drm (the backend lists no device; builds still work)"

nodes=$(ls /dev/dri/renderD* 2>/dev/null || true)
if [ -z "$nodes" ]; then
    [ $found = 0 ] || missing "no /dev/dri/renderD* node"
else
    for n in $nodes; do
        if [ -r "$n" ] && [ -w "$n" ]; then ok "$n opens for $(id -un)"
        else
            missing "$n is not readable and writable by $(id -un)"
            note "fix: $(as_root usermod -aG render "$(id -un)")   # then log in again"
        fi
    done
fi

section "Level Zero loader and Intel's GPU compute runtime (run time)"
libpath() { /sbin/ldconfig -p 2>/dev/null | awk -v l="$1" '$1 == l { print $NF; exit }'; }
loader=$(libpath libze_loader.so.1)
gpu_rt=$(libpath libze_intel_gpu.so.1)
need_rt=0
if [ -n "$loader" ]; then
    real=$(readlink -f "$loader"); v=${real##*.so.}
    if version_ge "$v" 1.10.0; then ok "Level Zero loader $v ($real)"
    else missing "Level Zero loader $v is older than 1.10"; need_rt=1; fi
else
    missing "libze_loader.so.1 (the Level Zero loader)"; need_rt=1
fi
if [ -n "$gpu_rt" ]; then
    ok "Intel GPU compute runtime ($(readlink -f "$gpu_rt"))"
    if have dpkg-query; then
        rv=$(dpkg-query -W -f '${Version}' libze-intel-gpu1 2>/dev/null || dpkg-query -W -f '${Version}' intel-level-zero-gpu 2>/dev/null || true)
        if [ -n "$rv" ]; then
            note "package version $rv"
            case "$rv" in 2[0-3].*) warn "compute runtime $rv predates Xe2 (Battlemage) support; install Intel's current packages" ; need_rt=1 ;; esac
        fi
    fi
else
    missing "libze_intel_gpu.so.1 (Intel's GPU driver for Level Zero)"; need_rt=1
fi
if [ $need_rt = 1 ]; then
    case "$(os_id):$(os_version)" in
        ubuntu:24.04|ubuntu:24.10|ubuntu:25.04)
            # Intel's packages for Arc on Ubuntu; the archive's are older than Xe2.
            install_step "$(as_root add-apt-repository -y ppa:kobuk-team/intel-graphics)" &&
                apt_install libze1 libze-intel-gpu1 intel-opencl-icd ;;
        ubuntu:*|debian:*) apt_install libze1 libze-intel-gpu1 intel-opencl-icd ;;
        *) note "install Intel's compute runtime (github.com/intel/compute-runtime releases) and the Level Zero loader" ;;
    esac
fi

if [ $onednn = 1 ]; then
    section "oneAPI compiler and oneDNN (levelzero-onednn, build and run time)"
    icpx=${TURBO_ICPX:-icpx}
    if ! have "$icpx" && [ -r /opt/intel/oneapi/setvars.sh ]; then
        note "icpx is not on the PATH; trying with /opt/intel/oneapi/setvars.sh"
        # shellcheck disable=SC1091
        . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    fi
    if have "$icpx"; then
        ok "$icpx: $("$icpx" --version 2>/dev/null | head -n1)"
        cat > "$scratch/probe.cpp" <<'CPP'
#include <oneapi/dnnl/dnnl.hpp>
#include <sycl/sycl.hpp>
int main() { return dnnl::version()->major > 0 ? 0 : 1; }
CPP
        if "$icpx" -fsycl -std=c++17 "$scratch/probe.cpp" -ldnnl -o "$scratch/probe" >/dev/null 2>&1; then
            ok "oneDNN headers and libdnnl link with $icpx -fsycl"
            if "$scratch/probe" >/dev/null 2>&1; then ok "libdnnl and the SYCL runtime load"
            else warn "the probe linked but did not run: source /opt/intel/oneapi/setvars.sh, or put oneDNN's lib directory on LD_LIBRARY_PATH"; fi
        else
            missing "oneDNN (dnnl.hpp and libdnnl) for $icpx -fsycl"
        fi
    else
        missing "the oneAPI DPC++/C++ compiler (icpx; or set TURBO_ICPX)"
    fi
    if ! have "$icpx" || ! "$icpx" -fsycl -std=c++17 "$scratch/probe.cpp" -ldnnl -o "$scratch/probe" >/dev/null 2>&1; then
        case "$(os_id)" in
            ubuntu|debian)
                if [ ! -e /etc/apt/sources.list.d/oneAPI.list ]; then
                    install_step "curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | $(as_root tee /usr/share/keyrings/oneapi-archive-keyring.gpg) >/dev/null && echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main' | $(as_root tee /etc/apt/sources.list.d/oneAPI.list) >/dev/null && $(as_root apt-get update)" &&
                        apt_install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dnnl-devel
                else
                    apt_install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dnnl-devel
                fi
                note "then: source /opt/intel/oneapi/setvars.sh" ;;
            *) note "install the oneAPI Base Toolkit's DPC++ compiler and oneDNN, then source /opt/intel/oneapi/setvars.sh" ;;
        esac
    fi
fi

finish
