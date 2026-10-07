#!/usr/bin/env bash
# Intel NPUs (AI Boost on Core Ultra) through the Level Zero graph
# extension, the npu feature (docs/setup/intel-npu.md). Linux only; the
# Windows steps are in that page.
#
# Nothing is needed to build. To run: the Level Zero loader, Intel's NPU
# user-space driver and the kernel's intel_vpu module. The NPU driver is
# released as packages on github.com/intel/linux-npu-driver; this script
# checks for it and says where it comes from, and --install adds only the
# loader from the distribution.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
case "${TURBO_SETUP_ARGS[0]:-}" in -h|--help) usage_common ""; exit 0 ;; "") ;; *) usage_common ""; exit 2 ;; esac

echo "TurboEmbed setup: intel-npu ($(os_id) $(os_version), $(uname -m))"
if [ "$(uname -s)" != Linux ]; then
    missing "this script checks Linux; on Windows see docs/setup/intel-npu.md"
    finish; exit
fi
check_rust
check_c_compiler

section "NPU and kernel driver (run time)"
if [ -d /sys/module/intel_vpu ]; then ok "the intel_vpu kernel module is loaded"
else missing "the intel_vpu kernel module is not loaded (in mainline Linux since 6.3; try: sudo modprobe intel_vpu)"; fi
nodes=$(ls /dev/accel/accel* 2>/dev/null || true)
if [ -z "$nodes" ]; then
    missing "no /dev/accel/accel* node (no NPU, or the module did not bind)"
else
    for n in $nodes; do
        if [ -r "$n" ] && [ -w "$n" ]; then ok "$n opens for $(id -un)"
        else
            missing "$n is not readable and writable by $(id -un)"
            note "fix: $(as_root usermod -aG render "$(id -un)")   # then log in again"
        fi
    done
fi

section "Level Zero loader and Intel's NPU driver (run time)"
libpath() { /sbin/ldconfig -p 2>/dev/null | awk -v l="$1" '$1 == l { print $NF; exit }'; }
loader=$(libpath libze_loader.so.1)
if [ -n "$loader" ]; then
    real=$(readlink -f "$loader"); v=${real##*.so.}
    if version_ge "$v" 1.10.0; then ok "Level Zero loader $v ($real)"
    else missing "Level Zero loader $v is older than 1.10, the first with zeInitDrivers"; fi
else
    missing "libze_loader.so.1 (the Level Zero loader, 1.10 or newer)"
    case "$(os_id)" in ubuntu|debian) apt_install libze1 ;; esac
fi
vpu=$(libpath libze_intel_vpu.so.1)
[ -n "$vpu" ] || vpu=$(libpath libze_intel_vpu.so)
if [ -n "$vpu" ]; then
    ok "Intel NPU driver ($(readlink -f "$vpu"))"
    note "the driver must offer graph extension 1.8 or newer and compiler 5.9 or newer; model loading names either when older"
else
    missing "libze_intel_vpu.so (Intel's NPU user-space driver)"
    note "fix: install the packages from the latest release at https://github.com/intel/linux-npu-driver/releases (it lists the steps per Ubuntu version)"
fi

finish
