#!/usr/bin/env bash
# Runs the checks for this machine: cpu and the bundle tool always, then
# each setup whose hardware is present (docs/setup/README.md). Takes the
# same --check, --install and --dry-run as the scripts it runs, and exits
# non-zero when any of them found something missing.
set -u
dir=$(dirname "$0")
status=0
run() { s=$1; shift; printf '\n==== %s\n' "$s"; "$dir/$s" "$@" || status=1; }

run cpu.sh "$@"
run bundle-tool.sh "$@"
case "$(uname -s)" in
    Darwin) run metal.sh "$@" ;;
    Linux)
        if [ -r /etc/nv_tegra_release ] || ls /dev/nvidia0 >/dev/null 2>&1 || grep -qs 0x10de /sys/bus/pci/devices/*/vendor; then
            run cuda.sh "$@"
        fi
        for card in /sys/class/drm/card[0-9]*; do
            if [ "$(cat "$card/device/vendor" 2>/dev/null)" = 0x8086 ] && [ "$(cut -c1-4 "$card/device/class" 2>/dev/null)" = 0x03 ]; then
                run intel-gpu.sh "$@"; break
            fi
        done
        if [ -d /sys/module/intel_vpu ] || ls /dev/accel/accel* >/dev/null 2>&1; then run intel-npu.sh "$@"; fi
        if grep -qs 0x1e60 /sys/bus/pci/devices/*/vendor; then run hailo.sh "$@"; fi
        ;;
esac
printf '\n'
[ $status = 0 ] && echo "All checks passed." || echo "Some checks found something missing; see above."
exit $status
