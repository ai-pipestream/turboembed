#!/usr/bin/env bash
# Hailo accelerators through HailoRT, the hailo feature
# (docs/setup/hailo-10h.md, docs/setup/hailo-8.md).
#
#   hailo.sh [--check|--install|--dry-run] [--device hailo10h|hailo8]
#
# Without --device the script reports whatever board it finds. --install on
# Raspberry Pi OS installs the HailoRT packages for the board from the Pi
# repository (hailo-h10-all for a Hailo-10H, hailo-all for a Hailo-8 or
# Hailo-8L); elsewhere HailoRT comes from Hailo's Developer Zone, which
# needs an account, so the script only says where it goes.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
want=""
set -- "${TURBO_SETUP_ARGS[@]+"${TURBO_SETUP_ARGS[@]}"}"
while [ $# -gt 0 ]; do
    case "$1" in
        --device) want=${2:?--device needs hailo10h or hailo8}; shift ;;
        -h|--help) usage_common "[--device hailo10h|hailo8]"; exit 0 ;;
        *) usage_common "[--device hailo10h|hailo8]"; exit 2 ;;
    esac
    shift
done
case "$want" in ""|hailo10h|hailo8|hailo8l) ;; *) echo "--device: hailo10h or hailo8" >&2; exit 2 ;; esac

echo "TurboEmbed setup: hailo ($(os_id) $(os_version), $(uname -m))"
if [ "$(uname -s)" != Linux ]; then
    missing "the hailo backend builds and runs on Linux"
    finish; exit
fi
check_rust
check_c_compiler

section "Board (PCI)"
board=""
for d in /sys/bus/pci/devices/*; do
    [ -e "$d/vendor" ] || continue
    [ "$(cat "$d/vendor")" = 0x1e60 ] || continue
    case "$(cat "$d/device")" in
        0x45c4) board=hailo10h; ok "Hailo-10H at $(basename "$d")" ;;
        0x2864) board=hailo8; ok "Hailo-8 or Hailo-8L at $(basename "$d")" ;;
        *) ok "Hailo device $(cat "$d/device") at $(basename "$d")" ;;
    esac
done
[ -n "$board" ] || missing "no Hailo board on the PCI bus (vendor 1e60)"
[ -n "$want" ] || want=$board
case "$want" in hailo8l) want=hailo8 ;; esac
if [ -n "$board" ] && [ -n "$want" ] && [ "$board" != "$want" ]; then
    warn "asked about $want, found $board"
fi

case "$want" in
    hailo10h) module=hailo1x_pci; pkg=hailo-h10-all; rt_major=5 ;;
    hailo8) module=hailo_pci; pkg=hailo-all; rt_major=4 ;;
    *) module=""; pkg=""; rt_major="" ;;
esac

section "Kernel driver"
if [ -n "$module" ]; then
    if [ -d "/sys/module/$module" ]; then ok "$module is loaded"
    else missing "$module is not loaded (it comes with $pkg on Raspberry Pi OS, or with HailoRT's PCIe driver package)"; fi
else
    for m in hailo1x_pci hailo_pci; do [ -d "/sys/module/$m" ] && ok "$m is loaded"; done
fi
if ls /dev/hailo* >/dev/null 2>&1; then ok "$(ls /dev/hailo* | tr '\n' ' ')"
else missing "no /dev/hailo* node"; fi

section "HailoRT (build and run time)"
root=${TURBO_HAILORT_ROOT:-/usr}
need_rt=0
if [ -f "$root/include/hailo/hailort.h" ]; then ok "$root/include/hailo/hailort.h"
else missing "$root/include/hailo/hailort.h (set TURBO_HAILORT_ROOT to HailoRT's prefix)"; need_rt=1; fi
lib=""
for d in lib "lib/$(uname -m)-linux-gnu" lib64; do
    if [ -e "$root/$d/libhailort.so" ]; then lib=$root/$d/libhailort.so; break; fi
done
if [ -n "$lib" ]; then ok "$lib -> $(readlink -f "$lib")"
else missing "libhailort.so under $root/lib, lib/$(uname -m)-linux-gnu or lib64"; need_rt=1; fi
if have hailortcli; then
    rv=$(hailortcli --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    ok "hailortcli $rv"
    if [ -n "$rt_major" ] && [ "${rv%%.*}" != "$rt_major" ]; then
        missing "HailoRT $rv: a $want takes HailoRT $rt_major.x"
    fi
    if [ -n "$board" ]; then
        note "$(hailortcli scan 2>&1 | tr '\n' ' ' | cut -c1-160)"
    fi
fi
if [ $need_rt = 1 ] && [ -n "$pkg" ]; then
    if [ -r /etc/rpi-issue ] || grep -qs 'Raspberry Pi' /proc/device-tree/model 2>/dev/null; then
        install_step "$(as_root apt-get update)" && apt_install "$pkg"
        note "then reboot so the driver and firmware load"
    else
        note "fix: install HailoRT $rt_major.x and its PCIe driver from Hailo's Developer Zone (hailo.ai, an account is needed), with the headers"
    fi
fi

finish
