#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# hailo-select-hef.sh — pick the arch-correct HEF for this board.
#
# A fetched model dir (models/hailo/<alias>) carries one HEF per compiled
# architecture (model.hailo8l.hef, model.hailo8.hef; model.hailo10h.hef once
# a DFC 5.x build exists). HailoRT refuses a foreign-arch HEF at configure
# time; this script copies the one matching the local chip to model.hef,
# which is the path the provider loads.
#
# Usage: scripts/hailo-select-hef.sh [models/hailo/minilm]
# Requires: hailortcli (sudo apt install hailo-all / hailo-h10-all).

set -euo pipefail

dir="${1:-models/hailo/minilm}"

if ! command -v hailortcli >/dev/null 2>&1; then
    echo "hailo-select-hef: hailortcli not found — install HailoRT first" >&2
    echo "  Hailo-8/8L:  sudo apt install dkms hailo-all" >&2
    echo "  Hailo-10H:   sudo apt install dkms hailo-h10-all" >&2
    exit 1
fi

ident="$(hailortcli fw-control identify 2>&1 | tr -d '\0')" || {
    echo "hailo-select-hef: hailortcli fw-control identify failed:" >&2
    echo "$ident" >&2
    exit 1
}

# Order matters: HAILO8L contains HAILO8 as a substring.
arch=""
case "$ident" in
    *HAILO8L*) arch="hailo8l" ;;
    *HAILO10H*) arch="hailo10h" ;;
    *HAILO8*) arch="hailo8" ;;
    *)
        echo "hailo-select-hef: unrecognized device in hailortcli output:" >&2
        echo "$ident" >&2
        exit 1
        ;;
esac

src="$dir/model.$arch.hef"
if [ ! -f "$src" ]; then
    echo "hailo-select-hef: $src not found." >&2
    if [ "$arch" = "hailo10h" ]; then
        echo "There is no public hailo10h HEF yet — compile one per" >&2
        echo "docs/hailo-embed.md (DFC 5.x on an x86 host) and drop it in." >&2
    else
        echo "Fetch it first: make fetch-hailo" >&2
    fi
    exit 1
fi

cp "$src" "$dir/model.hef"
echo "hailo-select-hef: $arch detected -> $dir/model.hef"
