#!/usr/bin/env bash
# Apple silicon GPUs through Metal, the metal feature (docs/setup/apple-metal.md).
#
# Needs macOS 14 or later on Apple silicon and the Xcode command line
# tools; --install starts Apple's installer for the tools, which asks on
# screen and needs no account.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
case "${TURBO_SETUP_ARGS[0]:-}" in -h|--help) usage_common ""; exit 0 ;; "") ;; *) usage_common ""; exit 2 ;; esac

echo "TurboEmbed setup: metal ($(os_id) $(os_version), $(uname -m))"
if [ "$(uname -s)" != Darwin ]; then
    missing "the metal feature builds only for macOS"
    finish; exit
fi

section "Mac"
v=$(sw_vers -productVersion)
if version_ge "$v" 14.0; then ok "macOS $v"
else missing "macOS $v: the backend needs macOS 14 or later (Metal 3.1); an older Mac lists its GPU as UNSUPPORTED"; fi
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
    ok "Apple silicon: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    if [ "$(uname -m)" != arm64 ]; then warn "this shell runs under Rosetta ($(uname -m)); build from an arm64 shell so cargo targets aarch64"; fi
else
    missing "an Intel Mac: its GPU is listed and refused as UNSUPPORTED (the backend needs Apple silicon's unified memory)"
fi

section "Xcode command line tools (build time)"
if xcode-select -p >/dev/null 2>&1 && sdk=$(xcrun --sdk macosx --show-sdk-version 2>/dev/null) && [ -n "$sdk" ]; then
    ok "macOS SDK $sdk at $(xcrun --sdk macosx --show-sdk-path)"
    if xcrun --sdk macosx --find clang++ >/dev/null 2>&1; then ok "clang++: $(xcrun --sdk macosx --find clang++)"
    else missing "xcrun --find clang++ fails"; fi
else
    missing "the Xcode command line tools (xcrun --sdk macosx --show-sdk-version fails)"
    install_step "xcode-select --install"
    note "the installer opens a window; run this script again when it is done"
fi
check_rust

finish
