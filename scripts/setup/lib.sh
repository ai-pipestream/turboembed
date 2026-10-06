# Shared helpers for the setup scripts in this directory. Sourced, not run.
#
# Each script checks by default and changes nothing. With --install it
# installs what is missing and can be installed without an account or a
# licence, printing every command before it runs it; anything that needs
# a login or a licence is only checked, and the script says where the file
# goes. Running a script twice does nothing the second time.
#
# Written for bash 3.2 (macOS) and later.

TURBO_SETUP_MODE=check
TURBO_SETUP_MISSING=0
TURBO_SETUP_WARNINGS=0
TURBO_SETUP_DRY=0

# The repository root, from this file's location.
TURBO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    _c_ok=$'\033[32m' _c_miss=$'\033[31m' _c_warn=$'\033[33m' _c_off=$'\033[0m'
else
    _c_ok='' _c_miss='' _c_warn='' _c_off=''
fi

ok()      { printf '  %sok%s       %s\n' "$_c_ok" "$_c_off" "$*"; }
missing() { printf '  %smissing%s  %s\n' "$_c_miss" "$_c_off" "$*"; TURBO_SETUP_MISSING=$((TURBO_SETUP_MISSING + 1)); }
warn()    { printf '  %swarning%s  %s\n' "$_c_warn" "$_c_off" "$*"; TURBO_SETUP_WARNINGS=$((TURBO_SETUP_WARNINGS + 1)); }
note()    { printf '           %s\n' "$*"; }
section() { printf '\n%s\n' "$*"; }

usage_common() {
    cat <<USAGE
Usage: $(basename "$0") [--check | --install | --dry-run] $1

  --check    report what is present and what is missing (the default)
  --install  install what is missing and needs no login or licence
  --dry-run  print what --install would run, and run nothing
USAGE
}

# Parses the mode flags; anything else is left in TURBO_SETUP_ARGS.
parse_mode() {
    TURBO_SETUP_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --check) TURBO_SETUP_MODE=check ;;
            --install) TURBO_SETUP_MODE=install ;;
            --dry-run) TURBO_SETUP_MODE=install; TURBO_SETUP_DRY=1 ;;
            *) TURBO_SETUP_ARGS+=("$1") ;;
        esac
        shift
    done
}

have() { command -v "$1" >/dev/null 2>&1; }

# True when version $1 is at least $2 (dotted numbers).
version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | head -n1)" = "$2" ]
}

os_id() {
    if [ "$(uname -s)" = Darwin ]; then echo macos; return; fi
    if [ -r /etc/os-release ]; then (. /etc/os-release && echo "${ID:-linux}"); else echo linux; fi
}

os_version() {
    if [ "$(uname -s)" = Darwin ]; then sw_vers -productVersion; return; fi
    if [ -r /etc/os-release ]; then (. /etc/os-release && echo "${VERSION_ID:-}"); fi
}

# Runs a command in --install mode (printing it first), or says what would
# run. In --check mode it only prints the command as the fix.
install_step() {
    if [ "$TURBO_SETUP_MODE" != install ]; then
        note "fix: $*"
        return 0
    fi
    printf '  + %s\n' "$*"
    if [ "$TURBO_SETUP_DRY" = 1 ]; then return 0; fi
    if ! sh -c "$*"; then
        printf '  %sfailed%s   the step above; the steps that depend on it are skipped\n' "$_c_miss" "$_c_off"
        return 1
    fi
}

# The command prefixed with sudo when not root.
as_root() { if [ "$(id -u)" = 0 ]; then echo "$*"; else echo "sudo $*"; fi; }

apt_install() { install_step "$(as_root apt-get install -y --no-install-recommends "$@")"; }

# Rust: rustc and cargo, from rustup when missing.
check_rust() {
    section "Rust toolchain"
    if have cargo && have rustc; then
        local v; v=$(rustc --version | awk '{print $2}')
        if version_ge "${v%%-*}" 1.85.0; then
            ok "rustc $v (the workspace is edition 2024; CI builds on current stable)"
        else
            missing "rustc $v is older than 1.85, the first with edition 2024"
            if have rustup; then install_step "rustup update stable"; fi
        fi
    else
        missing "rustc and cargo"
        install_step "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component rustfmt,clippy"
        note "then open a new shell, or: . \"\$HOME/.cargo/env\""
    fi
}

check_c_compiler() {
    section "C and C++ compilers"
    if have cc; then ok "cc: $(cc --version 2>/dev/null | head -n1)"; else missing "cc (a C compiler, for the build scripts' links)"; fi
    if have c++; then ok "c++: $(c++ --version 2>/dev/null | head -n1)"; else missing "c++ (a C++17 compiler)"; fi
    if ! have cc || ! have c++; then
        case "$(os_id)" in
            ubuntu|debian|raspbian) apt_install build-essential ;;
            fedora) install_step "$(as_root dnf install -y gcc gcc-c++)" ;;
            macos) install_step "xcode-select --install" ;;
            *) note "install gcc and g++ (or clang) with your package manager" ;;
        esac
    fi
}

# Prints the summary and returns the exit status: 0 with nothing missing.
finish() {
    printf '\n'
    if [ "$TURBO_SETUP_MISSING" = 0 ]; then
        printf '%sReady%s: nothing missing' "$_c_ok" "$_c_off"
        [ "$TURBO_SETUP_WARNINGS" = 0 ] || printf ' (%s warning(s) above)' "$TURBO_SETUP_WARNINGS"
        printf '.\n'
        return 0
    fi
    printf '%s%s missing%s' "$_c_miss" "$TURBO_SETUP_MISSING" "$_c_off"
    [ "$TURBO_SETUP_WARNINGS" = 0 ] || printf ', %s warning(s)' "$TURBO_SETUP_WARNINGS"
    if [ "$TURBO_SETUP_MODE" = check ]; then
        printf '. Each "fix:" line is what --install would run, where it can.\n'
    else
        printf '. Run the script again with --check once the steps above are done.\n'
    fi
    return 1
}
