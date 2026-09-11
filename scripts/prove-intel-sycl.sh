#!/usr/bin/env bash
# GPU + no-Python proof for in-process SYCL StreamInfer.
# Talks to an already-running inferstream-intel. No python3.
#
# Usage:
#   scripts/prove-intel-sycl.sh [host:port] [bearer] [pid]
#
# Samples the process list *during* ModelStreamInfer (not just before/after).
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8473}"
TOKEN="${2-change-me}"
PID="${3:-}"

if [ -z "$PID" ]; then
    PID="$(pgrep -n -f 'inferstream-intel' || true)"
fi
if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then
    echo "error: inferstream-intel not running (pass pid as \$3)" >&2
    exit 1
fi

dump_tree() {
    local label="$1"
    echo "===== $label $(date -Iseconds) pid=$PID ====="
    ps -o pid,ppid,user,cmd -p "$PID"
    echo "-- children --"
    ps --ppid "$PID" -o pid,ppid,user,cmd || echo "(no children)"
    echo "-- tree --"
    pstree -a -p "$PID" 2>/dev/null || ps -ef | awk -v p="$PID" '$2==p || $3==p {print}'
    echo "-- cmdlines (self + descendants) --"
    descendants() {
        local p="$1" c
        echo "$p"
        for c in $(ps --ppid "$p" -o pid= 2>/dev/null); do
            descendants "$c"
        done
    }
    for p in $(descendants "$PID"); do
        if [ -r "/proc/$p/cmdline" ]; then
            printf '  %s: ' "$p"
            tr '\0' ' ' < "/proc/$p/cmdline"
            echo
        fi
    done
}

python_in_tree() {
    local p cmd
    descendants() {
        local x="$1" c
        echo "$x"
        for c in $(ps --ppid "$x" -o pid= 2>/dev/null); do
            descendants "$c"
        done
    }
    for p in $(descendants "$PID"); do
        [ -r "/proc/$p/cmdline" ] || continue
        cmd="$(tr '\0' ' ' < "/proc/$p/cmdline")"
        case "$cmd" in
            *python*|*Python*)
                echo "$p $cmd"
                return 0
                ;;
        esac
    done
    return 1
}

echo "=== process $PID ==="
dump_tree "before"
if python_in_tree; then
    echo "FAIL: python appears in the inferstream process tree (before)" >&2
    exit 1
fi
if ldd "/proc/$PID/exe" 2>/dev/null | grep -Eiq 'libpython'; then
    echo "FAIL: inferstream-intel is linked against libpython" >&2
    ldd "/proc/$PID/exe" | grep -i python >&2
    exit 1
fi
echo "ok: no python / libpython before StreamInfer"

echo
echo "=== maps (SYCL / Level Zero; must not include libpython) ==="
grep -E 'sycl|libze|ggml|python' "/proc/$PID/maps" | awk '{print $6}' | sort -u | head -40
if grep -Eiq 'libpython' "/proc/$PID/maps"; then
    echo "FAIL: libpython mapped into inferstream-intel" >&2
    exit 1
fi

xe_sample() {
    local label="$1"
    echo "--- xe drm ($label) ---"
    if [ -d "/proc/$PID/fdinfo" ]; then
        grep -hE 'drm-driver|drm-pdev|drm-cycles-ccs|drm-resident-vram' "/proc/$PID/fdinfo/"* 2>/dev/null \
            | sort | uniq -c | sort -nr | head -20 || true
    fi
}

xe_sample "before StreamInfer"

WATCH="$(mktemp)"
python_hits="$(mktemp)"
(
    while [ -d "/proc/$PID" ]; do
        {
            dump_tree "during"
            if hits=$(python_in_tree); then
                echo "$hits" >> "$python_hits"
            fi
        } >> "$WATCH"
        sleep 0.25
    done
) &
WATCH_PID=$!
cleanup() {
    kill "$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo
echo "=== smoke (Tokenize + StreamInfer) — watcher pid $WATCH_PID ==="
scripts/smoke-llms.sh "$ADDR" "$TOKEN" default-llm qwen-0.5b qwen-7b
SMOKE=$?

cleanup
trap - EXIT

echo
echo "=== process list captured during StreamInfer ($(wc -l < "$WATCH") lines) ==="
# Show the last two snapshots so the evidence is in the log, not only on disk.
awk 'BEGIN{n=0} /^===== during /{n++} {buf[n]=buf[n] $0 "\n"} END{for(i=n-1;i<=n;i++) if(i>0) printf "%s", buf[i]}' "$WATCH"
echo "(full log: $WATCH)"

if [ -s "$python_hits" ]; then
    echo "FAIL: python appeared in the inferstream tree DURING StreamInfer" >&2
    cat "$python_hits" >&2
    exit 1
fi
echo "ok: no python in the inferstream process tree during StreamInfer"

echo
dump_tree "after"
xe_sample "after StreamInfer"

echo
echo "=== sycl-ls ==="
if command -v sycl-ls >/dev/null; then
    sycl-ls
else
    echo "(sycl-ls not on PATH; source oneAPI in this shell)"
fi

exit "$SMOKE"
