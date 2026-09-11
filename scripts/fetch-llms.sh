#!/usr/bin/env bash
# SHA-256-verified GGUF (+ tokenizer.json) fetch. No Python.
#
# Reads models/manifests/llms.json (pinned HF revision + sha256 + size) and
# downloads into models/gguf/<alias>/. Idempotent: a file already on disk
# with a matching SHA-256 is skipped. A stale or tampered file is
# re-downloaded. A post-download hash/size mismatch is a hard error.
#
# Usage:
#   scripts/fetch-llms.sh                  # all aliases (--all)
#   scripts/fetch-llms.sh --all
#   scripts/fetch-llms.sh qwen-0.5b qwen-7b
#   scripts/fetch-llms.sh --verify-only [--all | alias ...]
#   scripts/fetch-llms.sh --list
#   make fetch-llms [ALIASES=qwen-0.5b]
#
# Needs: curl, jq, sha256sum (coreutils). No python3.
set -euo pipefail

ROOT="${INFERSTREAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="${INFERSTREAM_LLM_MANIFEST:-$ROOT/models/manifests/llms.json}"
HF_BASE="https://huggingface.co"
USER_AGENT="inferstream-fetch-llms/1.0"

if ! command -v jq >/dev/null; then
    echo "error: jq is required (no Python JSON parser)" >&2
    exit 1
fi
if ! command -v curl >/dev/null; then
    echo "error: curl is required" >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null; then
    echo "error: sha256sum is required" >&2
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "error: manifest not found: $MANIFEST" >&2
    exit 1
fi

MODE=fetch
ALIASES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --verify-only) MODE=verify; shift ;;
        --list) MODE=list; shift ;;
        --all) ALIASES+=(--all); shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        --*)
            echo "error: unknown flag $1 (this fetcher does not re-pin; edit the manifest)" >&2
            exit 1
            ;;
        *) ALIASES+=("$1"); shift ;;
    esac
done

schema="$(jq -r '.schema_version // empty' "$MANIFEST")"
if [ "$schema" != "1" ]; then
    echo "error: unsupported manifest schema_version=${schema:-missing} (expected 1)" >&2
    exit 1
fi

all_aliases() {
    jq -r '.models | keys[]' "$MANIFEST"
}

resolve_alias() {
    local alias="$1"
    if ! jq -e --arg a "$alias" '.models[$a]' "$MANIFEST" >/dev/null; then
        echo "error: alias '$alias' is not in $MANIFEST" >&2
        return 1
    fi
    local target
    target="$(jq -r --arg a "$alias" '.models[$a].alias_of // empty' "$MANIFEST")"
    if [ -n "$target" ]; then
        if [ "$(jq -r --arg t "$target" '.models[$t].alias_of // empty' "$MANIFEST")" != "" ]; then
            echo "error: $alias: nested alias_of is not supported" >&2
            return 1
        fi
        printf '%s' "$target"
    else
        printf '%s' "$alias"
    fi
}

# Print one line per artifact:
#   repo<TAB>revision<TAB>dest<TAB>path<TAB>sha256<TAB>size
artifact_lines() {
    local key="$1"
    jq -r --arg k "$key" '
        .models[$k] as $e
        | (
            ($e.files // [])[]
            | [$e.repo, $e.revision, $e.dest, .path, .sha256, (.size|tostring)]
            | @tsv
          ),
          (if $e.tokenizer then
              ($e.tokenizer.files // [])[]
              | [$e.tokenizer.repo, $e.tokenizer.revision,
                 $e.dest, .path, .sha256, (.size|tostring)]
              | @tsv
           else empty end)
    ' "$MANIFEST"
}

human() {
    local n="$1"
    if [ "$n" -ge 1073741824 ]; then
        awk -v n="$n" 'BEGIN { printf "%.1f GiB", n/1073741824 }'
    elif [ "$n" -ge 1048576 ]; then
        awk -v n="$n" 'BEGIN { printf "%.1f MiB", n/1048576 }'
    elif [ "$n" -ge 1024 ]; then
        awk -v n="$n" 'BEGIN { printf "%.1f KiB", n/1024 }'
    else
        printf '%s B' "$n"
    fi
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

if [ "$MODE" = "list" ]; then
    printf '%-14s  %-48s  %s\n' ALIAS REPO@REV DEST
    while IFS= read -r alias; do
        key="$(resolve_alias "$alias")"
        repo="$(jq -r --arg k "$key" '.models[$k].repo' "$MANIFEST")"
        rev="$(jq -r --arg k "$key" '.models[$k].revision' "$MANIFEST")"
        dest="$(jq -r --arg k "$key" '.models[$k].dest' "$MANIFEST")"
        extra=""
        if [ "$alias" != "$key" ]; then
            extra=" (alias_of $key)"
        fi
        printf '%-14s  %s@%s  %s%s\n' "$alias" "$repo" "${rev:0:12}" "$dest" "$extra"
    done < <(all_aliases)
    exit 0
fi

if [ ${#ALIASES[@]} -eq 0 ] || [ "${ALIASES[*]}" = "--all" ]; then
    mapfile -t ALIASES < <(all_aliases)
fi

# Drop the sentinel if mixed with names.
FILTERED=()
for a in "${ALIASES[@]}"; do
    [ "$a" = "--all" ] && continue
    FILTERED+=("$a")
done
ALIASES=("${FILTERED[@]}")

FAILED=0
DOWNLOADED=0
SKIPPED=0
CHECKED=0

fetch_one() {
    local repo="$1" revision="$2" dest="$3" rel="$4" expect_sha="$5" expect_size="$6"
    local path="$ROOT/$dest/$rel"
    local url="$HF_BASE/$repo/resolve/$revision/$rel"
    mkdir -p "$(dirname "$path")"
    if [ -f "$path" ]; then
        local got
        got="$(sha256_file "$path")"
        if [ "$got" = "$expect_sha" ]; then
            echo "  ok (cached)  $rel  [$(human "$expect_size")]"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi
        echo "  stale hash, re-downloading  $rel"
    fi
    echo "  downloading  $rel  [$(human "$expect_size")] ..."
    local tmp
    tmp="$(mktemp "$path.XXXXXX.part")"
    if ! curl -fL --retry 5 --retry-delay 2 \
        -A "$USER_AGENT" \
        -o "$tmp" \
        "$url"; then
        rm -f "$tmp"
        echo "error: download failed: $url" >&2
        return 1
    fi
    local got size
    got="$(sha256_file "$tmp")"
    size="$(wc -c < "$tmp" | tr -d ' ')"
    if [ "$got" != "$expect_sha" ] || [ "$size" != "$expect_size" ]; then
        rm -f "$tmp"
        echo "error: hash/size mismatch for $rel" >&2
        echo "  expected sha256=$expect_sha size=$expect_size" >&2
        echo "  got      sha256=$got size=$size" >&2
        return 1
    fi
    mv -f "$tmp" "$path"
    echo "  wrote        $rel  [$(human "$expect_size")]"
    DOWNLOADED=$((DOWNLOADED + 1))
}

verify_one() {
    local dest="$3" rel="$4" expect_sha="$5" expect_size="$6"
    local path="$ROOT/$dest/$rel"
    CHECKED=$((CHECKED + 1))
    if [ ! -f "$path" ]; then
        echo "  MISSING  $rel"
        return 1
    fi
    local got size
    got="$(sha256_file "$path")"
    size="$(wc -c < "$path" | tr -d ' ')"
    if [ "$got" != "$expect_sha" ] || [ "$size" != "$expect_size" ]; then
        echo "  MISMATCH $rel  sha=$got size=$size (want $expect_sha / $expect_size)"
        return 1
    fi
    echo "  ok         $rel  [$(human "$expect_size")]"
}

SEEN=()
already_seen() {
    local k="$1" s
    for s in "${SEEN[@]+"${SEEN[@]}"}"; do
        [ "$s" = "$k" ] && return 0
    done
    return 1
}

for alias in "${ALIASES[@]}"; do
    key="$(resolve_alias "$alias")" || { FAILED=$((FAILED + 1)); continue; }
    if already_seen "$key"; then
        echo "--- $alias  ->  $key (already ${MODE}ed) ---"
        continue
    fi
    SEEN+=("$key")
    dest="$(jq -r --arg k "$key" '.models[$k].dest' "$MANIFEST")"
    repo="$(jq -r --arg k "$key" '.models[$k].repo' "$MANIFEST")"
    rev="$(jq -r --arg k "$key" '.models[$k].revision' "$MANIFEST")"
    label="$alias"
    [ "$alias" != "$key" ] && label="$alias -> $key"
    echo "--- $label  <-  $repo@${rev:0:12}  ->  $dest ---"
    while IFS=$'\t' read -r arepo arev adest apath asha asize; do
        [ -z "${arepo:-}" ] && continue
        if [ "$MODE" = "verify" ]; then
            verify_one "$arepo" "$arev" "$adest" "$apath" "$asha" "$asize" || FAILED=$((FAILED + 1))
        else
            fetch_one "$arepo" "$arev" "$adest" "$apath" "$asha" "$asize" || FAILED=$((FAILED + 1))
        fi
    done < <(artifact_lines "$key")
done

echo
if [ "$MODE" = "verify" ]; then
    echo "=== verify-llms: $CHECKED files checked, $FAILED failed ==="
else
    echo "=== fetch-llms: $DOWNLOADED downloaded, $SKIPPED cached, $FAILED failed ==="
fi
[ "$FAILED" -eq 0 ]
