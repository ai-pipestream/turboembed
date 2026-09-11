#!/usr/bin/env bash
# Offline tests for scripts/fetch-llms.sh + models/manifests/llms.json.
# No python3. Needs jq, sha256sum.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FETCH="$ROOT/scripts/fetch-llms.sh"
MANIFEST="$ROOT/models/manifests/llms.json"
CATALOG="$ROOT/config/catalog.toml"
FAIL=0

pass() { echo "  ok  $*"; }
fail() { echo "  FAIL  $*" >&2; FAIL=$((FAIL + 1)); }

echo "=== manifest structure ==="
schema="$(jq -r '.schema_version' "$MANIFEST")"
[ "$schema" = "1" ] && pass "schema_version=1" || fail "schema_version=$schema"

alias_of="$(jq -r '.models["default-llm"].alias_of' "$MANIFEST")"
[ "$alias_of" = "qwen-0.5b" ] && pass "default-llm alias_of qwen-0.5b" \
    || fail "default-llm alias_of=$alias_of"

for alias in qwen-0.5b qwen-7b; do
    dest="$(jq -r --arg a "$alias" '.models[$a].dest' "$MANIFEST")"
    [ "$dest" = "models/gguf/$alias" ] && pass "$alias dest=$dest" \
        || fail "$alias dest=$dest"
    rev="$(jq -r --arg a "$alias" '.models[$a].revision' "$MANIFEST")"
    echo "$rev" | grep -Eq '^[0-9a-f]{40}$' && pass "$alias revision pinned" \
        || fail "$alias revision not a 40-hex commit: $rev"
    jq -e --arg a "$alias" '
        .models[$a].files[]
        | select((.sha256 | test("^[0-9a-f]{64}$")) and .size > 0)
    ' "$MANIFEST" >/dev/null && pass "$alias files hashed" \
        || fail "$alias missing hashed files"
    jq -e --arg a "$alias" '
        .models[$a].tokenizer.files[]
        | select(.path == "tokenizer.json" and (.sha256 | test("^[0-9a-f]{64}$")))
    ' "$MANIFEST" >/dev/null && pass "$alias tokenizer.json pinned" \
        || fail "$alias tokenizer.json not pinned"
done

jq -e '[.models["qwen-7b"].files[].path]
    | index("qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf")
      and index("qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf")
' "$MANIFEST" >/dev/null && pass "qwen-7b pins both Q5_K_M shards" \
    || fail "qwen-7b missing a Q5_K_M shard"

echo "=== catalog fetch paths (nvidia + intel) ==="
# Parse catalog.toml without Python: pull llama-cpp path= lines under
# [models.*.nvidia] / [models.*.intel] that point at models/gguf/.
# A tiny awk over the compiled-in catalog.
awk '
    $0 ~ /^\[models\./ {
        alias=$0
        sub(/^\[models\./,"",alias)
        sub(/\]$/,"",alias)
        gsub(/"/,"",alias)
        n=split(alias, parts, ".")
        name=parts[1]
        arch=""
        if (n>=2) arch=parts[n]
        # [models."qwen-0.5b".intel] → parts: "qwen-0" "5b" "intel" after
        # quote strip of qwen-0.5b. Re-parse quoted aliases.
    }
' "$CATALOG" >/dev/null

# Quoted-alias aware scan: record backend+path per [models.<alias>.<arch>].
current=""
arch=""
backend=""
path=""
flush() {
    if [ -n "$current" ] && [ "$backend" = "llama-cpp" ] \
        && [ -n "$path" ] && [[ "$path" == models/gguf/* ]]; then
        dest="$(jq -r --arg a "$current" '
            if .models[$a] == null then empty
            else (.models[$a].alias_of // $a) as $k | .models[$k].dest
            end
        ' "$MANIFEST")"
        if [ -z "$dest" ]; then
            fail "catalog alias $current ($arch) not in LLM manifest"
            return
        fi
        rel="${path#"$dest"/}"
        if jq -e --arg a "$current" --arg rel "$rel" '
            (.models[$a].alias_of // $a) as $k
            | .models[$k].files[] | select(.path == $rel)
        ' "$MANIFEST" >/dev/null; then
            pass "catalog $current.$arch -> $path"
        else
            fail "catalog $current.$arch path $path not in manifest dest $dest"
        fi
    fi
}
while IFS= read -r line; do
    case "$line" in
        \[models.*\])
            flush
            header="${line#\[models.}"
            header="${header%]}"
            # Strip quotes around dotted aliases: "qwen-0.5b".intel
            header="$(printf '%s' "$header" | sed 's/"//g')"
            current="${header%.*}"
            arch="${header##*.}"
            # [models.qwen-7b] has no arch suffix equal to the whole name
            if [ "$current" = "$header" ]; then
                current="$header"
                arch=""
            fi
            backend=""
            path=""
            ;;
        backend\ =\ *)
            backend="${line#backend = }"
            backend="${backend#\"}"
            backend="${backend%\"}"
            ;;
        path\ =\ *)
            path="${line#path = }"
            path="${path#\"}"
            path="${path%\"}"
            ;;
    esac
done < "$CATALOG"
flush

echo "=== fetch-llms.sh --list (no network) ==="
LIST_OUT="$("$FETCH" --list)"
if printf '%s\n' "$LIST_OUT" | grep -Fq 'qwen-0.5b'; then
    pass "--list prints qwen-0.5b"
else
    fail "--list did not print qwen-0.5b; got: $LIST_OUT"
fi

echo "=== fixture verify / idempotent fetch / tamper ==="
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

payload="tiny gguf stand-in"$'\n'
printf '%s' "$payload" > "$TMP/payload"
digest="$(sha256sum "$TMP/payload" | awk '{print $1}')"
size="$(wc -c < "$TMP/payload" | tr -d ' ')"
mkdir -p "$TMP/models/gguf/tiny" "$TMP/models/manifests"
printf '%s' "$payload" > "$TMP/models/gguf/tiny/model.bin"
cat > "$TMP/models/manifests/llms.json" <<EOF
{
  "schema_version": 1,
  "models": {
    "tiny": {
      "repo": "example/tiny",
      "revision": "0000000000000000000000000000000000000000",
      "dest": "models/gguf/tiny",
      "files": [
        {"path": "model.bin", "sha256": "$digest", "size": $size}
      ]
    }
  }
}
EOF

if INFERSTREAM_ROOT="$TMP" \
    INFERSTREAM_LLM_MANIFEST="$TMP/models/manifests/llms.json" \
    "$FETCH" --verify-only tiny >/dev/null; then
    pass "verify-only succeeds on matching fixture"
else
    fail "verify-only rejected a matching fixture"
fi

# Idempotent fetch: matching file on disk must not hit the network
# (example/tiny does not exist upstream; a download would fail).
if INFERSTREAM_ROOT="$TMP" \
    INFERSTREAM_LLM_MANIFEST="$TMP/models/manifests/llms.json" \
    "$FETCH" tiny >/dev/null; then
    pass "fetch is idempotent offline when hash matches"
else
    fail "fetch tried to download a cached matching file"
fi

printf 'tampered' > "$TMP/models/gguf/tiny/model.bin"
if INFERSTREAM_ROOT="$TMP" \
    INFERSTREAM_LLM_MANIFEST="$TMP/models/manifests/llms.json" \
    "$FETCH" --verify-only tiny >/dev/null; then
    fail "verify-only missed a tampered file"
else
    pass "verify-only detects tamper"
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "=== test-fetch-llms: all passed ==="
    exit 0
fi
echo "=== test-fetch-llms: $FAIL failed ===" >&2
exit 1
