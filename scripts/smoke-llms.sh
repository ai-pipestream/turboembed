#!/usr/bin/env bash
# Smoke-test Tokenize + ModelStreamInfer for every *generation* alias a
# running inferstream binary serves. Works against any arch (nvidia
# llama.cpp-CUDA in-process, intel llama.cpp-SYCL in-process, apple mlx-lm):
# the whole point of logical model names is that this script does not care
# which engine answers.
#
# Live GPU is the acceptance path. This script talks to an already-running
# server — it does not start one, and it does not download weights.
#
# Usage:
#   scripts/smoke-llms.sh [host:port] [bearer-token] [model ...]
#
#   host:port      default 127.0.0.1:8461
#   bearer-token   default "change-me" (pass "" for auth mode = none)
#   model ...      subset to test; default = catalog LLM aliases present
#                  in ListModels (default-llm, qwen-0.5b, qwen-7b)
#
# Examples:
#   scripts/smoke-llms.sh                                # local, known aliases
#   scripts/smoke-llms.sh krick:8461 "$KEY"              # nvidia host
#   scripts/smoke-llms.sh 127.0.0.1:8473 "$KEY" default-llm qwen-0.5b qwen-7b
#
# Needs: grpcurl, jq, coreutils (date, wc, base64). No python3.
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8461}"
TOKEN="${2-change-me}"
shift $(( $# > 2 ? 2 : $# )) || true

EXT=(-proto crates/protocol/proto/inferstream_extension.proto)
OIP=(-proto crates/protocol/proto/open_inference_grpc.proto)
AUTH=()
if [ -n "$TOKEN" ]; then
    AUTH=(-H "authorization: Bearer $TOKEN")
fi

command -v grpcurl >/dev/null || { echo "error: grpcurl not installed" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required (no Python JSON parser)" >&2; exit 1; }
command -v base64 >/dev/null || { echo "error: base64 not installed" >&2; exit 1; }

KNOWN_LLM='default-llm qwen-0.5b qwen-7b'
SMOKE_PROMPT='Write a short paragraph about rivers flowing to the sea.'
MAX_TOKENS=64

now_ns() {
    local t
    t=$(date +%s%N 2>/dev/null || true)
    if echo "$t" | grep -Eq '^[0-9]{16,}$'; then
        echo "$t"
    else
        echo $(($(date +%s) * 1000000000))
    fi
}

# Length-prefixed BYTES tensor → base64 (OIP raw contents).
oip_text_b64() {
    local t="$1" n
    n=$(printf '%s' "$t" | wc -c)
    {
        printf '%b' "$(printf '\\%03o\\%03o\\%03o\\%03o' \
            $((n & 255)) $(((n >> 8) & 255)) $(((n >> 16) & 255)) $(((n >> 24) & 255)))"
        printf '%s' "$t"
    } | base64 | tr -d '\n\r'
}

# grpcurl may emit one JSON object per chunk, or a JSON array.
stream_objs() {
    local raw="$1"
    if printf '%s' "$raw" | jq -e 'type == "array"' >/dev/null 2>&1; then
        printf '%s' "$raw"
    else
        printf '%s' "$raw" | jq -s '.'
    fi
}

echo "--- ListModels @ $ADDR ---"
LISTING=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
    inferstream.v1.InferstreamService/ListModels)
echo "$LISTING" | jq -r '.models[] | "\(.name)\tbackend=\(.backend)\tready=\(.ready // false)\thas_tokenizer=\(.hasTokenizer // false)"'

SERVED=()
while IFS= read -r name; do
    SERVED+=("$name")
done < <(echo "$LISTING" | jq -r '.models[].name')

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for alias in $KNOWN_LLM; do
        for served in "${SERVED[@]}"; do
            if [ "$served" = "$alias" ]; then
                MODELS+=("$alias")
                break
            fi
        done
    done
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "error: no LLM aliases to smoke (served: ${SERVED[*]:-none})" >&2
    echo "pass model names, or add default-llm / qwen-0.5b / qwen-7b to serve" >&2
    exit 1
fi

tokenize_once() {
    local model="$1" out
    local req
    req=$(jq -n --arg m "$model" \
        '{model_name:$m, texts:["Hello, inferstream!"]}')
    out=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" -d "$req" "$ADDR" \
        inferstream.v1.InferstreamService/Tokenize) || return 1
    printf '%s' "$out" | jq -r '
        (.encodings // []) as $encs
        | if ($encs | length) == 0 then "error: no encodings\n" | halt_error(1) else . end
        | $encs[0] as $e
        | ($e.ids // $e.inputIds // $e.input_ids // []) as $ids
        | ($e.tokens // []) as $toks
        | if ($ids | length) < 1 then "error: no encodings\n" | halt_error(1)
          else "\($ids | length) \($toks | length) \($toks[0] // "-")"
          end
    '
}

stream_once() {
    local model="$1" t0 t1 ms
    local b64 req out parsed
    b64=$(oip_text_b64 "$SMOKE_PROMPT")
    req=$(jq -n --arg m "$model" --arg b64 "$b64" --argjson n "$MAX_TOKENS" \
        '{model_name:$m, id:"smoke-llms",
          inputs:[{name:"text", datatype:"BYTES", shape:[1]}],
          raw_input_contents:[$b64],
          parameters:{max_tokens:{int64Param:($n|tostring)}}}')
    t0=$(now_ns)
    out=$(grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" -d "$req" "$ADDR" \
        inference.GRPCInferenceService/ModelStreamInfer) || return 1
    t1=$(now_ns)
    ms=$(( (t1 - t0) / 1000000 ))
    parsed=$(stream_objs "$out" | jq -r '
        if type != "array" or length == 0 then "error: empty stream\n" | halt_error(1) else . end
        | map(.inferResponse // .)
        | (map(select(.errorMessage != null and .errorMessage != "")) | first) as $err
        | if $err then "error: \($err.errorMessage)\n" | halt_error(1) else . end
        | {
            chunks: [.[] | select(.rawOutputContents != null and (.rawOutputContents | length) > 0)] | length,
            final: any(.parameters.final.boolParam == true),
            predicted: (
                [.[].parameters.tokensPredicted.int64Param,
                 .[].parameters.tokens_predicted.int64Param]
                | map(select(. != null)) | last // "0"
            )
          }
        | "\(.chunks) \(if .final then "true" else "false" end) \(.predicted)"
    ') || return 1
    echo "$parsed $ms"
}

PASS=0
FAIL=0
FAILED_MODELS=()
printf '\n%-16s %8s %8s %8s %10s %10s %8s\n' \
    MODEL TOK_IDS CHUNKS FINAL COLD_MS WARM_MS TOK_S
for model in "${MODELS[@]}"; do
    if ! tok_run=$(tokenize_once "$model"); then
        printf '%-16s FAIL (Tokenize error; rerun grpcurl by hand for detail)\n' "$model"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    set -- ${tok_run}
    tok_ids=$1
    if [ "$tok_ids" -lt 1 ]; then
        printf '%-16s FAIL: Tokenize returned no ids\n' "$model"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    if ! cold_run=$(stream_once "$model"); then
        printf '%-16s FAIL (ModelStreamInfer error; Tokenize ids=%s)\n' "$model" "$tok_ids"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    if ! warm_run=$(stream_once "$model"); then
        printf '%-16s FAIL (warm ModelStreamInfer error)\n' "$model"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    set -- ${cold_run}
    cold_ms=$4
    set -- ${warm_run}
    chunks=$1
    final=$2
    predicted=$3
    warm_ms=$4
    tok_s="-"
    if [ "${predicted:-0}" -gt 0 ] && [ "${warm_ms:-0}" -gt 0 ]; then
        tok_s=$(awk -v n="$predicted" -v ms="$warm_ms" 'BEGIN { printf "%.1f", n / (ms/1000) }')
    fi
    if [ "$chunks" -ge 1 ] && [ "$final" = "true" ]; then
        printf '%-16s %8s %8s %8s %10s %10s %8s\n' \
            "$model" "$tok_ids" "$chunks" "$final" "$cold_ms" "$warm_ms" "$tok_s"
        PASS=$((PASS + 1))
    else
        printf '%-16s FAIL: stream shape chunks=%s final=%s\n' "$model" "$chunks" "$final"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
    fi
done

echo
echo "=== llm smoke: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi
