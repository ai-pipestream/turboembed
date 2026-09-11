#!/usr/bin/env bash
# Smoke-test Tokenize + ModelStreamInfer for every generation alias a
# running inferstream binary serves. Works against any arch.
#
# Usage:
#   scripts/smoke-llms.sh [host:port] [bearer-token] [model ...]
#
# Needs grpcurl, jq, perl (pack BYTES). No python3.
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
command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

KNOWN_LLM='default-llm qwen-0.5b qwen-7b'

now_ms() {
    perl -MTime::HiRes=time -e 'printf("%d\n", time*1000)'
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
        .encodings[0] as $e
        | (($e.ids // $e.inputIds // []) | length) as $n
        | (($e.tokens // []) | length) as $t
        | "\($n) \($t) \($e.tokens[0] // "-")"
    '
}

stream_once() {
    local model="$1"
    local b64
    b64=$(printf '%s' 'Say hello in five words or fewer.' \
        | perl -e 'undef $/; $t=<>; print pack("V", length($t)).$t' | base64)
    local req
    req=$(jq -n --arg m "$model" --arg b64 "$b64" \
        '{model_name:$m, id:"smoke-llms", inputs:[{name:"text", datatype:"BYTES", shape:[1]}], raw_input_contents:[$b64], parameters:{max_tokens:{int64Param:"32"}}}')
    local t0 t1 out
    t0=$(now_ms)
    out=$(grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" -d "$req" "$ADDR" \
        inference.GRPCInferenceService/ModelStreamInfer) || return 1
    t1=$(now_ms)
    local parsed
    parsed=$(printf '%s' "$out" | jq -s -r '
        (map(.inferResponse // .) ) as $objs
        | ($objs | map(select((.rawOutputContents // []) | length > 0)) | length) as $chunks
        | ([$objs[] | .parameters.final.boolParam // false] | any) as $final
        | ([$objs[] | .parameters.decode_tokens_per_second.doubleParam // empty] | .[0] // 0) as $tps
        | "\($chunks) \(if $final then "true" else "false" end) \($tps)"
    ')
    echo "$parsed $((t1 - t0))"
}

tok_s() {
    awk -v n="$1" -v ms="$2" 'BEGIN { if (ms > 0) printf "%.1f", n / (ms/1000.0); else print "inf" }'
}

PASS=0
FAIL=0
FAILED_MODELS=()
printf '\n%-16s %8s %8s %8s %10s %10s %9s %9s %10s\n' \
    MODEL TOK_IDS CHUNKS FINAL COLD_MS WARM_MS COLD_T/S WARM_T/S ENG_T/S
for model in "${MODELS[@]}"; do
    if ! tok_run=$(tokenize_once "$model"); then
        printf '%-16s FAIL (Tokenize error)\n' "$model"
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
    eng_tps=$3
    warm_ms=$4
    if [ "$chunks" -ge 1 ] && [ "$final" = "true" ]; then
        cold_tps=$(tok_s "$chunks" "$cold_ms")
        warm_tps=$(tok_s "$chunks" "$warm_ms")
        printf '%-16s %8s %8s %8s %10s %10s %9s %9s %10s\n' \
            "$model" "$tok_ids" "$chunks" "$final" "$cold_ms" "$warm_ms" \
            "$cold_tps" "$warm_tps" "$eng_tps"
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
