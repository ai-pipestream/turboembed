#!/usr/bin/env bash
# Smoke-test Tokenize + ModelStreamInfer for every *generation* alias a
# running inferstream binary serves. Works against any arch (nvidia
# llama.cpp-CUDA, intel llama.cpp-SYCL server-client, apple mlx-lm): the
# whole point of logical model names is that this script does not care
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
#   scripts/smoke-llms.sh krick-1:8461 "$KEY" default-llm qwen-7b
#
# Needs grpcurl, jq, python3. Exits nonzero if Tokenize or StreamInfer
# fails for any tested model.
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

# Tokenize one prompt; prints "<n_ids> <n_tokens> <first_token>" or fails.
tokenize_once() {
    local model="$1" out
    local req
    req=$(jq -n --arg m "$model" \
        '{model_name:$m, texts:["Hello, inferstream!"]}')
    out=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" -d "$req" "$ADDR" \
        inferstream.v1.InferstreamService/Tokenize) || return 1
    printf '%s' "$out" | python3 -c '
import json, sys
r = json.load(sys.stdin)
encs = r.get("encodings", [])
if not encs:
    sys.exit("no encodings")
ids = encs[0].get("ids") or encs[0].get("inputIds") or []
toks = encs[0].get("tokens") or []
print(len(ids), len(toks), (toks[0] if toks else "-"))
'
}

# One short streamed completion; prints "<n_chunks> <final> <latency_ms>".
stream_once() {
    local model="$1" t0 t1
    local b64
    b64=$(python3 -c 'import base64,struct; t=b"Say hello in five words or fewer."; print(base64.b64encode(struct.pack("<I",len(t))+t).decode())')
    local req
    req=$(jq -n --arg m "$model" --arg b64 "$b64" \
        '{model_name:$m, id:"smoke-llms", inputs:[{name:"text", datatype:"BYTES", shape:[1]}], raw_input_contents:[$b64], parameters:{max_tokens:{int64Param:"16"}}}')
    t0=$(python3 -c 'import time; print(time.time_ns())')
    local out
    out=$(grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" -d "$req" "$ADDR" \
        inference.GRPCInferenceService/ModelStreamInfer) || return 1
    t1=$(python3 -c 'import time; print(time.time_ns())')
    printf '%s' "$out" | python3 -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    sys.exit("empty stream")
# grpcurl may emit one JSON object per chunk, or a JSON array.
if raw.startswith("["):
    objs = json.loads(raw)
else:
    objs = json.loads("[" + raw.replace("}\n{", "},{") + "]")
chunks, final = 0, False
for obj in objs:
    r = obj.get("inferResponse", obj)
    if r.get("errorMessage"):
        sys.exit(r["errorMessage"])
    if r.get("parameters", {}).get("final", {}).get("boolParam"):
        final = True
    if r.get("rawOutputContents"):
        chunks += 1
print(chunks, str(final).lower(), end=" ")
'
    echo $(( (t1 - t0) / 1000000 ))
}

PASS=0
FAIL=0
FAILED_MODELS=()
printf '\n%-16s %8s %8s %8s %10s %10s\n' MODEL TOK_IDS CHUNKS FINAL COLD_MS WARM_MS
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
    cold_ms=${cold_run##* }
    set -- ${warm_run}
    chunks=$1
    final=$2
    warm_ms=$3
    if [ "$chunks" -ge 1 ] && [ "$final" = "true" ]; then
        printf '%-16s %8s %8s %8s %10s %10s\n' "$model" "$tok_ids" "$chunks" "$final" "$cold_ms" "$warm_ms"
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
