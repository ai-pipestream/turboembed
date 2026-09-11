#!/usr/bin/env bash
# Embed smoke against a RUNNING inferstream server: for every served alias
# (or the aliases passed as arguments), issue two Embed calls and report
# dimension, L2 norm, and cold vs warm latency. The first call on a fresh
# server is "cold" (model load + possible HF download); the second is "warm"
# (model hot in the backend).
#
#   ./scripts/smoke-embeddings.sh                       # every ListModels entry
#   ./scripts/smoke-embeddings.sh minilm bge-small      # just these aliases
#
# Env: INFERSTREAM_ADDR (default 127.0.0.1:8461),
#      INFERSTREAM_TOKEN (default change-me).
# Run from the repo root (proto paths are relative). Requires grpcurl.
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${INFERSTREAM_ADDR:-127.0.0.1:8461}"
TOKEN="${INFERSTREAM_TOKEN:-change-me}"
AUTH=(-H "authorization: Bearer $TOKEN")
EXT=(-proto crates/protocol/proto/inferstream_extension.proto)

if [ "$#" -gt 0 ]; then
    ALIASES=("$@")
else
    # Every model the server lists; mock/generation models fail Embed loudly,
    # which is the point of a smoke. (No mapfile: macOS ships bash 3.2.)
    ALIASES=()
    while IFS= read -r name; do
        ALIASES+=("$name")
    done < <(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
        inferstream.v1.InferstreamService/ListModels \
        | python3 -c 'import json,sys; [print(m["name"]) for m in json.load(sys.stdin).get("models",[])]')
fi

if [ "${#ALIASES[@]}" -eq 0 ]; then
    echo "no models to smoke (server listed none and no aliases given)" >&2
    exit 1
fi

embed_once() { # alias -> "<dim> <norm> <latency_ms>" on stdout
    local alias="$1" t0 t1 out
    t0=$(python3 -c 'import time; print(time.time_ns())')
    out=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
        -d '{"model_name":"'"$alias"'","texts":["inferstream embedding smoke"],"normalize":true}' \
        "$ADDR" inferstream.v1.InferstreamService/Embed)
    t1=$(python3 -c 'import time; print(time.time_ns())')
    printf '%s' "$out" | python3 -c 'import json,math,sys; v=json.load(sys.stdin)["embeddings"][0]["values"]; print(len(v), f"{math.sqrt(sum(x*x for x in v)):.4f}", end=" ")'
    echo $(( (t1 - t0) / 1000000 ))
}

FAIL=0
printf '%-18s %6s %8s %10s %10s\n' ALIAS DIM NORM COLD_MS WARM_MS
for alias in "${ALIASES[@]}"; do
    if ! cold_run=$(embed_once "$alias"); then
        printf '%-18s FAIL (see grpcurl error above)\n' "$alias"
        FAIL=1
        continue
    fi
    warm_run=$(embed_once "$alias")   # warm pass; dim/norm must be stable
    cold_ms=${cold_run##* }
    set -- ${warm_run}                # -> dim norm warm_ms
    printf '%-18s %6s %8s %10s %10s\n' "$alias" "$1" "$2" "$cold_ms" "$3"
done

[ "$FAIL" -eq 0 ] && echo "EMBED SMOKE OK" || { echo "EMBED SMOKE FAILED"; exit 1; }
