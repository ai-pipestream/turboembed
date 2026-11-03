#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
image=rustfs/rustfs:latest
scratch=$(mktemp -d "${TMPDIR:-/tmp}/turbo-rustfs.XXXXXXXX")
container=""
cleanup() {
    if [[ -n "$container" ]]; then docker rm -f "$container" >/dev/null; fi
    rm -rf "$scratch"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Resolve the current image on every run, then use that immutable image ID.
docker pull "$image"
image_id=$(docker image inspect "$image" --format '{{.Id}}')
docker image inspect "$image_id" --format 'RustFS image: {{.Id}} {{json .RepoDigests}}'
RUSTFS_ACCESS_KEY="turbo$(openssl rand -hex 12)"
RUSTFS_SECRET_KEY="$(openssl rand -hex 24)"
export RUSTFS_ACCESS_KEY RUSTFS_SECRET_KEY
container=$(docker run -d --rm --memory 1g --cpus 2 \
    -p 127.0.0.1::9000 \
    --tmpfs /data:rw,size=128m,uid=10001,gid=10001 \
    --tmpfs /logs:rw,size=16m,uid=10001,gid=10001 \
    -e RUSTFS_ACCESS_KEY -e RUSTFS_SECRET_KEY \
    -e RUSTFS_CONSOLE_ENABLE=false -e RUSTFS_OBS_LOG_DIRECTORY=/logs \
    "$image_id" /data)
port=$(docker port "$container" 9000/tcp)
export TURBO_RUSTFS_ENDPOINT="http://$port"
export TURBO_RUSTFS_SCRATCH="$scratch"

ready=false
for ((attempt=0; attempt<90; attempt++)); do
    # A signed bucket-list request checks API readiness, not just an open port.
    if printf 'user = "%s:%s"\n' "$RUSTFS_ACCESS_KEY" "$RUSTFS_SECRET_KEY" |
        curl --disable --noproxy '*' --config - --silent --fail --max-time 2 \
            --aws-sigv4 aws:amz:us-east-1:s3 "$TURBO_RUSTFS_ENDPOINT/" >/dev/null; then
        ready=true
        break
    fi
    if [[ $(docker inspect "$container" --format '{{.State.Running}}') != true ]]; then break; fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo 'RustFS did not become ready' >&2
    exit 1
fi

cargo test --locked -p turbo-zoo-remote --features rustfs-tests --test rustfs -- --nocapture
