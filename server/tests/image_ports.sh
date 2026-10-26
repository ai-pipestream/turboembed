#!/usr/bin/env bash
# Run from the repository root with the built image as the first argument.
set -euo pipefail
image=${1:?image required}
# These files accompany the actual binaries, not just the source archive.
docker run --rm --entrypoint cat "$image" /opt/turbo/licenses/NOTICE | cmp NOTICE -
docker run --rm --entrypoint cat "$image" /opt/turbo/licenses/LICENSE | cmp LICENSE -
name="turbo-image-ports-$$"
cleanup() {
  docker logs "$name" 2>/dev/null || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
wait_healthy() {
  for _ in $(seq 90); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$name")
    if [ "$status" = healthy ]; then return; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]; then
      echo 'server exited before becoming healthy' >&2
      return 1
    fi
    sleep 2
  done
  echo 'server never became healthy' >&2
  return 1
}
# The environment changes both processes, without a custom health command.
docker run -d --name "$name" --health-interval=1s \
  -e TURBO_KSERVE_LISTEN=0.0.0.0:9090 \
  -v "$PWD/testdata/tiny-bert-bundle:/bundles/tiny-bert-bundle:ro" "$image"
wait_healthy
docker exec "$name" turbo-kserve health --addr 127.0.0.1:9090 --service tiny-bert-bundle
cleanup
# The CLI changes only the server; the documented command changes the probe.
docker run -d --name "$name" --health-interval=1s \
  --health-cmd='turbo-kserve health --addr 127.0.0.1:9091' \
  -v "$PWD/testdata/tiny-bert-bundle:/bundles/tiny-bert-bundle:ro" \
  "$image" --listen 0.0.0.0:9091
wait_healthy
docker exec "$name" turbo-kserve health --addr 127.0.0.1:9091 --service tiny-bert-bundle
