#!/usr/bin/env bash
# Build the server and client, start the server on the mock bundle, call it,
# check that an over-capacity request is refused with INVALID_ARGUMENT, and
# stop the server. Exit non-zero on any failure.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bundle="${TURBO_DEMO_BUNDLE:-$root/testdata/bundles/mock/embedding}"
port="${TURBO_DEMO_PORT:-50551}"
cmake -S "$here" -B "$here/build" >/dev/null
cmake --build "$here/build" -j >/dev/null
"$here/build/turbo-grpc-server" --bundle "$bundle" --listen "127.0.0.1:$port" --sessions 2 --max-batch 4 "$@" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
    if "$here/build/turbo-grpc-client" --target "127.0.0.1:$port" ping >/dev/null 2>&1; then break; fi
    sleep 0.2
done
"$here/build/turbo-grpc-client" --target "127.0.0.1:$port" "a brown dog runs through the grass" "a dog is running on the lawn" "the stock market closed higher"
if "$here/build/turbo-grpc-client" --target "127.0.0.1:$port" one two three four five >/dev/null 2>"$here/build/over.err"; then
    echo "an over-capacity request was accepted" >&2; exit 1
fi
grep -q "Embed failed (3)" "$here/build/over.err" || { cat "$here/build/over.err" >&2; echo "expected INVALID_ARGUMENT (3)" >&2; exit 1; }
echo "over-capacity request refused as expected: $(cat "$here/build/over.err")"
echo "grpc demo: OK"
