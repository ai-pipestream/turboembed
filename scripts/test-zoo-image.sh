#!/usr/bin/env bash
# The zoo image's smoke test (docs/zoo-container.md): docker-compose.zoo.yml
# with RustFS, run against an image already built (TURBO_ZOO_IMAGE, default
# turbo-zoo:ci; docker build -f server/Dockerfile.zoo -t turbo-zoo:ci .).
# The zoo serves a small index whose own addresses answer nothing, and the
# bucket is seeded with its files by SHA-256, so every byte a read returns
# came through the store. Then server/tests/zoo_image_smoke.rs reads them
# over gRPC, and the page is fetched once.
set -euo pipefail

cd "$(dirname "$0")/.."
image=${TURBO_ZOO_IMAGE:-turbo-zoo:ci}
project="turbo-zoo-smoke-$$"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/turbo-zoo-smoke.XXXXXXXX")
compose() {
    local extra=()
    if [[ -f "$scratch/mtls.yml" ]]; then extra=(-f "$scratch/mtls.yml"); fi
    docker compose -p "$project" -f docker-compose.zoo.yml -f "$scratch/smoke.yml" "${extra[@]}" "$@"
}
tls=""
cleanup() {
    if [[ -n "$tls" ]]; then docker rm -f "$tls" >/dev/null 2>&1 || true; fi
    if [[ -f "$scratch/smoke.yml" ]]; then
        compose logs --no-color zoo bucket 2>/dev/null | tail -n 50 || true
        compose down -v --remove-orphans >/dev/null 2>&1 || true
    fi
    rm -rf "$scratch"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TURBO_ZOO_S3_ACCESS_KEY="turbo$(openssl rand -hex 12)"
TURBO_ZOO_S3_SECRET_KEY="$(openssl rand -hex 24)"
TURBO_DEMO_TOKEN="$(openssl rand -hex 16)"
export TURBO_ZOO_S3_ACCESS_KEY TURBO_ZOO_S3_SECRET_KEY TURBO_DEMO_TOKEN TURBO_ZOO_BIND=127.0.0.1

# Two artifacts, one permissive and one share-alike, at addresses that
# refuse every connection (HTTPS to port 9 on loopback).
mkdir -p "$scratch/files"
printf 'hello from the zoo store\n' > "$scratch/files/hello.txt"
printf 'a share-alike word list\n' > "$scratch/files/sa.txt"
entry() {
    local path=$1 f="$scratch/files/$1"
    printf '{"path": "%s", "url": "https://127.0.0.1:9/%s", "size": %s, "sha256": "%s", "sha512": "%s"}' \
        "$path" "$path" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)" "$(sha512sum "$f" | cut -d' ' -f1)"
}
artifact() {
    printf '{"id": "%s", "kind": "KIND_LEXICON", "language": "en", "format": "FORMAT_TEXT", "version": "1",
      "license": "%s", "license_terms": "https://example.org/%s/LICENSE", "trained_on": [], "class": "%s",
      "attribution": "Smoke test data.", "files": [%s]}' "$1" "$2" "$1" "$3" "$4"
}
printf '{"zoo_version": 1, "artifacts": [%s, %s]}\n' \
    "$(artifact smoke-permissive Apache-2.0 CLASS_PERMISSIVE "$(entry hello.txt)")" \
    "$(artifact smoke-sa CC-BY-SA-4.0 CLASS_SHARE_ALIKE "$(entry sa.txt)")" > "$scratch/zoo.json"
chmod a+r "$scratch/zoo.json"

# The smoke run's own image, index, and loopback ports chosen by Docker.
cat > "$scratch/smoke.yml" <<YAML
services:
  rustfs:
    ports: ["127.0.0.1::9000"]
  bucket:
    image: $image
    build: !reset null
  zoo:
    image: $image
    environment:
      TURBO_KSERVE_CATALOG_INDEX: /smoke/zoo.json
    volumes:
      - $scratch/zoo.json:/smoke/zoo.json:ro
    ports: !override
      - "127.0.0.1::8080"
      - "127.0.0.1::8081"
YAML

compose up -d --wait --wait-timeout 180 zoo
rustfs="http://$(compose port rustfs 9000)"
grpc="$(compose port zoo 8081)"
page="http://$(compose port zoo 8080)"

# Seed the bucket by SHA-256, as the zoo keeps files; the keys go to curl
# on its standard input.
for f in "$scratch"/files/*; do
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    printf 'user = "%s:%s"\n' "$TURBO_ZOO_S3_ACCESS_KEY" "$TURBO_ZOO_S3_SECRET_KEY" |
        curl --disable --noproxy '*' --config - --silent --show-error --fail \
            --aws-sigv4 aws:amz:us-east-1:s3 --header "x-amz-content-sha256: $sha" \
            --upload-file "$f" "$rustfs/turbo-zoo/zoo/blobs/sha256/$sha"
done

TURBO_ZOO_SMOKE_ADDR="$grpc" TURBO_ZOO_SMOKE_INDEX="$scratch/zoo.json" \
    cargo test --locked -p turbo-kserve --test zoo_image_smoke -- --nocapture

# The page is served, and its calls need the token.
curl --noproxy '*' --silent --show-error --fail "$page/" | grep -q '<html'
code=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' --request POST \
    --header 'content-type: application/proto' "$page/turbo.training.v1.TurboCatalogService/ListModelCatalog")
if [[ $code != 401 ]]; then
    echo "a page call with no token answered HTTP $code, not 401" >&2
    exit 1
fi

# Recreate the zoo with a fresh cache and mTLS, keeping the seeded RustFS store.
source scripts/zoo-smoke-mtls.sh

# The image alone with the page on HTTPS turns healthy: the health check
# follows the page's scheme. The certificate is made for this run.
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 1 -subj /CN=zoo.test \
    -keyout "$scratch/page.key" -out "$scratch/page.crt" 2>/dev/null
chmod a+r "$scratch/page.key" "$scratch/page.crt"
tls=$(docker run -d -e TURBO_DEMO_TLS_CERT=/tls/page.crt -e TURBO_DEMO_TLS_KEY=/tls/page.key \
    -v "$scratch/page.crt:/tls/page.crt:ro" -v "$scratch/page.key:/tls/page.key:ro" "$image")
health=starting
for ((attempt = 0; attempt < 90; attempt++)); do
    health=$(docker inspect -f '{{.State.Health.Status}}' "$tls")
    if [[ $health != starting ]]; then break; fi
    sleep 1
done
docker rm -f "$tls" >/dev/null
tls=""
if [[ $health != healthy ]]; then
    echo "the zoo with the page on HTTPS is $health, not healthy" >&2
    exit 1
fi
echo "zoo image smoke: ok"
