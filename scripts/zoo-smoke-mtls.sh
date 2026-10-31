#!/usr/bin/env bash
# Sourced by test-zoo-image.sh, using its private scratch and Compose project.
# Separate server/client CAs prove that a server key grants no client access.
mkdir -p "$scratch/tls"
for role in server client; do
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 1 \
        -subj "/CN=zoo-smoke-$role-ca" -addext 'basicConstraints=critical,CA:TRUE' \
        -addext 'keyUsage=critical,keyCertSign,cRLSign' \
        -keyout "$scratch/tls/$role-ca.key" -out "$scratch/tls/$role-ca.pem" 2>/dev/null
done
for role in server client rogue; do
    ca=client
    usage=clientAuth
    if [[ "$role" == server ]]; then ca=server; usage=serverAuth; fi
    if [[ "$role" == rogue ]]; then ca=server; fi
    openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -subj /CN=zoo.test \
        -keyout "$scratch/tls/$role.key" -out "$scratch/tls/$role.csr" 2>/dev/null
    printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=%s\nsubjectAltName=DNS:zoo.test\n' \
        "$usage" > "$scratch/tls/$role.ext"
    openssl x509 -req -in "$scratch/tls/$role.csr" -CA "$scratch/tls/$ca-ca.pem" \
        -CAkey "$scratch/tls/$ca-ca.key" -set_serial "$((RANDOM + 1))" -days 1 \
        -extfile "$scratch/tls/$role.ext" -out "$scratch/tls/$role.pem" 2>/dev/null
done
# Only leaf keys and public CAs enter the test container, never CA signing keys.
rm "$scratch/tls/"*-ca.key "$scratch/tls/"*.csr "$scratch/tls/"*.ext
chmod 755 "$scratch/tls"
chmod 644 "$scratch/tls/"*
cat > "$scratch/mtls.yml" <<YAML
services:
  zoo:
    environment:
      TURBO_KSERVE_TLS_CERT: /tls/server.pem
      TURBO_KSERVE_TLS_KEY: /tls/server.key
      TURBO_KSERVE_TLS_CLIENT_CA: /tls/client-ca.pem
      TURBO_ZOO_TLS_CA: /tls/server-ca.pem
      TURBO_ZOO_TLS_CERT: /tls/client.pem
      TURBO_ZOO_TLS_KEY: /tls/client.key
      TURBO_ZOO_TLS_NAME: zoo.test
    volumes: !override
      - zoo-mtls-home:/zoo
      - $scratch/zoo.json:/smoke/zoo.json:ro
      - $scratch/tls:/tls:ro
volumes:
  zoo-mtls-home:
YAML
compose up -d --wait --wait-timeout 180 zoo
grpc="$(compose port zoo 8081)"
page="http://$(compose port zoo 8080)"
TURBO_ZOO_SMOKE_ADDR="$grpc" TURBO_ZOO_SMOKE_INDEX="$scratch/zoo.json" \
    TURBO_ZOO_SMOKE_TLS_DIR="$scratch/tls" TURBO_ZOO_SMOKE_PAGE="$page" \
    cargo test --locked -p turbo-kserve --test zoo_image_smoke -- --nocapture
# This is a separate process, like Docker HEALTHCHECK, not inherited exports.
compose exec -T zoo zoo-health
