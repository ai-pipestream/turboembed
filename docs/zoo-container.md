# The model zoo container

The turbo-zoo image is a self-contained model zoo for a team or a LAN.
It runs `turbo-kserve` serving only its model catalog, in zoo mode
(docs/grpc-catalog.md, Zoo mode), and the demo page (docs/demo.md) in
front of it. In the page, a person browses the models, reads each one's
licence and accepts its terms, and can save a model's files; a model
that may be used only non-commercially says so beside its licence. The
turboembed library on any machine then reads the models from the zoo
over gRPC, with `TURBO_ZOO_SOURCE=grpc://HOST:8081` (docs/zoo.md). The
library checks every file by size, SHA-256 and SHA-512, exactly as it
checks a download from anywhere else.

The page acts for one user, `TURBO_DEMO_USER` (`zoo` unless set; see
docs/demo.md, The user): the acceptances it records are that user's. A
library that reads an accept-first model from the zoo names the same
user, since the zoo checks the acceptances of the user a read names.

The image holds no model. A model is fetched from its origin the first
time a user who is permitted to have it reads it or installs it, and
then served from the zoo's cache. A share-alike, non-commercial or other
accept-first model therefore reaches a person only after they accept its
terms, never as part of the image.

A redistributable model is also kept in the zoo's store, so another zoo
sharing the bucket, or this one after its cache is gone, reads the
stored copy. A `CLASS_LOCAL_ONLY` model is never put in the store or
read from it, whoever has accepted it: its class allows no copy beyond
the cache it is installed into, so each zoo fetches it from its
publisher (docs/grpc-catalog.md).

## With RustFS: docker compose

`docker-compose.zoo.yml` at the repository root runs three services:

- RustFS, an S3-compatible store, pinned by digest;
- `bucket`, which makes the store's bucket once;
- `zoo`, the image itself, which keeps every redistributable model it
  fetches in that bucket under `blobs/sha256/<hex>` (docs/zoo.md,
  Stores).

```
export TURBO_ZOO_S3_ACCESS_KEY=... TURBO_ZOO_S3_SECRET_KEY=...
docker compose -f docker-compose.zoo.yml up -d --build
```

The page is at `http://HOST:8080` and gRPC at `HOST:8081`.

| Variable | Meaning | Absent |
|---|---|---|
| `TURBO_ZOO_S3_ACCESS_KEY`, `TURBO_ZOO_S3_SECRET_KEY` | RustFS's keys, which the zoo also signs with. | Required. |
| `TURBO_DEMO_TOKEN` | The bearer token the page's calls carry; the page asks for it. | The zoo makes one at each start and logs it once (`docker compose logs zoo`). |
| `TURBO_DEMO_USER` | Who the page's acceptances and installs are for. | `zoo`. |
| `TURBO_ZOO_PREFETCH` | `permissive` installs every permissive model into the store at start (docs/grpc-catalog.md, Prefetch). | Nothing is fetched until someone asks. |
| `TURBO_ZOO_BIND` | The host address both ports are published on. | `0.0.0.0`, every interface. |
| `TURBO_ZOO_PAGE_PORT`, `TURBO_ZOO_GRPC_PORT` | The host ports. | `8080` and `8081`. |
| `TURBO_ZOO_BUCKET` | The bucket's name. | `turbo-zoo`. |
| `TURBO_ZOO_RUSTFS_IMAGE` | The same RustFS image under another name, such as a registry mirror. | The pinned image. |

The volumes `rustfs-data` (the store) and `zoo-home` (the zoo's cache and
acceptance file) outlive the containers. `docker compose down -v` removes
them.

## With another S3, or with no store: docker run

The image alone keeps its cache and acceptances in the `/zoo` volume.
With nothing else set, each model is fetched from its origin into that
cache:

```
docker build -f server/Dockerfile.zoo -t turbo-zoo .
docker run -d -p 8080:8080 -p 8081:8081 -v turbo-zoo:/zoo turbo-zoo
```

To keep models in a bucket you already run, name it as the catalog's
store and pass the usual AWS variables. The bucket must already exist.
`zoo-bucket`, which is in the image, makes it at an S3-compatible
endpoint.

```
docker run -d -p 8080:8080 -p 8081:8081 -v turbo-zoo:/zoo \
    -e TURBO_KSERVE_CATALOG_STORE=s3://my-bucket/zoo \
    -e AWS_ENDPOINT_URL=https://s3.example.internal -e AWS_REGION=us-east-1 \
    -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
    turbo-zoo
```

Any `turbo-kserve` flag after the image name is passed to the server
(docs/grpc.md, Flags), and the `TURBO_KSERVE_*` and `TURBO_DEMO_*`
variables work as they do for each binary. The exceptions are the
server's listen address and TLS: the page and the health check dial
`http://127.0.0.1:8081`, so keep `--listen` and the `--tls-*` flags as
the image sets them (Who can do what, below). The image sets these:

| Variable | Value |
|---|---|
| `TURBO_KSERVE_LISTEN` | `0.0.0.0:8081` |
| `TURBO_KSERVE_CATALOG_DIR` | `/zoo` |
| `TURBO_KSERVE_ZOO` | `1` |
| `TURBO_DEMO_LISTEN` | `0.0.0.0:8080` |
| `TURBO_DEMO_SERVER` | `http://127.0.0.1:8081` |
| `TURBO_DEMO_ALLOW` | `RecordLicenceAcceptance`, `RevokeLicenceAcceptance` and `InstallModel` |
| `TURBO_DEMO_USER` | `zoo`, set by `zoo-start` when unset |

## What runs in the image

The entrypoint, `zoo-start` (server/zoo/zoo-start), starts `turbo-kserve`
and `turbo-kserve-demo` side by side. When either one ends, it stops the
other, so the container ends and its restart policy applies. The image
runs as user 10001. Its health check, `zoo-health`, passes once the
server answers SERVING (`turbo-kserve health`) and the page answers on
the port `TURBO_DEMO_LISTEN` names, over HTTPS when `TURBO_DEMO_TLS_CERT`
is set. It checks liveness on loopback, so it does not check the page
certificate's name, which is the zoo's own host.

## Who can do what

The page's calls need its token. Over plain HTTP the token crosses the
network in the clear; serve the page with `TURBO_DEMO_TLS_CERT` and
`TURBO_DEMO_TLS_KEY` off a trusted network. gRPC on port 8081 needs no
token. In zoo mode any caller records an acceptance, and the user a
request names is the caller's own word (docs/grpc-catalog.md, Zoo
mode). That suits a personal or LAN zoo.

The image does not yet support the server's own TLS or mutual TLS. The
page and the health check reach the server over plain gRPC on
loopback, so setting `TURBO_KSERVE_TLS_CERT`, `TURBO_KSERVE_TLS_KEY` or
`TURBO_KSERVE_TLS_CLIENT_CA` leaves the page failing every call and the
container unhealthy (#1686). Until then, a shared zoo publishes 8081
only on a trusted network, or behind a proxy that terminates TLS and
checks client certificates itself.

## The smoke test

`scripts/test-zoo-image.sh` runs the compose file against an image that
is already built, with a small index whose own addresses refuse every
connection, and a bucket seeded with that index's files. Every byte a
read returns has therefore come through the store.
`server/tests/zoo_image_smoke.rs` then checks over gRPC:

- the zoo starts with no model;
- a share-alike file is refused to a user who has not accepted it;
- the user records their own acceptance without `--admin`;
- each file reads through from the store and verifies.

`zoo-remote/tests/zoo_image.rs` then loads the same models with the
library, through `TURBO_ZOO_SOURCE=grpc://...` into a local cache of its
own, as a user who has accepted nothing yet:

- a permissive model loads, each file verified;
- a share-alike model is refused locally before the zoo is asked;
- accepted locally but not on the zoo, the zoo sends nothing;
- accepted on both sides, it loads;
- the local cache then serves both models with no source at all.

The script also checks that the page is served and that the page's
calls need the token, and that with the token the page acts for the
image's default user. Then it runs the image alone with the page on
HTTPS, using a certificate made for the run, and waits for Docker to
report the container healthy. The `Zoo image` workflow
(.github/workflows/zoo-image.yml) builds the image and runs the script
whenever the image, the catalog or the demo changes.
