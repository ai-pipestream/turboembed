# turbo-zoo-remote

Remote sources for the model zoo (#1613). The library can load a model from a
turboembed zoo server over gRPC, from an S3 bucket (AWS, RustFS), or
from a list of sources tried in order,
the way DJL loads a model from an `s3://` address. The local zoo keeps its
own index, cache and licence check. Every file from any source is checked
by size, SHA-256 and SHA-512 before it is kept.

```rust
use turbo_zoo::{Acceptances, Http, Index, Policy, Zoo};
use turbo_zoo_remote::{ENV_SOURCE, sources};

let spec = std::env::var(ENV_SOURCE).unwrap_or_else(|_| "origin".into());
let source = sources(&spec, &user, Http::new())?;
let zoo = Zoo::new(Index::builtin(), cache_dir, Policy::default(), &source);
let fetched = zoo.fetch("segment-any-text-sat-3l", &user, &acceptances)?;
```

`TURBO_ZOO_SOURCE` takes a comma-separated list of sources:

- `file:///dir`: a content-addressed store, with each file at
  `/dir/blobs/sha256/<hex>` (`turbo_zoo::Blobs`). The first one named
  also keeps every file a later source serves, once it verifies, so the
  next read is local and the store can seed a bucket or a server;
- `grpc://host:port` or `grpcs://host:port`: a zoo server, read with its
  `ReadModelFile` call (`ZooServer`). The request carries the SHA-256 the
  local index pins, so a server whose index pins other bytes sends none.
  The server makes its own licence check for the same user;
- `s3://bucket/prefix`: an S3 bucket holding the same
  `prefix/blobs/sha256/<hex>` layout (`S3Bucket`), read with one
  SigV4-signed GET per file, or unsigned when no key is set. It is
  configured the way the AWS tools are: `AWS_ENDPOINT_URL_S3` or
  `AWS_ENDPOINT_URL` for RustFS or another S3-compatible store
  (addressed path-style), `AWS_REGION` or `AWS_DEFAULT_REGION` (else
  `us-east-1`), and `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `AWS_SESSION_TOKEN`. A local store syncs to a bucket as is
  (`aws s3 sync /dir s3://bucket/prefix`);
- `origin`: the index's own addresses, through the transport the caller
  passes (HTTPS, for one). Leave it out to fetch nothing from those
  addresses.

The first source whose bytes verify is kept. No source is asked for an
artifact whose terms the user has not accepted locally.

`ZooServer` blocks on a runtime of its own, so code that is already
running async calls the zoo from a blocking task. A server waits for open
connections when it stops, so drop the source before stopping a server it
reads from.

## Tests

`tests/server.rs` starts a `turbo-kserve` catalog in-process on a tiny
index served from memory, so nothing is downloaded. It checks that:

- the library loads a two-file model from the server, and the local cache
  then serves it with the server gone;
- a share-alike model needs its terms accepted both locally and on the
  server;
- a server whose index pins other bytes sends none;
- a source list falls through a server that is down to a local store;
- the list parser refuses empty entries, `origin` named twice, unknown
  schemes, relative `file://` paths and addresses with a path.

The S3 bucket itself, `S3Bucket`, is turbo-zoo's (docs/zoo.md), so a
zoo server can keep its files in one too; its tests are turbo-zoo's
`tests/s3.rs`.

## Features

- `tls` (default): `grpcs://`. Without it a `grpcs://` source is refused.
- `s3` (default): `s3://`, through turbo-zoo's `http` feature. Without
  it an `s3://` source is refused.

Both reach `ring`, so the arm64 check in CI builds the crate with
neither.

## RustFS integration

`bash scripts/test-rustfs.sh` starts an isolated RustFS service, exercises signed
downloads, uploads, access checks, digest verification and the local cache, then removes the
service and its generated data. It requires Docker, curl with SigV4 support,
OpenSSL and Rust. `TMPDIR` chooses the temporary data directory. The RustFS CI
workflow runs this check for relevant changes and daily; every run pulls
`rustfs/rustfs:latest` and reports the resolved image digest.
