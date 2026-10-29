# turbo-zoo-remote

Remote sources for the model zoo (#1613). The library can load a model from a
turboembed zoo server over gRPC, or from a list of sources tried in order,
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
  `/dir/blobs/sha256/<hex>` (`turbo_zoo::Blobs`);
- `grpc://host:port` or `grpcs://host:port`: a zoo server, read with its
  `ReadModelFile` call (`ZooServer`). The request carries the SHA-256 the
  local index pins, so a server whose index pins other bytes sends none.
  The server makes its own licence check for the same user;
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
  schemes (`s3://` comes later on #1613), relative `file://` paths and
  addresses with a path.
