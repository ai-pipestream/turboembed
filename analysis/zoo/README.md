# turbo-zoo

The model zoo: a licence-aware index of models, classifier heads,
lexicons, label centroid sets and datasets, and a fetcher that enforces
the licence policy on #326. docs/zoo.md describes the classes, the index
and acceptance formats and the fetch rules; docs/licences.md is the
table of datasets and checkpoints the index is built from.

```
cargo run -p turbo-zoo -- list
cargo run -p turbo-zoo -- show opennlp-en-ud-ewt-sentence-1.3-2.5.4
cargo run -p turbo-zoo -- accept CC-BY-SA-4.0
cargo run -p turbo-zoo -- fetch opennlp-en-ud-ewt-sentence-1.3-2.5.4
```

From code:

```rust
use turbo_zoo::{Acceptances, Http, Index, Policy, Zoo};

let zoo = Zoo::new(Index::builtin(), cache_dir, Policy::default(), &Http);
let acceptances = Acceptances::load(&home.join("acceptances.json"))?;
let fetched = zoo.fetch("opennlp-ca-ud-ancora-sentence-1.3-2.5.4", &user, &acceptances)?;
let model_path = fetched.single().unwrap();
```

`index/zoo.json` is the built-in index: the OpenNLP 1.3 UD models the
policy lets the zoo fetch. Its sizes and digests were computed from the
published files and matched against the `.sha256` and `.sha512` files
Apache publishes beside them. `cargo test -p turbo-zoo -- --ignored`
fetches every entry again and checks it.

## Tests

- `tests/zoo.rs`: one artifact per class, served from memory. Refused
  artifacts are neither listed nor fetched; share-alike, LGPL and GPL
  artifacts are refused until the user accepts each licence they need,
  per user, rechecked on every call; GPL acceptance can be turned off;
  a size, SHA-256 or SHA-512 mismatch fails and writes nothing; a corrupt
  cached file is fetched again; a symlink in the cache is refused; the
  index parser rejects every malformed field.
- `tests/builtin.rs`: the built-in index is canonical, holds nothing
  refused, gives each UD model its treebank's licence and class, and
  matches the model files the repository checks in.
