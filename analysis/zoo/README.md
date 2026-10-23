# turbo-zoo

The model zoo: a licence-aware index of models, classifier heads,
lexicons, label centroid sets and datasets, and a fetcher that enforces
the licence policy on #326. docs/zoo.md describes the classes, the index
and acceptance formats and the fetch rules; docs/licences.md is the
table of datasets and checkpoints the index is built from.

```
cargo run -p turbo-zoo -- list
cargo run -p turbo-zoo -- show opennlp-en-ud-ewt-sentence-1.3-2.5.4
cargo run -p turbo-zoo -- accept CC-BY-SA-4.0          # prints the licence text address
cargo run -p turbo-zoo -- accept CC-BY-SA-4.0 --yes    # records the acceptance
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

`index/zoo.json` is the built-in index: the 112 OpenNLP 1.3 UD models
trained on permissive and share-alike treebanks, the four English models
the MASC recipes train (`analysis/train-recipes`), at their addresses in
this repository's release `models-masc-3.0.0-1`, the ten Spanish,
Catalan and Finnish models the UD recipes train, at their addresses in
its release `models-ud-2.16-1`, and the language detector the Tatoeba
recipe trains, with `training-sentences.txt` (the ids of the sentences
it was trained on) as its second file, at their addresses in its release
`models-tatoeba-2026-10-10-1`, and the language detector the web recipe
trains on those sentences and FineWeb-2 and FineWeb text, with
`training-sentences.txt` and `training-documents.txt`, at their
addresses in its release `models-tatoeba-fineweb-2026-10-10-1`.
It also lists two published neural checkpoints at their Hugging Face
revisions: `fastino-gliner2-base-v1` (Apache-2.0, the GLiNER2 relation
scorer's, docs/gliner2-relations.md), and `jackboyla-glirel-large-v0`
(CC BY-NC-SA 4.0, local-only, `FORMAT_PYTORCH`), which the zoo fetches
from its publisher's own address only for a user who has accepted that
licence. Nothing in this repository bundles or ships it. Its tokenizer,
`microsoft-deberta-v3-large-tokenizer` (MIT, `FORMAT_SENTENCEPIECE`), is
the base model's `spm.model` and `config.json`
(docs/glirel-relations.md).
`.github/workflows/models.yml`, `models-ud.yml`, `models-tatoeba.yml`
and `models-fineweb.yml` create those releases when they are run by
hand on main; until then fetching those models fails. The recipes
write their entries: `turbo-train-recipes zoo-index` prints this index
with them added. The 32 models trained
on the eight non-commercial treebanks (four per treebank) are
local-only, which the policy allows the zoo to fetch, but the index does
not list them; docs/licences.md does, with the models the policy
refuses. The index's sizes and digests were computed from the published
files and matched against the `.sha256` and `.sha512` files Apache
publishes beside them.
`cargo test -p turbo-zoo -- --ignored` fetches every entry again and
checks it.

## Tests

- `tests/zoo.rs`: one artifact per class, served from memory. Refused
  artifacts, no-derivatives ones among them, are neither listed nor
  fetched; share-alike, LGPL and GPL artifacts are refused until the
  user accepts each licence they need, per user, rechecked on every
  call; restricted but public data needs its host's own terms accepted,
  and non-commercial data its licence, under either GPL policy; for
  both, and any model trained on them, every file's address in the
  index must lie under the entry's origin, a path on a host rather than
  the whole host, so an address on another host, elsewhere on the same
  host or at a sibling path is refused, as is a path with a `.` or `..`
  step, a `\` or `;` (plain or encoded), an encoded `/`, `%` or NUL, a
  malformed escape, or encoded bytes that are not valid UTF-8 (overlong
  forms, surrogates, five- and six-byte forms, bad continuation bytes),
  while well-formed UTF-8 at the edge of each length is kept; they are
  never shippable; the most restrictive licence decides the class;
  GPL acceptance can be turned off;
  a size, SHA-256 or SHA-512 mismatch fails and writes nothing; a corrupt
  cached file is fetched again; a symlink in the cache is refused; the
  index parser rejects every malformed field.
- `tests/cli.rs`: before recording an acceptance, `turbo-zoo accept`
  shows the text's address and, for local-only terms, what the terms
  allow apart from this project's policy, which accepting agrees to; it
  names a permissive or refused licence as such, a refused `LicenseRef-`
  one included, and offers nothing to accept, and `turbo-zoo revoke`
  names them the same way; `turbo-zoo show` gives the same two
  statements for a local-only artifact.
- `src/fetch.rs`, with the `http` feature: the HTTPS transport follows
  at most 10 redirects and refuses a plain `http://` address before
  connecting, the check `ureq` applies to every redirect too.
- `src/index.rs`: the path check accepts the percent-encoded UTF-8 of
  every scalar value except `/`, `\`, `%`, `;` and NUL, agrees with the
  standard library's UTF-8 validation on every one- and two-byte
  sequence, every sequence after a three-byte lead and four-byte
  sequences over every lead and second byte, refuses a character cut
  off at the end of the path, and refuses malformed escapes.
- `tests/builtin.rs`: the built-in index is canonical, holds nothing
  refused, gives each OpenNLP UD model its treebank's licence and class,
  lists nothing else but the models the MASC and UD recipes train, which
  need no acceptance, and matches the model files the repository checks
  in.
