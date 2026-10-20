# turbo-geo

The gazetteer API of the OpenNLP addons `geo` module and its gazetteers,
ported to Rust: the bundled Natural Earth table, GeoNames and Overture
tables the caller supplies, user tables with suppression rules, records
built in code, and the overlay. The crate documentation (`src/lib.rs`)
describes the index and where OpenNLP's behaviour is surprising;
`data/README.md` gives the bundled table's origin.

## Golden files

`tests/jvm_goldens.rs` compares the crate with the `gazetteer` goldens of
`analysis/goldens/data`, which `turbo-goldens generate` writes in the
recipe's container ("Gazetteer goldens" in
`analysis/goldens-tool/README.md`). Nothing calls the JVM at test time.

| Run | Gazetteer | Inputs |
|---|---|---|
| `naturalearth-populated-places` | `BundledGazetteer` | `names-*`, `ids`, `regions` |
| `gazetteer-geonames` | `GeoNamesGazetteer` over `fixtures/gazetteer-geonames.txt` | the same |
| `gazetteer-overture` | `OvertureGazetteer` over `fixtures/gazetteer-overture.txt` | the same |
| `gazetteer-user` | `UserGazetteer` over `fixtures/gazetteer-user.txt`, source `sites` | the same |
| `overlay` | `OverlayGazetteer`: the bundled gazetteer, the user table as additions, `fixtures/gazetteer-suppressions.txt` | the same |
| `load-geonames`, `load-overture`, `load-user`, `load-suppressions` | the loader of that kind of table | `table-<kind>-*` |

For every line of a `names` input the test compares the entries `lookup`
returns, in order, with every field (coordinates bit for bit, attributes
sorted by key); for `ids` the entry `by_id` returns, and for `regions` the
entry `by_region` returns or the message of the refused code. For a table
it compares whether it loads and, when it does not, OpenNLP's message, and
the rules of a suppression table. A second test checks that the bundled
table is the bytes of the resource in the OpenNLP jar (the SHA-512 in the
golden's model record).

## Place profiles

`PlaceProfiles` ports `opennlp.geo.PlaceProfiles`: a tab separated table of
numeric metrics per place (no data is bundled, in OpenNLP or here), each
column standardized to mean zero and unit population variance, and places
ranked by cosine similarity. The standardization is OpenNLP's `BigDecimal`
arithmetic, exact but for the same 34-digit roundings, so profiles and
similarities match the JVM bit for bit. `tests/profile_goldens.rs` compares
every `place-profiles` golden ("Place profile goldens" in
`analysis/goldens-tool/README.md`): the load result or OpenNLP's message,
the standardized profiles, the similarity of every pair and `mostSimilar`
with the counts 1, 2 and all.

## Load time and memory

`examples/load_report.rs` measures them:

```
cargo run --release -p turbo-geo --example load_report -- bundled
cargo run --release -p turbo-geo --example load_report -- synthesize 12400000 all.txt
cargo run --release -p turbo-geo --example load_report -- geonames all.txt all.snapshot
```

`synthesize` writes a made up table in the GeoNames format with as many
rows as `allCountries.txt` and a similar spread of alternate names and
scripts, for when the real file is not at hand.

## Names in running text

`name_image` (on `InMemoryGazetteer`, `BundledGazetteer`, `GeoNamesGazetteer`,
`OvertureGazetteer` and `UserGazetteer`) writes the gazetteer's distinct match
keys as one `turbo-glossary` automaton image, and `NameFinder` reads it in place
(from an `ImageBuf`, any 4-byte aligned slice or a mapped file) to find every
run of word tokens in a text whose match key is a name: the text's UAX #29 word
tokens are folded one by one as a name is, joined by one space, and the
automaton matches at token edges. `find_all` gives every name, nested and
overlapping ones included; `find` keeps the leftmost, then the longest.
`lookup_key` gives a match's records without folding again. `NameAnnotator`
writes the `find` matches as `location` entities, so `GeocodeAnnotator` and
`GeoEntityLinker` can resolve mentions no model tagged; use the mention policy
below with it.

`tests/names.rs` checks the finder against the gazetteer itself: on the bundled
table and on random gazetteers of accented, Greek, Han and punctuated names, a
match is exactly a run of tokens whose `lookup` finds a record, and its key's
records are that lookup's. `examples/names_bench.rs` times it against looking
up every run of tokens up to the longest name with `Gazetteer::lookup`, which
finds the same names, the two alternating run by run. On a Zen 5 desktop, one
pinned core, release, load average 13 to 26 of 32 threads from other work,
about 2 MB of text:

| Gazetteer | Keys | Longest | Heap | Image | Image load | Lookup of every run | Name finder |
|---|---|---|---|---|---|---|---|
| bundled | 7,698 | 7 tokens | 1.8 MB | 1.0 MB | 0.25 ms | 402 ms | 45 ms |
| GeoNames `cities15000` | 313,840 | 112 tokens | 17.9 MB | 41.8 MB | 16 ms | 95,087 ms | 163 ms |

The image of a table with many alternate names in many scripts is larger
than the index it was written from: the index stores each key once, the
automaton a state per character of every key.

## Dictionary mention admission

Gazetteer aliases are lookup keys, not evidence that every matching token
is a place. The bundled `S` and `C` aliases can resolve possessive `s` or a
scientific `c` to Sao Paulo or Cordoba with a population-prior score of 0.9.
That score describes candidate separation, not entity recognition accuracy.

For dictionary-generated candidates, use
`GeocodeAnnotator::new(geocoder).with_mention_policy(MentionPolicy::ExcludeSingleAsciiLetters)`.
This rejects single ASCII letters before the geocoder receives any context.
It keeps original entity annotations and preserves eligible mention spans.
The default `All` policy retains OpenNLP behavior. Non-ASCII names remain
eligible; legitimate single-letter ASCII names require `All` or a separate
pipeline. This narrow filter does not replace contextual disambiguation.
