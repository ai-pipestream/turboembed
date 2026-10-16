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
