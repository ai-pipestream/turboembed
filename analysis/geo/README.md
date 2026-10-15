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
