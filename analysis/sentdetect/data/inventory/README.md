# Classified abbreviation inventories

These tables are compiled into the library. They contain data, not upstream
matching code. Each row is `class<TAB>context rule<TAB>abbreviation`, sorted
by UTF-8 bytes. Duplicate spellings are merged. A language's four class
tables are initialized once on first use, and borrowed for later inventories.
English retains its existing static Rust tables and does not load these files.

Sources:

- CLDR release 48.2, commit `11299982335beb974c1c63c45265184e759c0f41`,
  `common/segments/{de,es,fr,it,pt,ru}.xml` from
  <https://github.com/unicode-org/cldr>. The unchanged source files and
  Unicode License v3 are in `../rules/cldr/`.
- Pragmatic Segmenter commit `358fe97e62902cd86fa6b85063c3bf91aa6c156a`,
  `lib/pragmatic_segmenter/languages/*.rb`, from
  <https://github.com/diasks2/pragmatic_segmenter>. MIT, Copyright (c) 2015
  Kevin S. Dias; notice in `LICENSE-pragmatic-segmenter`.
- pySBD commit `5905f13be4fc95f407b98392e0ec303617a33d86`,
  `pysbd/lang/slovak.py`, from <https://github.com/nipunsadvilkar/pySBD>.
  MIT, Copyright (c) 2019 Nipun Sadvilkar; notice in `LICENSE-pysbd`.

`generate.py` reads only the three literal abbreviation arrays in the
Pragmatic Segmenter and Slovak files. It trims and collapses spaces, adds
one final full stop, removes duplicates, and excludes entries longer than
64 UTF-8 bytes. CLDR entries must already have a final full stop. The
upstream numeric list supplies `Numeric`; its prepositive list supplies
`Prepositive`; other entries use `Trailing`. Project classifications in
the generator mark titles, references, connectives and company suffixes.
These are contextual lists, not unconditional no-break dictionaries.

For reproducibility, download those exact revisions, then run from the
repository root:

```sh
python3 analysis/sentdetect/data/inventory/generate.py "$PRAGMATIC_ROOT" "$SLOVAK_PY"
cargo test --locked -p turbo-sentdetect --test inventory_languages
```

Thirteen non-English languages have entries: Arabic, Bulgarian, Danish,
German, Spanish, French, Italian, Kazakh, Dutch, Polish, Portuguese,
Russian and Slovak. Other conformance languages whose upstream files
contain no local abbreviation arrays retain the existing rule profiles.
No English list is implicitly assigned to another language.
