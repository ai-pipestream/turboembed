# Test data

| File | Source |
|---|---|
| `WordBreakProperty-18.0.0.txt`, `WordBreakTest-18.0.0.txt` | UCD 18.0.0, `ucd/auxiliary/`, copied unchanged |

The Unicode files are used under the Unicode License v3
(`analysis/unicode/ucd/LICENSE`). `WordBreakProperty-18.0.0.txt` is also
the source of the table in `src/contractions/word_break/table.rs`; the
unit test `table_matches_the_property_file` compares the two.
