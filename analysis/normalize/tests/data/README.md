# Test data

| File | Source |
|---|---|
| `WordBreakProperty-18.0.0.txt`, `WordBreakTest-18.0.0.txt` | UCD 18.0.0, `ucd/auxiliary/`, copied unchanged |

The Unicode files are used under the Unicode License v3
(`analysis/unicode/ucd/LICENSE`). The unit test
`table_matches_the_property_file` compares `turbo_unicode::word_break`
with `WordBreakProperty-18.0.0.txt`.
