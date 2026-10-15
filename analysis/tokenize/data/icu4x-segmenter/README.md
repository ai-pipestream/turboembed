# Word dictionaries for Thai, Lao, Khmer and Burmese

The dictionaries of ICU4X's segmenter data, which the dictionary word
tokenizer (`turbo_tokenize::complex`) bundles. They are used under the
Unicode License v3 and the third-party notices in `LICENSE` in this
directory, which must travel with them.

| File | Script | Words | Alphabet | ICU4X name |
|---|---|---|---|---|
| `thai.dict` | Thai | 26383 | 76 | `thaidict` |
| `lao.dict` | Lao | 30550 | 93 | `laodict` |
| `khmer.dict` | Khmer | 81028 | 96 | `khmerdict` |
| `burmese.dict` | Myanmar | 41120 | 82 | `burmesedict` |

## Source

ICU4X `provider/data/segmenter/data/segmenter_dictionary_extended_v2.rs.data`
at commit `bb3e1f9a0c966f7334aa400066f520f64f3dc3d6` of
`unicode-org/icu4x` (2026-10-14), SHA-256
`689a5800da82b3d20b52cc64fa92b2e77d6749c3e2946cfcd3b2e4a2a829cf5a`. That
data's README says it was generated with CLDR 49.0.0-BETA4 and Unicode
18.0.0; its generator reads the dictionaries from ICU's export data, tag
`release-79.1rc` (`TESTED_ICUEXPORT_TAG` in
`provider/source/src/lib.rs`). The CLDR version matters little here: the
word lists are ICU4C's `icu4c/source/data/brkitr/dictionaries/*dict.txt`,
and the words of each file equal those of ICU4C's lists at tag
`release-79.1rc` (commit `c4eb46bbb0da3b06ed1f6975babddd593ce4514e`)
exactly, which `matches_icu4c_word_lists` in
`analysis/tokenize/tests/complex_data.rs` checks when given those files.
The latest published crate, `icu_segmenter_data` 2.3.0, is older (CLDR
48.2.1, ICU 78.1rc) and stores these dictionaries in a different form
(`Char16Trie`); it is not used.

## Format

Each file is the alphabet and the trie of one
`DictionaryBreakData::ZeroTrie` entry, unchanged:

| Bytes | Contents |
|---|---|
| 4 | the magic `TZD1` |
| 4 | the number `R` of alphabet ranges (little-endian) |
| `8 R` | each range: its first code point and the code point after its last, little-endian `u32` |
| rest | the `ZeroTrieSimpleAscii` store, whose bytes are the positions of the characters in the alphabet |

The tokenizer reads the trie where it lies in the binary
(`include_bytes!`); only the alphabet ranges are copied when a
dictionary is opened.

## Regenerating

```sh
TURBO_ICU4X_SEGMENTER_DATA=<icu4x>/provider/data/segmenter/data/segmenter_dictionary_extended_v2.rs.data \
TURBO_ICU4X_WRITE=1 \
cargo test -p turbo-tokenize --test complex_data -- --ignored regenerate_from_icu4x
```

Without `TURBO_ICU4X_WRITE=1` the same test compares the committed
files with what the data gives. `SHA256SUMS` lists each file's digest.
