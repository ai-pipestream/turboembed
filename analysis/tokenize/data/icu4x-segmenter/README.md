# Word dictionaries and LSTM models for Thai, Lao, Khmer and Burmese

The dictionaries and bidirectional LSTM models of ICU4X's segmenter data,
which the complex-script word segmenter (`turbo_tokenize::complex`)
bundles. They are used under the Unicode License v3 and the third-party
notices in `LICENSE` in this directory, which must travel with them.

| File | Script | Kind | Entries / dims | Alphabet | ICU4X name |
|---|---|---|---|---|---|
| `thai.dict` | Thai | Dictionary | 26383 words | 76 | `thaidict` |
| `lao.dict` | Lao | Dictionary | 30550 words | 93 | `laodict` |
| `khmer.dict` | Khmer | Dictionary | 81028 words | 96 | `khmerdict` |
| `burmese.dict` | Myanmar | Dictionary | 41120 words | 82 | `burmesedict` |
| `thai.lstm` | Thai | LSTM (`f32`) | `E=40, H=27` | 73 | `Thai_codepoints_exclusive_model4_heavy` |
| `lao.lstm` | Lao | LSTM (`f32`) | `E=40, H=27` | 72 | `Lao_codepoints_exclusive_model4_heavy` |
| `khmer.lstm` | Khmer | LSTM (`f32`) | `E=40, H=27` | 87 | `Khmer_codepoints_exclusive_model4_heavy` |
| `burmese.lstm` | Myanmar | LSTM (`f32`) | `E=40, H=27` | 187 | `Burmese_codepoints_exclusive_model4_heavy` |

By default (`turbo_tokenize::complex::bundled`), Thai uses `thai.lstm`
and Lao, Khmer and Burmese use `lao.dict`, `khmer.dict` and
`burmese.dict`. Both families are bundled so callers and evaluations can
construct either per script (`bundled_dictionaries`, `bundled_lstms`,
`ComplexSegmenter::with_segmenters`).

## Source

From `unicode-org/icu4x` at commit `bb3e1f9a0c966f7334aa400066f520f64f3dc3d6`
(2026-10-14):

- Dictionaries: `provider/data/segmenter/data/segmenter_dictionary_extended_v2.rs.data`,
  SHA-256 `689a5800da82b3d20b52cc64fa92b2e77d6749c3e2946cfcd3b2e4a2a829cf5a`.
  The word lists are ICU4C's `icu4c/source/data/brkitr/dictionaries/*dict.txt`,
  and the words of each file equal those of ICU4C's lists at tag
  `release-79.1rc` (commit `c4eb46bbb0da3b06ed1f6975babddd593ce4514e`)
  exactly, which `matches_icu4c_word_lists` in
  `analysis/tokenize/tests/complex_data.rs` checks when given those files.
- LSTM models: `provider/data/segmenter/data/segmenter_lstm_auto_v1.rs.data`,
  SHA-256 `f13932d01ebba5fb26a27dc14cf135b8f09e14fd0e360214f28882d5b6e9ec5d`.

## Format

### `.dict` (`TZD1`)

Each `.dict` file is the alphabet and the trie of one
`DictionaryBreakData::ZeroTrie` entry, unchanged:

| Bytes | Contents |
|---|---|
| 4 | the magic `TZD1` |
| 4 | the number `R` of alphabet ranges (little-endian) |
| `8 R` | each range: its first code point and the code point after its last, little-endian `u32` |
| rest | the `ZeroTrieSimpleAscii` store, whose bytes are the positions of the characters in the alphabet |

### `.lstm` (`TZL1`)

Each `.lstm` file is the alphabet and the nine `f32` weight slices of one
`LstmData::Float32` (`ModelType::Codepoints`) entry:

| Bytes | Contents |
|---|---|
| 4 | the magic `TZL1` |
| 4 | the number `R` of alphabet ranges (little-endian) |
| `8 R` | each range: its first code point and the code point after its last, little-endian `u32` |
| 4 | `embed_dim` (`40`, little-endian `u32`) |
| 4 | `hunits` (`27`, little-endian `u32`) |
| rest | the 9 little-endian `f32` arrays: `embedding` (`[V+1, 40]`), `fw_w` (`[4, 27, 40]`), `fw_u` (`[4, 27, 27]`), `fw_b` (`[4, 27]`), `bw_w` (`[4, 27, 40]`), `bw_u` (`[4, 27, 27]`), `bw_b` (`[4, 27]`), `time_w` (`[2, 4, 27]`), `time_b` (`[4]`) |

## Regenerating

```sh
TURBO_ICU4X_SEGMENTER_DATA=<icu4x>/provider/data/segmenter/data/segmenter_dictionary_extended_v2.rs.data \
TURBO_ICU4X_LSTM_DATA=<icu4x>/provider/data/segmenter/data/segmenter_lstm_auto_v1.rs.data \
TURBO_ICU4X_WRITE=1 \
cargo test -p turbo-tokenize --test complex_data -- --ignored regenerate
```

Without `TURBO_ICU4X_WRITE=1` the same tests compare the committed files
with what the data gives. `SHA256SUMS` lists each file's digest.
