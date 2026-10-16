# turbo-encoding

Bytes in, a `Document` out, with every offset of every view mapping back
to the bytes of the file. The crate has three parts:

- **Decoders** (`src/decode/`) for the 40 encodings of the WHATWG
  Encoding Standard, written by hand from the standard's algorithms. Each
  records an `Alignment` from the bytes to the text as it decodes: a
  character that is not one byte for one byte is one edit, a byte order
  mark or an ISO-2022-JP escape sequence is a deletion, and a U+FFFD
  covers exactly the bytes it replaced.
- **Detection** (`src/detect/`), a port of chardetng 0.1.17 (Copyright
  Mozilla Foundation, Apache-2.0 or MIT), and `open`, which settles the
  encoding (byte order mark, then a declared label, then the detector),
  decodes, and records the result as the document layer
  `turbo:encoding` (`SourceEncoding`, in `turbo-encoding-layers`).
- **Mojibake repair** (`src/repair/`, `src/charmap.rs`), a port of the
  encoding fix of ftfy 6.3.1 by Robyn Speer (Apache-2.0), with every fix
  recorded as one edit. The `Mojibake` normalizer in `turbo-normalize`
  runs it as a view.

docs/analysis.md ("Source bytes") describes how the bytes sit below
view 0, and NOTICE credits the work these parts are made from. The crate
depends on the analysis crates and `turbo-unicode` only.

## Generated files

The tables are generated and committed; the downloads they come from are
not.

| Generated file | From | Generator |
|---|---|---|
| `src/tables/*.rs`, `analysis/encoding-layers/src/labels.rs` | The WHATWG Encoding Standard's `index-*.txt` files and `encodings.json` (`https://github.com/whatwg/encoding`, files dated 2024-09-18), BSD 3-Clause (`src/tables/NOTICE`) | `turbo-encoding-gen` |
| `src/repair/classes.rs`, `tests/data/ftfy_cases.tsv` | ftfy 6.3.1 (the sdist from PyPI): `ftfy/badness.py`, `ftfy/chardata.py`, `tests/test_cases.json` | `analysis/encoding-gen/ftfy.py` |
| `src/detect/data.rs` | chardetng 0.1.17 (the crate from crates.io): `src/data.rs`, CC0 1.0 | copied, with the encoding references changed to this crate's `Encoding` |
| `tests/data/chardetng_cases.tsv` | chardetng 0.1.17's tests (`src/lib.rs`), with each text encoded by the encoding_rs crate as those tests do | a one-off program, not kept; the file header says what each row is |

To regenerate the WHATWG tables, fetch every `index-*.txt` file and
`encodings.json` of the Encoding Standard into one directory, then:

```sh
cargo run -p turbo-encoding-gen -- <download directory> .
```

To regenerate the ftfy classes and fixture, unpack the ftfy 6.3.1 sdist
and run, from the repository root:

```sh
python3 -I analysis/encoding-gen/ftfy.py <ftfy-6.3.1 directory> .
```

The script needs Python 3 and nothing else (it stubs ftfy's `wcwidth`
import). Nothing at build or run time uses Python.

## Tests

- `tests/decode.rs`: windows-1252, Shift_JIS, EUC-KR, UTF-16LE and BE,
  EUC-JP, GBK, gb18030, Big5 and KOI8-R samples, each token mapped to its
  bytes and back, and each token's bytes decoded alone to the token;
  replacement characters and the bytes they replaced; byte order marks;
  empty input in every encoding; every cut of a multi-byte sequence at
  the end of the input; every edit of random input decoding alone to its
  text.
- `tests/detect.rs`: the 82 inputs of chardetng's own tests, in one
  buffer and in chunks of 1, 2, 3 and 7 bytes; the tests chardetng
  writes by hand (streaming, empty input); confidence; `settle` and
  `open`.
- `tests/repair.rs`: all of ftfy's test cases, with ftfy's own output
  and plan as the expectation, and the alignment of every repair.
- `tests/markup.rs`: a windows-1252 HTML file through a markup view, each
  extracted word mapped to the file's bytes.

The decoders, the detector and the repair were also compared with
encoding_rs, chardetng and ftfy on random input (over a million inputs
for the decoders and the detector, 100,000 strings for the repair). The
decoded texts, the guesses and scores, and the repaired texts agreed
everywhere; the decoders' replacement spans differ from encoding_rs only
where the standard and encoding_rs attribute an error differently (an
ISO-2022-JP escape sequence directly after another, and the replacement
encoding, whose one U+FFFD covers the whole input here). Those
comparisons need the original libraries and are not part of the test
suite.
