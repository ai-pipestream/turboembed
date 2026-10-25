# Prebuilt lattice dictionaries

`MecabDictionary` accepts either a UTF-8 MeCab source directory or a
project-owned prebuilt file. Compile once with:

```sh
cargo run --locked --release -p turbo-tokenize --example compile_lattice -- SOURCE dictionary.tlat
```

Add `legacy` as the final argument for `WhitespaceMode::Legacy`. The default
is Unicode whitespace. `MecabDictionary::load(path, whitespace)` detects a
file and checks that its stored whitespace policy matches the requested
policy. `load_prebuilt(path, limits)` reads and validates the stored policy.
The existing source loader remains supported. This format does not read
MeCab's `sys.dic` or change its dictionary licensing requirements.

The prebuilt file contains the runtime trie, connection matrix, sorted
lexicon entries, unknown-word ranges, character assignments, and pooled
features. Loading it does not parse CSV, sort surfaces, construct the trie,
or expand character ranges. The large arrays and feature strings borrow
one shared image; small category, assignment and label metadata is owned.
Cloning a prebuilt dictionary shares those large tables. Scratch buffers
used by tokenization remain separate from the immutable dictionary.

`PrebuiltImage::read` reads a file into aligned storage. For a read-only file
mapping supplied by an application, `PrebuiltImage::from_owner(Arc<T>)`
retains its owner and `MecabDictionary::from_prebuilt(image, limits)` checks
it before exposing a dictionary. No mapping crate is added to the runtime.
The owner constructor is unsafe: it must return the same immutable slice
at the same address and length for its whole lifetime. A mapped file must
not be changed or truncated while any dictionary/image clone exists.
Contents and alignment still undergo normal validation. `copy_from` is the
safe alternative when an application cannot guarantee immutable storage.

This is not a constant-time loader. SHA-256, UTF-8, table bounds, parent
links and entry references are checked before use. The checksum detects
accidental corruption, not hostile replacement: it is not a signature.
The source digest records the compiler's source identity; a consumer that
requires a particular dictionary must compare it with an expected digest
from a trusted source.

## Format version 1

Integers are little-endian. Tables begin on eight-byte boundaries, with
zero padding. The current reader rejects big-endian hosts and unaligned
external storage. Source loading remains available on those hosts.

The 512-byte header starts with `TLATTICE` (eight bytes), version `u32`,
endian marker `0x01020304`, whitespace policy `u32` (Legacy 0, Unicode 1),
lexicon count, matrix rows and columns, default character category and
fallback assignment (all `u32`). Bytes 40..72 hold the source manifest
SHA-256. Bytes 72..312 hold fifteen `(offset: u64, length: u64)` pairs.
Remaining header bytes are reserved and zero. Sections follow in this
order, without gaps except alignment padding:

| Section | Representation |
|---|---|
| 0 | Matrix costs, `i16` |
| 1 | Double-array trie units, `(base, check): (u32, u32)` |
| 2 | Surface group entry boundaries, `u32` |
| 3 | Entries: left/right context `u32`, cost `i16`, two zero bytes, feature record `u32` |
| 4 | Unknown template ranges per category, `(start, end): (u32, u32)` |
| 5 | Supplementary ranges, `(start, end, assignment): (u32, u32, u32)` |
| 6 | 65,536 BMP assignment indices, `u32` |
| 7 | Pooled feature UTF-8 bytes |
| 8 | Feature column end byte offsets, `u32` |
| 9 | Feature record column boundaries, `u32` |
| 10 | Entry POS tag indices, `u16` |
| 11 | POS labels, each `u32` byte length followed by UTF-8 |
| 12 | Character assignments, `(primary, mask): (u32, u32)` |
| 13 | Categories: `u32` name length, UTF-8 name, `u8` invoke/group/length/reserved flags |
| 14 | Reserved, empty |

A final 32-byte SHA-256 covers all preceding bytes, including padding and
the source digest. Serialization writes integer fields explicitly, never
Rust padding or pointers. The reader checks canonical section order,
widths, arithmetic, UTF-8 boundaries, matrix and entry limits, assignment
masks, tag indices, unknown ranges, and trie parent cycles, terminal group
coverage and UTF-8 paths. Unsupported versions and malformed images fail
before tokenization. Load-time structural validation uses linear scratch
storage for trie parent chains; it is released when loading finishes.

## Output and provenance checks

`lattice_golden.rs` and `lattice_readings.rs` execute each case through both
the source and prebuilt forms. This includes the checked-in Japanese and
Korean fixtures, the partial dictionary's errors, and the full IPADIC,
mecab-ko-dic and UniDic dictionaries when supplied. The full dictionaries
were checked locally for this change. The source digest and tokenizer
configuration hash remain identical across forms. Features, POS labels,
byte offsets, readings, unknown flags and tie order are preserved.

`lattice_prebuilt.rs` checks deterministic byte-for-byte reserialization,
feature slices inside the retained image after caller clones are dropped,
unaligned owners, truncation, corrupt checksums, wrong whitespace mode,
invalid table indices and UTF-8, and re-signed trie cycles. Its mutation
probes alter real dictionary records and exercise accepted images against
Japanese text. None of these tests substitutes a fake tokenizer.

## Reproducing the reference comparison

The standalone source is
[`analysis/tokenize/comparisons/lattice.rs`](../analysis/tokenize/comparisons/lattice.rs).
It uses the same UTF-8 IPADIC source for all engines and the same Japanese
inputs as `lattice_throughput`. The external libraries belong to a separate
scratch package, not the workspace dependency graph. From the repository
root, make `target/lattice-comparison/src` and copy the comparison source
there as `main.rs`. Put this manifest in that package:

```toml
[package]
name = "lattice-comparison"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
lindera = "=6.2.0"
lindera-dictionary = "=6.2.0"
vibrato = "=0.5.2"
memmap2 = "=0.9.12"
turbo-tokenize = { path = "../../analysis/tokenize" }
turbo-maxent = { path = "../../analysis/maxent" }
```

```sh
cargo build --release --manifest-path target/lattice-comparison/Cargo.toml
# Retain the resulting Cargo.lock and use --locked for subsequent builds.
taskset -c 4 target/lattice-comparison/target/release/lattice-comparison \
  IPADIC_UTF8 target/lattice-run analysis/goldens/data/inputs/ja
```

The first argument is the source directory, the second a scratch directory
for the compiled files. Do not modify the output while the program runs.
The source dictionary is fetched and converted by `turbo-goldens-tool
fetch-models`; see [the dictionary notes](segmentation.md#compiled-dictionaries-lindera-and-vibrato).

The program checks every Turbo morpheme and feature column against the
source-loaded output before timing tokenization. It prints all nine samples
and their median after a warm pass, interleaving the engines and rotating
which runs first. Load timings include validation but exclude destruction;
the OS page cache stays warm. Build timings include source loading and
writing, with buffered I/O for vibrato. Tokenization timings exclude loading
and the output comparison. Turbo reuses scratch/output buffers, lindera
uses `segment_with_lattice` (reusing the lattice, allocating returned tokens),
and vibrato reuses its worker on Unicode whitespace-separated stretches.
Lindera's different whitespace treatment is visible in the token counts.

## Measured comparison

AMD Ryzen 9 9950X3D, Linux x86_64, Rust 1.99.0, release profile, one logical
CPU (4). This was a shared host with other jobs running: outliers remain
in the raw samples below. These numbers establish the effect of this file
format on this run, not a throughput guarantee or a cold-start comparison.

IPADIC has 392,126 lexicon entries and 325,871 distinct surfaces. The input
is 10,485,900 bytes, composed from the Japanese golden inputs in filename
order and repeated to at least 10 MiB.

- Source manifest SHA-256:
  `6103f8e96a3c1cc7275da1efea6257656f8e50d2c427ab08646d920a4aece735`
- Text SHA-256:
  `97b817559b05be586c663b559bc837996a1473eae7b731c1dbe3f4805d5c6d24`
- Prebuilt file: 66,589,960 bytes, SHA-256:
  `28e6c6b7c61d6772fa943acbf3578ed7779b87dcabfb76ba5300031ed0abdf1f`

| Engine / storage | Median load (ms) | Median tokenize (ms) | Morphemes |
|---|---:|---:|---:|
| source | 426.400 | 305.753 | 2,220,400 |
| prebuilt-read | 236.027 | 285.270 | 2,220,400 |
| prebuilt-mmap | 195.632 | 275.240 | 2,220,400 |
| lindera | 0.402 | 336.841 | 2,281,400 |
| vibrato | 30.129 | 292.795 | 2,220,400 |

Both prebuilt paths reduced load time while preserving all Turbo output
and features on the timed input. Full-image checksum and structural checks
still dominate much of the prebuilt load. Lindera's mapped loader does not
perform this format's full-image checksum/graph validation, so its load
number is not an equivalent validation cost. Token counts also differ as
noted above. No claim of matching its sub-millisecond load time is made.

Raw samples in milliseconds, in execution order within each engine:

```text
load-source: 423.421109, 421.695100, 457.711053, 424.475039, 480.622500, 428.323327, 444.771998, 421.350479, 426.400005
load-prebuilt-read: 236.297610, 230.977775, 232.726055, 235.877697, 264.137755, 236.027199, 264.094573, 239.308916, 235.397892
load-prebuilt-mmap: 194.859708, 194.162632, 195.097507, 239.485601, 197.449248, 194.056722, 195.631796, 195.934377, 200.026495
load-lindera: 0.386640, 0.387031, 0.402019, 0.657161, 0.402841, 0.403332, 0.404934, 0.402119, 0.394716
load-vibrato: 29.995033, 29.771882, 31.315838, 29.803101, 30.234405, 30.221401, 30.312252, 29.525687, 30.129186
tokenize-source: 324.355952, 317.838585, 305.752761, 329.953202, 292.499243, 281.034202, 280.373835, 327.596702, 286.057488
tokenize-prebuilt-read: 278.257719, 426.453726, 291.180442, 289.628122, 282.894004, 287.983325, 281.029844, 284.286213, 285.270141
tokenize-prebuilt-mmap: 272.038074, 271.109180, 283.494769, 286.071976, 273.230986, 272.545181, 315.793853, 275.240370, 299.525151
tokenize-lindera: 328.055248, 336.840679, 769.606905, 348.722045, 371.103521, 328.644270, 328.357759, 329.968681, 414.213531
tokenize-vibrato: 292.795413, 290.399157, 433.483401, 301.736948, 292.417338, 291.302542, 291.064594, 300.006290, 295.651566
```

Single build measurements (source read plus compiled output write):

- `turbo_build_ms=858.954663`
- `vibrato_build_ms=562.324187`
- `lindera_build_ms=759.035642`
