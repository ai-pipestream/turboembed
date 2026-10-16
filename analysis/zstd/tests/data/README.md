# Test fixtures

## Made by the `zstd` tool

These were made with the `zstd` command line tool, version 1.5.5, from
originals the tests rebuild with seeded generators
(`tests/common/mod.rs`), so only the compressed files are checked in.
The `originals` example writes the originals out:

```text
cargo run -p turbo-zstd --example originals -- /tmp/orig
cd /tmp/orig
D=<this folder>
zstd -q empty -o $D/empty.zst
zstd -q byte -o $D/byte.zst
zstd -q -1 --no-check text -o $D/text-1.zst
zstd -q -3 --check text -o $D/text-3.zst
zstd -q -9 text -o $D/text-9.zst
zstd -q -19 text -o $D/text-19.zst
zstd -q --ultra -22 -c < text > $D/text-22-pipe.zst
zstd -q -19 --zstd=wlog=10 small -o $D/small-19-wlog10.zst
zstd -q -3 -c < small > $D/small-pipe.zst
zstd -q --zstd=wlog=10 random -o $D/random-wlog10.zst
zstd -q -19 runs -o $D/runs-19.zst
zstd -q --long=27 long -o $D/long.zst
zstd -q --long=28 -c < byte > $D/window-2-28.zst
```

| Fixture | What it holds |
|---|---|
| `empty.zst` | No content: one empty raw block |
| `byte.zst` | One byte, with its checksum |
| `text-1.zst` | 150 KB of text at level 1, no checksum |
| `text-3.zst`, `text-9.zst`, `text-19.zst` | The same text at levels 3, 9 and 19, with checksums |
| `text-22-pipe.zst` | The same text at `--ultra -22`, read from a pipe, so with no content size and a window of 2^27 |
| `small-19-wlog10.zst` | 40 KB of text with a 1 KiB window, so in 40 blocks, many of whose sequence tables are in repeat mode |
| `small-pipe.zst` | The same text from a pipe: no content size |
| `random-wlog10.zst` | 20,000 random bytes with a 1 KiB window: raw blocks |
| `runs-19.zst` | Long runs of one byte (RLE blocks) and text whose only new bytes are spaces (RLE literals) |
| `long.zst` | A text, 3 MB of zeros and the text again, which `--long` matches from 3 MB back |
| `window-2-28.zst` | One byte in a frame whose window is 2^28, which the decoder refuses |

## From `decodecorpus`

`decodecorpus.zst` is seven frames written by `decodecorpus`, a tool in
the zstd project's `tests` folder that writes random valid frames, and
`decodecorpus.bin` is what they decode to. Together the seven hold what
the files above do not: Huffman literals in one stream, Huffman weights
written directly, sequence tables in RLE mode, sequences sections with
no sequences, the 8-byte content size and windows that are not powers
of 2. The tool was built from the sources of facebook/zstd at tag
v1.5.5, in the root of that checkout, and run as:

```text
gcc -O2 -DZSTD_DISABLE_ASM -DZSTD_MULTITHREAD=0 -Ilib -Ilib/common -Ilib/compress \
  -Ilib/dictBuilder -Iprograms -o decodecorpus lib/common/*.c lib/compress/fse_compress.c \
  lib/compress/hist.c lib/compress/huf_compress.c lib/compress/zstd_compress_literals.c \
  lib/compress/zstd_compress_sequences.c lib/compress/zstd_compress_superblock.c \
  lib/compress/zstd_double_fast.c lib/compress/zstd_fast.c lib/compress/zstd_lazy.c \
  lib/compress/zstd_ldm.c lib/compress/zstd_opt.c lib/compress/zstdmt_compress.c \
  lib/decompress/*.c lib/dictBuilder/*.c programs/util.c programs/timefn.c \
  tests/decodecorpus.c -lm
mkdir -p z o
./decodecorpus -n400 -s42 --max-content-size-log=13 -pz -oo
L="z000013 z000042 z000123 z000218 z000237 z000247 z000270"
for f in $L; do cat z/$f.zst; done > $D/decodecorpus.zst
for f in $L; do cat o/$f; done > $D/decodecorpus.bin
```

The tests build by hand what neither source reaches: skippable frames
in `tests/decode.rs`, and in `tests/rules.rs` the 3-byte sequence
count, a count of 0 in two bytes, raw literals with a 3-byte header,
and a frame breaking each rule the decoder checks.
