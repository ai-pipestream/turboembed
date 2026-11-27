# confusables.txt of Unicode 17.0.0

The input of the table of confusable mappings that changed after Unicode
17.0.0 (`src/runs/confusables_17.rs`). The confusable skeleton of the
goldens was computed with the 17.0.0 data and `turbo-unicode` has a later
release, so the fuzz run needs the mappings on which the two differ to
tell that difference from a fault.

A test, `tests/confusables_17.rs`, renders the table from this file and
the `confusables.txt` in `analysis/unicode/ucd` and fails when the
committed table differs; run it with `TURBO_WRITE_CONFUSABLES_17=1` to
rewrite the table after `analysis/unicode/ucd` moves to a new release.
Nothing in this directory is read at run time.

| File | Source |
|---|---|
| `confusables.txt` | Unicode Security Mechanisms 17.0.0 (UTS #39), `https://www.unicode.org/Public/17.0.0/security/confusables.txt` |
| `LICENSE` | Unicode License v3, `https://www.unicode.org/license.txt` |

Both were fetched on 2026-11-27 and are copied unchanged. The data file is
used under the Unicode License v3 (`LICENSE`). Its SHA-256 is
`091c7f82fc39ef208faf8f94d29c244de99254675e09de163160c810d13ef22a`.
