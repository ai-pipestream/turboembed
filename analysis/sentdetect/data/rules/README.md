# Rule splitter data

The inputs of the rule sentence splitter (`src/rules/`). A test,
`tests/rules_tables.rs`, renders `src/rules/tables.rs` from these files and
fails when the committed table differs; run it with
`TURBO_WRITE_RULE_TABLES=1` to rewrite the table after changing a file
here. Nothing in this directory is read at run time.

## `cldr/`

Sentence-break suppressions from the Unicode CLDR, release 48.2
(`common/segments/<language>.xml` in `github.com/unicode-org/cldr`, tag
`release-48-2`), copied unchanged for `de en es fr it pt ru`, the
languages CLDR has suppressions for. Used under the Unicode License v3
(`cldr/LICENSE`).

Each `<suppression>` is an abbreviation that keeps the sentence going,
written with its final full stop and matched case-sensitively, as ICU
matches them. Three Portuguese entries have no final full stop and are
dropped; the generator test counts them.

## `sentencex/`

Abbreviation lists (`abbrev/<language>.txt`) and the English sentence
starters (`starters/en.txt`) from sentencex
(`github.com/wikimedia/sentencex`, commit
268ff94335809977fb5adde334c3b42c2434136f, `src/languages/`), copied
unchanged. MIT License, Copyright (c) 2025 Santhosh Thottingal
(`sentencex/LICENSE`).

Entries are matched lowercased, without their full stop. Lines starting
with `//` are comments; entries that contain whitespace are dropped, and a
trailing full stop is removed.

Only `am ar bg da kk pl uk` are copied. Many sentencex abbreviation lists
overlap the Moses `nonbreaking_prefix` files (`mosesdecoder`,
`scripts/share/nonbreaking_prefixes`), which are LGPL, and this project
does not ship LGPL data. Each sentencex list was compared with the Moses
list for the same language: the seven kept share between 0% and 8% of
their entries with Moses, while the lists for
`bn de el en es fi fr gu hi it kn ml nl pa pt ru sk ta te` share a large
part (Italian holds 88% of the Moses Italian list) and are left out. For
`de en es fr it pt ru` the CLDR suppressions above take their place; the
other languages in that list get no abbreviation list, which shows in the
conformance results.

The language table, the fallback chains and the terminator tailorings in
`src/rules/languages.rs` and `src/rules/profile.rs` were written for this
project; they follow sentencex's choices where noted there but copy no
data from it or from the MediaWiki language fallbacks.
