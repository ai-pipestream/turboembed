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
does not ship LGPL data. The lists for
`bn de el en es fi fr gu hi it kn ml nl pa pt ru sk ta te` share a large
part of their entries with Moses (Italian holds 88% of the Moses Italian
list) and are left out. Of the seven kept, Moses has a list only for
Polish. Counted as the splitter reads them (lowercased, without a
trailing full stop, entries with whitespace dropped), `pl.txt` has 129
entries, of which 16 (12%) are in the Moses Polish list, mostly short
forms any list has (`np`, `tzn`, `ul`). The kept lists come from MIT
data: `da.txt` matches the Danish list of Pragmatic Segmenter
(`github.com/diasks2/pragmatic_segmenter`, MIT,
`lib/pragmatic_segmenter/languages/danish.rb`) on 472 of its 473
entries, and `pl.txt` matches its Polish list (`polish.rb`) on 128 of
129. For `de en es fr it pt ru` the CLDR suppressions above take the
place of the dropped lists; the other languages in that list get no
abbreviation list, which shows in the conformance results.

The language table, the fallback chains and the terminator tailorings in
`src/rules/languages.rs` and `src/rules/profile.rs` were written for this
project; they follow sentencex's choices where noted there but copy no
data from it or from the MediaWiki language fallbacks.
