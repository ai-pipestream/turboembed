# turbo-stopword

OpenNLP 3.x's bundled stopword lists and the lookup of `opennlp.tools.stopword`,
ported to Rust as a flag per token (`turbo:stopwords`) rather than a filter that
removes tokens. The crate documentation (`src/lib.rs`) describes the lookup and
where OpenNLP's behaviour is surprising; `data/README.md` says where each list
comes from and under what license. The layer and the language enum are in
`turbo-stopword-layers`, so a consumer of the flags does not depend on this
crate.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork, run on JDK 21.

## Golden files

`tests/golden.rs` compares the crate with files a JVM running OpenNLP wrote. The
Java program and its outputs are under `golden/`; nothing calls the JVM at test
time. `golden/generate.sh` rebuilds the outputs from OpenNLP built at the commit
above and installed into the local Maven repository; two runs write the same
bytes. Before writing anything it checks that every file in `data/` is the
jar's resource byte for byte.

| Path | Contents |
|---|---|
| `expected/lists.txt` | For each language in `StopwordLists.supportedLanguages()` order: its code, the size of `stopwords()`, `isCaseSensitive()`, then the words sorted |
| `expected/codes.txt` | What `StopwordLists.forLanguage` returns (the list's code, or `-` where it throws) for every code `LanguageCodeValidator` accepts, rebuilt the way it builds its set from the JDK's locales, and for upper and title case forms of the supported codes and a few malformed ones |
| `expected/words.txt` | 21,414 words: every word of every list, its upper case (`String.toUpperCase` and per code point), title case, decomposed and accent-stripped forms, spellings with other characters that lowercase to one of its letters (such as U+0130, U+212A, U+212B and U+1E9E) or that only `compareToIgnoreCase` equates with one (such as U+017F, U+0131 and U+1C80), forms with a character outside the BMP, an apostrophe, a space or a zero-width joiner added, and a few other words |
| `expected/flags.txt` | For each language, the indices of the words in `words.txt` that `isStopword` accepts (checked against `isStopword(String...)` with one token, which must agree) |
| `expected/filter.txt` | 150 random token sequences per language, from that language's words and all words without white space, with the positions `StopwordFilter.filter` keeps and the positions of the spans `StopwordFilteringTokenizer.tokenizePos` keeps over `WhitespaceTokenizer` (`StopwordFilterStream` is run too and must agree with `filter`) |
| `expected/letters.txt` | Every code point whose `Character.toLowerCase` is a character of a bundled word, with that lowercase |

A word is escaped: every UTF-16 code unit outside U+0020..U+007E, and the
backslash, is `\uXXXX`, and tab, line feed and carriage return are `\t`, `\n`
and `\r`. The lines of `filter.txt` are the language, the word indices of the
tokens, the positions `filter` keeps and the positions `tokenizePos` keeps,
separated by tabs.
