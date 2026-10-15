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

`tests/golden.rs` compares the crate with goldens a JVM running OpenNLP wrote:
the `stopword` runs of `analysis/goldens-tool` (`StopwordGolden.java`), made in
its container by `turbo-goldens generate` and checked by `turbo-goldens
verify`. Nothing calls the JVM at test time. A run's golden is
`analysis/goldens/data/stopword/<run>/stopword/<input>.golden`; the inputs are
`analysis/goldens/data/inputs/stopword/words.txt` (words the lookup is asked
about besides the lists' own, one per line, escaped: `\uXXXX` for one UTF-16
unit, `\\`, `\t`, `\n` and `\r`) and `codes.txt` (malformed and other
language codes, one per line).

| Run | Contents |
|---|---|
| `lists` | For each language in `StopwordLists.supportedLanguages()` order: its code, the size of `stopwords()`, `isCaseSensitive()` and the SHA-256 of its resource in the OpenNLP jar, which the test checks against the file in `data/`; then the words sorted |
| `codes` | What `StopwordLists.forLanguage` returns (the list's code, or an empty string where it throws) for every code `LanguageCodeValidator` accepts, rebuilt the way it builds its set from the JDK's locales, for upper and title case forms of the supported codes, and for the codes of `codes.txt` |
| `words` | 21,414 words: the words of `words.txt`, then every word of every list, its upper case (`String.toUpperCase` and per code point), title case, decomposed and accent-stripped forms, spellings with other characters that lowercase to one of its letters (such as U+0130, U+212A, U+212B and U+1E9E) or that only `compareToIgnoreCase` equates with one (such as U+017F, U+0131 and U+1C80), and forms with a character outside the BMP, an apostrophe, a space or a zero-width joiner added; and for each language the words `isStopword` accepts, by index (checked against `isStopword(String...)` with one token, which must agree) |
| `filter` | 150 random token sequences per language, from that language's words and all words without white space, each token by its index in the `words` golden, with whether `StopwordFilter.filter` keeps it and whether a span of `StopwordFilteringTokenizer.tokenizePos` over `WhitespaceTokenizer` starts at it (`StopwordFilterStream` is run too and must agree with `filter`) |
| `letters` | Every code point whose `Character.toLowerCase` is a character of a bundled word, with that lowercase |
