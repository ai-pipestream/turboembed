//! OpenNLP's bundled stopword lists and the lookup of
//! `opennlp.tools.stopword`, as a flag per token.
//!
//! - [`StopwordList`] is the filter `StopwordLists.forLanguage` returns for
//!   one of the eleven languages ([`StopwordLanguage`]): its words, and
//!   `isStopword` as [`StopwordList::contains`].
//! - [`StopwordAnnotator`] reads `opennlp:tokens` and, optionally,
//!   `opennlp:language`, and writes `turbo:stopwords` ([`Stopwords`]), a
//!   flag per token. It never removes a token.
//!
//! The lists in `data/` are OpenNLP's files byte for byte (`data/README.md`
//! says where each comes from and under what license). The build script
//! reads them as `DictionaryStopwordFilter` does and compiles each into a
//! sorted slice, so a lookup lowercases the token into a stack buffer and
//! binary searches about two hundred words, with no allocation and no
//! hashing.
//!
//! It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`
//! of the OpenNLP fork, run on JDK 21; `README.md` describes the JVM
//! goldens the tests compare with.
//!
//! # What OpenNLP does that may surprise
//!
//! - **Matching is case-insensitive, and nothing more.** The bundled
//!   filters are built case-insensitive. A token is lowercased one code
//!   point at a time with `Character.toLowerCase(int)` (the simple mapping,
//!   with no SpecialCasing and no final sigma) and must then equal a word.
//!   Accents are not folded (`tambien` is not `también`), a decomposed
//!   accent is a different word from the composed one, and `DASS` is not
//!   `daß`. Some uppercase letters lowercase to an ASCII letter: U+0130
//!   becomes `i`, so `İT` is the stopword `it`, and U+1E9E makes `DAẞ` the
//!   stopword `daß`.
//! - **`compareToIgnoreCase` equates more than the lookup finds.** The
//!   lists sit in a `HashSet` of `StringList`s, whose hash is that of the
//!   lowercased text and whose equality is `String.compareToIgnoreCase`.
//!   That method also equates characters whose lowercase differs, such as
//!   U+017F LONG S with `s`, U+0131 DOTLESS I with `i` and U+1C80 (a
//!   rounded Cyrillic ve) with `в`, but such a token lands in another hash
//!   bucket, so `ſo` is not the stopword `so`. It would match only if its
//!   lowercased hash happened to equal the word's; a test here checks every
//!   such spelling of every bundled word and finds none, so lowercasing and
//!   comparing gives Java's answer for every token.
//! - **Language codes are lower case only.** `forLanguage` checks the code
//!   against a set of lowercase codes before lowercasing it, so `EN` and
//!   `Eng` are refused. It takes the ISO 639-1 code, the ISO 639-2
//!   terminologic code and the bibliographic `dut`, `fre` and `ger`.
//! - **Duplicate lines collapse.** The Finnish file lists `minä`, `sinä` and
//!   `sillä` twice; the list has 229 words, not the file's 232 lines.
//! - **Removal is a scan for multi-word entries that the bundled lists do
//!   not have.** `StopwordFilter.filter`, `StopwordFilteringTokenizer` and
//!   `StopwordFilterStream` drop the longest run of tokens that forms an
//!   entry at each position. Every bundled entry is one word, so they drop
//!   exactly the flagged tokens ([`Stopwords::kept`] lists the others). The
//!   tokenizer tries every window up to the rest of the sentence at each
//!   token, which costs time quadratic in the token count but finds
//!   nothing a one-word lookup does not. Neither the scan nor lists read
//!   from the caller's own files is ported; the build fails if a bundled
//!   list gains a multi-word line.
//! - **Unicode version.** The lowercase mapping comes from
//!   `turbo-unicode`'s Unicode 18 tables, and JDK 21 has Unicode 15.0. A
//!   test compares the two for every code point whose lowercase, in either,
//!   is a character of a bundled word, and they agree, so the difference
//!   never changes an answer.

mod annotate;
mod list;

pub use annotate::{ListChoice, StopwordAnnotator};
pub use list::StopwordList;
pub use turbo_stopword_layers::{StopwordLanguage, Stopwords};
