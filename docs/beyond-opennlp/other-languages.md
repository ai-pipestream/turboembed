# Reference implementations beyond OpenNLP

This survey reconstructs the reference map from repository issue evidence,
principally [#392](https://github.com/ai-pipestream/turboembed/issues/392)
and [#380-#389](https://github.com/ai-pipestream/turboembed/issues/380).
It is a new synthesis, not a recovered copy of the earlier external survey.
The [reference inventory](../reference-inventory.md) supplies exact source
revisions and inspected licence links. The [Rust survey](rust-crates.md)
covers crates named in the issues, including Rust projects also listed here.

## Selection principles

OpenNLP remains the behavioral authority for a parity port. A reference
with a better API or broader coverage does not authorize changing Java
probabilities, UTF-16 behavior, token boundaries or feature templates.
For capabilities beyond OpenNLP, compare outputs using source byte spans,
explicit model identity and separate training-data/checkpoint permissions.
Source-code licences and asset licences are independent.

References serve three purposes: understand an algorithm, supply an
independent test oracle, or contribute a specifically audited source file.
None is a default runtime dependency. Preserve the licence and notices of
any copied file and record its exact revision. Model downloads, package
extras and dictionary fetch scripts require their own asset review.

## Capability map

| Capability | References and code licence | Useful comparison | TurboEmbed issue evidence |
|---|---|---|---|
| Layer APIs, matching and linguistic features | spaCy and spacy-lookups-data (MIT); Stanza and NLTK (Apache-2.0) | Token attributes, dependency/cluster representations, phrase matching; lookup assets need separate provenance | [#380](https://github.com/ai-pipestream/turboembed/issues/380), [#381](https://github.com/ai-pipestream/turboembed/issues/381), [#382](https://github.com/ai-pipestream/turboembed/issues/382), [#385](https://github.com/ai-pipestream/turboembed/issues/385) |
| Analysis graphs and search tokens | Apache Lucene (Apache-2.0) | Position length, synonym graphs, alternatives and normalization chains | [#383](https://github.com/ai-pipestream/turboembed/issues/383), [#184](https://github.com/ai-pipestream/turboembed/issues/184) |
| Unicode boundaries and locale data | ICU, ICU4X and CLDR JSON (Unicode-3.0) | Break rules, complex-script dictionaries, transforms and locale symbols; distinguish Java's Unicode version from the reference version | [#386](https://github.com/ai-pipestream/turboembed/issues/386), [#152](https://github.com/ai-pipestream/turboembed/issues/152), [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| Model tokenization | Hugging Face tokenizers and SentencePiece (Apache-2.0) | Normalization, token-piece alignments and offset contracts | [#166](https://github.com/ai-pipestream/turboembed/issues/166), [#392](https://github.com/ai-pipestream/turboembed/issues/392) |
| Sentence and word splitting | BlingFire, syntok, pySBD, Pragmatic Segmenter, sentencex and wtpsplit (MIT) | Abbreviation cases, punctuation, whitespace and learned sentence splitting; code terms do not approve checkpoints | [#392](https://github.com/ai-pipestream/turboembed/issues/392) |
| Language identification | CLD3 and GlotLID (Apache-2.0 source); fastText (MIT source) | Script filtering, n-grams, short-text behavior and language labels | [#389](https://github.com/ai-pipestream/turboembed/issues/389), [#164](https://github.com/ai-pipestream/turboembed/issues/164) |
| Numbers, dates, units and entities | Duckling (BSD-3-Clause), Recognizers-Text and Presidio (MIT) | Context-sensitive grammars, ambiguity and span extraction; external NLP models remain separate | [#217](https://github.com/ai-pipestream/turboembed/issues/217), [#219](https://github.com/ai-pipestream/turboembed/issues/219), [#220](https://github.com/ai-pipestream/turboembed/issues/220), [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| Repair and social-text rules | ftfy and twitter-text (Apache-2.0) | Mojibake hypotheses, Unicode hashtag/mention boundaries and conformance cases | [#384](https://github.com/ai-pipestream/turboembed/issues/384), [#182](https://github.com/ai-pipestream/turboembed/issues/182) |
| Japanese analysis | Sudachi (Apache-2.0 code) | Lattices, alternate tokenizations and readings; dictionaries have independent terms | [#191](https://github.com/ai-pipestream/turboembed/issues/191), [#354](https://github.com/ai-pipestream/turboembed/issues/354) |
| Component contracts and JVM parity | DKPro Core's allowed Apache-2.0 modules; Apache OpenNLP (Apache-2.0) | Typed analysis contracts, readers, features and JVM output | [#392](https://github.com/ai-pipestream/turboembed/issues/392), [#152](https://github.com/ai-pipestream/turboembed/issues/152), [#153](https://github.com/ai-pipestream/turboembed/issues/153) |

## Tier-1 source set

The explicit #392 set is spaCy, spacy-lookups-data, Stanza, NLTK, Lucene,
ICU, ICU4X, CLDR JSON, Hugging Face tokenizers, SentencePiece, BlingFire,
syntok, pySBD, Pragmatic Segmenter, sentencex, wtpsplit, CLD3, GlotLID,
fastText, Duckling, Recognizers-Text, Presidio, ftfy, twitter-text, Sudachi,
DKPro Core and Apache OpenNLP. Tokenizers has two versioned entries,
v0.22.2 and v0.23.3, giving 28 inventory entries. The pins, not an upstream
main branch, define the offset and behavioral comparison.

ICU's reference scope is break rules/dictionaries, transliteration and
miscellaneous data; CLDR JSON's scope is core and segmentation data.
DKPro excludes `dkpro-core-corenlp-gpl`, `dkpro-core-gpl`,
`dkpro-core-io-tgrep-gpl` and `dkpro-core-sfst-gpl`. Remaining modules still
need their individual notices. GlotLID is code-only: neither its label
inventory nor a permissive repository licence establishes rights to
weights trained on mixed-source corpora.

## Excluded implementations and assets

The #392 do-not-read list is CoreNLP, HeidelTime, YAKE, pke, gensim, GATE,
LanguageTool, Moses, KenLM, polyglot, KoNLPy, Elasticsearch, Maverick,
ReLiK and REBEL. This is the project's source-access rule, not a claim that
all these projects have the same licence. Wrappers and permissive binding
crates do not authorize reading an excluded implementation.

Do not bring CC BY-SA, LGPL or non-commercial models/data into this
reference set. A corpus reader may support a format without shipping a
corpus. Stanza and NLTK code licences do not cover all downloadable resources;
Sudachi dictionaries, spaCy pipelines, fastText models and wtpsplit weights
also require asset-level checks. See [licences](../licences.md) for the
product's dataset and checkpoint policy.
