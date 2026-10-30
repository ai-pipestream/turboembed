# turbo-lemmatizer

OpenNLP 3.x's lemmatizers, ported to Rust on the analysis API: the maxent
`LemmatizerME` (`LemmatizerModel` opens the model, `MaxentLemmatizer` runs the
beam search over each sentence's tagged tokens and applies the predicted lemma
classes, shortest edit scripts, to the lowercased tokens) and the lookup
`DictionaryLemmatizer`. Both read `opennlp:tokens` and `opennlp:pos` and write
`opennlp:lemmas`, a lemma per token; the maxent one also writes
`turbo:confidence<opennlp:lemmas>`. The crate documentation (`src/lib.rs`)
describes the modules and where OpenNLP's behaviour is surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork.

## Native lemma classes

New training through `LemmatizerFactory::new` keeps lemma case. Its
`turbo.lemmatizer.CaseSensitiveV1` factory selects a versioned format:
`K:` keeps the token as written, `L:` lowercases it with the shared Java
helper, and the remaining shortest edit script operates on UTF-16 units.
The class is learned, including the `K:` or `L:` choice. Thus `K:O` can
preserve an unseen proper name and `L:O` can lowercase a sentence-initial
common word. There is no inference-time POS rule for capitalization.
Edit indices end in `:`, so indices above nine decode without ambiguity.

`LemmatizerFactory::opennlp` retains the original training behavior for
JVM goldens. Published models and manifests without a
factory keep the original decoder. Native models use a distinct factory
name so a JVM without that factory refuses them instead of silently
misreading the new classes. Unknown factory versions are refused here.

See [the case evaluation](../../docs/lemma-case-evaluation.md) for
full-corpus results, model hashes and the Italian gate diagnosis.

## Models

The OpenNLP UD lemma models and their golden files were retired under #1647.
Lemmatization now uses our edit-tree head and crate-local dictionary snapshots
(`analysis/lemmatizer/tests/dictionary_snapshot.rs`).

The dictionary the dictionary lemmatizer's tests run,
`lemmatizer-dictionary.txt`, is made for this project and checked in as one of
the golden generator's fixtures, in `analysis/goldens/data/fixtures` (that
directory's README describes it).

## Golden files

The old OpenNLP lemma goldens were removed under #1647. Dictionary
lemmatization is covered by crate-local snapshot tests.

The `lemma-fixture-rust` model has two classes that delete the two units of
an emoji at the end of a token (`Smile😀` to `smile`, `ok👍` to `ok`), so each
holds half of a surrogate pair; the model's outcomes are `JavaString`s, the
dumps list those classes with every other, and the JVM chose one for the
emoji of `edge/emoji-single` and `en/emoji`, whose lemma is then `_`.

`en/emoji-split` has a token that is half of a surrogate pair. The JVM
lemmatizes that token into the same half, so the maxent golden's lemma column
holds a lone surrogate, which the golden reader refuses (a lemma is a string
here); the test counts and skips that golden. The dictionary golden of the same
input reads (the lemma is `O`), and the test skips it because the tokens cannot
be loaded onto the Rust text.
