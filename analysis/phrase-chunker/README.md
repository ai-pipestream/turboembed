# turbo-phrase-chunker

OpenNLP 3.x's `ChunkerME`, ported to Rust on the analysis API: `ChunkerModel`
opens the model and `PhraseChunker` runs the beam search over each sentence's
tokens and part-of-speech tags, writing `opennlp:chunk-tags` (a chunk tag such
as `B-NP`, `I-NP` or `O` and its probability per token) and `opennlp:chunks`
(the phrases those tags make, each a run of tokens with its kind, as
`chunkAsSpans` returns them). The crate documentation (`src/lib.rs`) describes
the modules and where OpenNLP's behaviour is surprising. The name keeps it
apart from `turbo-chunker`, which cuts retrieval units.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork.

## Model

The chunker models Apache OpenNLP publishes are OpenNLP 1.5 models, which this
project does not use. The tests run `chunker-fixture.bin`, a small perceptron
model trained for this project on eight sentences with Penn Treebank tags
(`analysis/sequence/golden/inputs/train-chunk.txt`, `analysis/sequence/README.md`),
copied to `analysis/goldens/data/fixtures` for the golden generator. Its
features name Penn Treebank tags (`t0=NN`), so it runs on the Penn Treebank
tags the English part-of-speech model is asked for (`POSTagFormat.PENN`,
`PosModel::with_format`); on Universal Dependencies tags it chunks nothing
useful. A `ChunkerME` model trained by OpenNLP 3.x on permissive data would be
loaded the same way.

## Golden files

`tests/golden.rs` compares the chunker with files a JVM running OpenNLP wrote,
made by the golden generator (`analysis/goldens-tool`, `ChunkGolden.java`).
Each golden holds the sentences, tokens and Penn Treebank tags the JVM's own
sentence detector, tokenizer and tagger made, the chunk tag `chunk` gave each
token with the probability `probs` reports, and the phrases `chunkAsSpans`
makes of them. The test loads the JVM's tokens and tags as golden layers and
runs the chunker on them, so what is compared is the chunking alone: tags and
phrases exactly, probabilities bit for bit.

| Run | Model | Input sets | What it covers |
|---|---|---|---|
| `chunk/chunker-fixture` | the fixture, beam 10 | `en`, `multi`, `edge` | 1,304 tokens, 768 phrases, 10,936 contexts asked of the model |
| `chunk/chunker-fixture+beam1` | the fixture rebuilt with a beam of 1 (`config beamSize 1`) | `en` | 496 tokens, 292 phrases, one context per token |

Beside each golden is a decision point dump, every context the beam search
asked the model about with the probability of every chunk tag; the test runs
the search with a recording context generator and checks the contexts feature
by feature, in the JVM's order (41 per context), and the probabilities bit for
bit.

`en/emoji-split` has a token that is half of a surrogate pair, which cannot be
loaded onto the Rust text; the test counts and skips that golden. Its
`opennlp:chunks` layer comes with `turbo:utf16<opennlp:chunks>`, as a phrase
there starts or ends inside the character.
