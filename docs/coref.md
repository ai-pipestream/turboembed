# Coreference (`turbo-coref`)

`analysis/coref/` implements multi-pass coreference resolution: mention
detection, the multi-pass sieve resolver and the CoNLL/MUC/B-cubed/CEAF
scorer. It needs no model. The trained antecedent-ranking head is a bundle
head, described under "The LitBank head in the zoo" below.

## Layers

`CorefAnnotator` reads `View`, `Sentences`, `Tokens`, `PosTags` and
`Entities`, and when present `PhraseChunks`, `Phrases` and `Speakers`.
It writes three layers under the default keys:

| Layer | Content |
|---|---|
| `Mentions` | One item per mention in text order: its span (fragments kept for a discontinuous entity), sentence, token range, head token, the entity it was lifted from, its kind (proper, nominal, pronominal, demonstrative), its origin (entity, noun phrase, pronoun, gold) and its packed agreement (number, gender, animacy, person) |
| `Groups<Mentions>` | The chains, singletons included, in first-mention order |
| `Edges<Mentions, Mentions>` | One edge per link that joined two chains, from anaphor to antecedent, tagged in the `coref-sieve` tag set with the sieve that decided it; the score is 1 |

`Speakers` is an input: speaker turns labelled with
the speaker, used by the speaker sieve. Every chain of `n` mentions has
`n - 1` edges, so the edge count is the mention count minus the chain
count. `turbo_coref::codecs()` adds the four codecs to the shared layer
codecs for the document graph.

`CorefInput` can be built outside a `Document`; before anything indexes
it, every layer's own check, the part-of-speech alignment, every token
and sentence index and every span and fragment against the text are
validated, and a failure is an error. `config_hash` covers the entity
types and a version tag for the fixed sieve set.

## The LitBank head in the zoo

`turbo-train-head train-coref` trains an antecedent-ranking head
(`HEAD_COREF_RANKER`) on LitBank's 100 literary works over a frozen
encoder, and `assemble` packs it with that encoder into one bundle. The
zoo lists such a bundle as `turbo-en-litbank-modernbert-base-coref-2`
(`KIND_HEAD`, `COMPONENT_COREFERENCE_RESOLVER`, `language` `en`,
`CLASS_PERMISSIVE`): `Index::find("en", Component::CoreferenceResolver)`
returns it, and `docs/licences.md` records the data. On the ten held-out
test works it reaches CoNLL F1 0.7348 against the singleton baseline's
0.3009. It has the first bundle's shape (0.7094), trained at seed 2: the
#1828 sweep found no span or ranking setting that beats the shape across
seeds (`trainer/recipes/records/1828/README.md`). The receipt beside the
release carries every setting and digest.

## Scoring

`CorefScorer` computes MUC, B-cubed, CEAF-m, CEAF-e, mention detection
scores and the CoNLL average.

## Design notes

- An entity whose span covers no token is dropped instead of becoming a
  mention without tokens.
- Chains include singletons as groups of one, listing every mention with
  its chain id.

## Tests

`analysis/coref/tests/` covers resolution (`resolution.rs`), the
annotator (`annotator.rs`), the scorer (`scorer.rs`), an analyzer run
(`pipeline.rs`) and document graph round trips (`graph_export.rs`).

## Quotations

`QuotationAnnotator` (`turbo:quotation_annotator`) writes `Quotations`
(`turbo:quotations`, in `turbo-analysis-layers`): each quoted or reported
statement with its content span, speaker span, reporting verb span, the
row of the `Groups<Mentions>` chain the speaker belongs to, its kind
(`Direct`, `Indirect` or `Mixed`) and a confidence. It needs `Tokens`;
`Dependencies`, `Mentions` and `Groups<Mentions>` are used when present.
Quote scanning is shared with the speaker sieve.

Version 2 rejects supplied mentions from a different text view instead of
silently ignoring them. Cue distances larger than the document search to
its ends without overflowing. `Document::add` validates speaker and cue
spans against the quotation view, including UTF-8 boundaries. The speaker
chain remains a documented unchecked raw row number.

| Sieve | Finds | Confidence |
| --- | --- | --- |
| dependencies | a verb of speech outside quotes with an arc into the quote; its `nsubj` is the speaker | 0.9 |
| `ccomp` | indirect speech, or mixed when the complement quotes part of itself | 0.8, 0.7 |
| surface | a verb of speech up to 3 tokens after the closing mark or before the opening one | 0.75 |
| dialog | an unattributed quote after quotes of two chains takes the chain before last | 0.5 |
| none | a quote with no attribution | 0.3 |

A cue after a quote that ends a sentence and leads into the next quote
(`"Hi." Bob said, "Hello."`) is left to the next quote, and a cue before a
quote with a sentence end between them is not used.

`analysis/coref/tests/quotation_eval.rs` is a small hand-built check, not
a corpus score: without a parse, 11 of 11 quotes on eight news and dialog
lines get their gold speaker, and turn alternation gives 4 of 4 dialog
turns the right chain against 2 of 4 without it.
