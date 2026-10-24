# Predicate-argument events (`EventExtractor`)

`turbo_relation::EventExtractor` (`turbo:event-extractor`) writes the
`Events` layer (`turbo:events`, in `turbo-analysis-layers`): one frame per
predicate, with its trigger span and lemma, its sentence, a confidence and
its N-ary arguments. Where `Relations` keeps a binary
`(subject, predicate, object)` triple, an event keeps every argument of
*Revenue rose 14% to $120M in Q3 2025*: the `Agent` *Revenue*, the
`Measure` *14%* (anchored to `Quantities`), the `Measure` *$120M*
(anchored to `Money`) and the `Time` *Q3 2025* (anchored to `Temporals`).

It reads `Sentences`, `Tokens` and token `Dependencies`, and uses
`PosTags`, `Lemmas`, `Temporals`, `Money`, `Quantities` and `Entities`
when they are present. It has no model of its own.

## Predicates and roles

A predicate is a verb (a `VERB` or `VB*` tag when there are tags, else a
word with a subject, object, `obl` or `aux` dependent), the head of a
copula (the copula is the trigger, the head its `Attribute` and the
subject its `Patient`), or a noun whose lemma is in
`EventConfig::nominal_triggers` (`acquisition`, `merger`, ...).

| Dependent | Role |
| --- | --- |
| `nsubj`, `csubj` | `Agent`; `Patient` in the passive |
| `nsubj:pass`, `obj` | `Patient` |
| `iobj`, `to`/`for` obl | `Recipient` |
| `by` obl | `Agent` in the passive, else `Measure` |
| `obl:agent` | `Agent` |
| `obl:tmod`, `during`/`after`/`before`/`since`/`until` obl | `Time` |
| `obl:npmod` | `Measure` |
| `in`/`at`/`from`/`on`/`near` obl | `Location` |
| `with`/`using` obl | `Instrument` |
| `because of`/`due to` obl, `because`/`since`/`as` advcl | `Cause` |
| `to`/`so` advcl, `in order to` xcomp | `Purpose` |
| `nmod:poss`, `by` nmod / `of` nmod of a nominal trigger | `Agent` / `Patient` |

The passive is `nsubj:pass`, `aux:pass` or `obl:agent`, or a form of *be*
as `aux` over a `VBN` when the relations carry no subtypes. A verb
conjunct with no subject of its own shares its first conjunct's
(`EventConfig::propagate_conj`).

## Anchoring

With `EventConfig::anchor` (the default), an argument whose head word is
in a mention of `Temporals`, `Money`, `Quantities` or `Entities` records
that layer and row. A temporal argument other than a subject becomes
`Time`, money or a quantity becomes `Measure`, and a location entity
(a kind containing `loc`, `gpe`, `place`, `city` or `country`) in an obl
becomes `Location`. A temporal, money, quantity or location mention
deeper in the subtree, whose nearest governing predicate is this one and
which no argument already anchors, is added as an argument of its own:
in *Acme raised $45M in Series B funding in March 2026*, *March 2026*
hangs off *funding* and still becomes the `Time` of *raised*.

## Evaluation

`analysis/relation/tests/events_eval.rs` is a small hand-built check, not
a corpus score: ten news and finance sentences with hand-written UD
parses and 38 gold `(trigger, role, argument)` triples, scored by exact
match. The parses are gold, so it measures the role mapping and the
anchoring, not parsing; real parser errors will lower every row.

| System | Correct | Predicted | P | R | F1 |
| --- | --- | --- | --- | --- | --- |
| subject and object only (a binary triple) | 20 / 38 | 22 | 0.909 | 0.526 | 0.667 |
| all roles, no anchoring | 28 / 38 | 36 | 0.778 | 0.737 | 0.757 |
| full extractor | 37 / 38 | 40 | 0.925 | 0.974 | 0.949 |

On the 14 gold `Time`, `Measure` and `Location` arguments alone,
anchoring lifts F1 from 0.435 (5 of 14) to 0.933 (14 of 14, 2 extra).
Without anchoring, prepositions mislead it: *on Tuesday* and *in May*
come out as `Location`, *to 12%* as `Recipient`.

The remaining errors: the governed-mention rule attaches an amount inside
an object to the verb (*approved a $2B buyback* gives *approved* the
`Measure` *$2B*; *expect revenue of $3B* likewise), *with little warning*
is read as an `Instrument`, and the controlled subject of *expand* in
*Acme hired 200 engineers to expand its cloud unit* is not recovered.
