# Modality: negation scope, factuality and deontic force (`turbo-relation`)

`ModalityAnnotator` writes a `Modality` layer (`turbo:modality`): one
row per scope, each a byte span with a `ModalityTag` of

- `polarity`: `Affirmed` or `Negated`;
- `epistemic`: `Asserted`, `Hedged`, `Reported`, `Conditional` or
  `Presupposed`;
- `deontic`: `None`, `Obligation`, `Permission` or `Prohibition`;
- `cue`: the bytes of the deciding cue, and a `confidence`.

`Modality::tag_for_span(span)` returns the tag of the innermost scope
covering a span (a predicate, an event trigger, a relation mention) in
O(log N) plus the overlapping scopes it walks back over, or `None` when
the span is unmarked. The layer crosses the document graph boundary
with every other layer codec.

The polarity type is `turbo_analysis_layers::modality::Polarity`; at
the crate root it is re-exported as `ModalityPolarity`, because the
root `Polarity` is sentiment's.

## How it works

Reads `View`, `Tokens` and, when present, `Dependencies`. No model and
no regex: the cue lexicon is one `turbo-matcher` phrase automaton over
lowercase tokens (negators such as `not`, `n't`, `never`, `without`,
`failed to`; negative hedges `impossible`, `unlikely`; hedges `might`,
`could`, `probably`, `estimated`; reported speech `allegedly`,
`according to`; conditionals `if`, `unless`, `provided that`;
factives `confirmed that`, `realized that`; obligations `must`,
`shall`, `should`, `required to`; permissions `permitted to`,
`allowed to`; prohibitions `prohibited from`, `banned from`; and
`may`).

With dependencies, a cue scopes to its governing predicate: the
nearest clausal node (`root`, `conj`, `advcl`, `ccomp`, `xcomp`,
`parataxis`, `acl`, `csubj`) above it, climbing over function words
and nominals. The scope is that predicate's subtree minus its
coordinated (`conj`), adverbial (`advcl`) and paratactic clauses, unless
the cue reaches into one (`prohibited from reselling`). A conditional
or factive hangs from its last word (`if`, `that`), so it scopes the
antecedent or the complement and not the main clause. A negating
preposition (`without approval`) scopes its object only.

Without dependencies, a cue scopes forward to the next punctuation
mark or clause conjunction, at most `max_window_tokens` (8) past the
cue.

Cues that share a scope compose into one tag:

- an odd number of negators is `Negated`, and negative hedges count, so
  `not impossible` is `Affirmed` and `Hedged`;
- the epistemic status is the strongest of `Conditional`, `Reported`,
  `Hedged` and `Presupposed`;
- a negator after an obligation is a prohibition (`must not`, `shall
  not`), one before it removes it (`not required to`), a negated
  permission is a prohibition and a negated prohibition a permission;
- `may` is a permission when the predicate's subject is one of
  `deontic_subjects` (`users`, `vendors`, `tenants`, `licensee`, ...),
  otherwise a hedge.

Confidence is 0.9 for a dependency scope, 0.7 for a window, and 0.6
when the scope turns on `may` or `should`.

## Eval

`analysis/relation/tests/modality_eval.rs`: 16 hand-written sentences
with hand-built UD parses and 24 target predicates, each with a gold
(polarity, epistemic, deontic). A target's system answer is
`tag_for_span` over its token, or the unmarked default. The window row
is the same sentences with the dependencies removed.

| System | Polarity | Epistemic | Deontic | Exact |
| --- | --- | --- | --- | --- |
| all asserted | 17/24 | 15/24 | 18/24 | 5/24 |
| window fallback | 23/24 | 21/24 | 23/24 | 20/24 |
| dependency scope | 24/24 | 23/24 | 23/24 | 23/24 |

Remaining misses:

- both systems: `Vendors must encrypt data and may delete logs`, where
  `delete` has no subject of its own, so `may` is read as a hedge and
  not a permission (the conjunct does not inherit `Vendors`);
- window only: `Acme allegedly said it might not sign` merges all three
  cues into one window, so `said` is wrongly negated and `sign` is
  `Reported` instead of `Hedged`; `According to Reuters, Acme will buy
  Beta` stops the window at the comma and misses `buy`.

Caveats: the set is small and was written alongside the annotator, and
one rule was changed after an eval miss (a factive's own verb is not
in its scope, `confirmed that`). The parses are gold, so a real parser
will lower the dependency row. Treat these numbers as a regression
check and a comparison between the scoping modes, not as a benchmark.
