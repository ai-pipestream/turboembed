# Dependency pattern matching

`turbo_matcher::DependencyMatcher` reads existing token and dependency
columns. It does not tokenize, run a parser, or copy the text. Supply a
`Tokens`-anchored `Dependencies` layer from a parser or corpus annotations
with `Input::with_dependencies`; word-anchored dependencies require an
explicit token/word conversion first. The input validates column lengths,
relation IDs, head indices, self-edges, cycles and supplied sentence IDs.

`spacy::read_dependency_pattern` reads one spaCy JSON dependency pattern:
an anchor with `RIGHT_ID` and `RIGHT_ATTRS`, followed by nodes declaring
`LEFT_ID`, `REL_OP`, `RIGHT_ID` and `RIGHT_ATTRS`. IDs are unique; every
left node must already exist. Attribute predicates use the token matcher's
reader and compiled evaluator, including `DEP`, `POS`, `TAG`, `LEMMA`,
`MORPH`, sets and fuzzy matching. `REGEX` returns the existing typed
unsupported-operator error. Quantified or negated dependency nodes are
refused; a node must match exactly one token. Unsupported keys are errors.

The twenty relations follow spaCy 3.8.7:

| Operators | Relation |
| --- | --- |
| `>`, `<` | Immediate child or parent |
| `>>`, `<<` | Descendant or ancestor |
| `.`, `.*`, `;`, `;*` | Adjacent/later/adjacent-earlier/earlier in the same tree |
| `$+`, `$-`, `$++`, `$--` | Adjacent right/left or any right/left sibling |
| `>+`, `>-`, `>++`, `>--` | Adjacent right/left or any right/left child |
| `<+`, `<-`, `<++`, `<--` | Adjacent right/left or any right/left parent |

For compatibility with the actual upstream engine, sibling lookup treats
a root as its own head and enumerates its children. This means a root can
match its own child through a sibling operator. Separate roots never match
a relation. Ancestor/descendant relations exclude the node itself.
Different pattern nodes may select the same token if their relationships
permit it, as spaCy does.

Each result has a label, pattern ID and token indices in **pattern node
order**. Read each node's original span from `Input::span`; no contiguous
envelope substitutes for discontinuous nodes. The matcher writes no layer
and cannot detach these indices from their original token view. Existing
alignment and offset-table APIs project these spans across views and units.

Adding an invalid pattern leaves the matcher unchanged. There are at most
64 nodes per pattern and 4096 patterns. `MatchLimits` defaults to ten
million predicate/relation/ancestor-link visits and 100,000 results per
call. A limit failure returns an error, never a partial list. Work is
bounded across the entire call, including candidate construction and deep
ancestor walks. Working columns are allocated only for attributes used by
a pattern; input text and dependency columns remain borrowed.

The source and output checks against the pinned spaCy container, original
UD EWT annotations, published rules, licensing and reproducible commands
are in [the oracle guide](../analysis/matcher/tests/data/README.md).
This matcher completes #949's dependency-pattern slice alongside the
previously landed JSON token/phrase reader. It does not implement spaCy
callbacks or add regular-expression support.
