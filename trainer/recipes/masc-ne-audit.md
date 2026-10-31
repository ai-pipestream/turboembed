# Pinned MASC NE annotation audit

Part of #1730. This is an executed Rust reader audit, not a training,
accuracy, peer-performance or release receipt. No gold was projected,
flattened or discarded to obtain these counts.

Executed on 2026-10-31 with Rust 1.99.0. Audit source SHA-256:
`e7d398be4f076362a3e78f71a033de41a7e40413abb8923afce711eb1506e40a`
(`examples/masc_audit.rs`).

## Input and reproduction

MASC 3.0.0 archive: 36,358,521 bytes, SHA-256
`0b300d96016233c7c79b43559b477a6bb0ed16bb416d5c517eccb108a6228493`.
The existing zoo fetcher verified the archive. The extracted directory
matched the recorded 4,425-file corpus digest
`445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`.
No corpus text is committed here.

```sh
cargo +1.99 run -p turbo-trainer-recipes --example masc_audit -- "$MASC_DIR"
cargo +1.99 run -p turbo-trainer-recipes --bin turbo-train-head -- \
  check --corpus masc-ne --dir "$MASC_DIR"
```

The audit uses the existing held-out document split and Twitter exclusion.
Its training side includes the later training-only development partition.
It reads each original entity and token through `MascDocument`; pair counts
compare every distinct pair sharing a token. Sentence regions do not remove
entities from this document view.

## Observed annotations

| Measurement | Training side | Held-out test |
|---|---:|---:|
| NE-annotated documents | 351 | 39 |
| Original entity memberships | 17,046 | 3,578 |
| Exact duplicate, same-type pairs | 42 | 0 |
| Nested, same-type pairs | 95 | 0 |
| Nested, different-type pairs | 208 | 10 |
| Identical-span, different-type pairs | 24 | 2 |
| Crossing pairs | 0 | 0 |
| Discontinuous memberships | 0 | 0 |
| Entity memberships referencing a zero-width source token | 0 | 2 |
| Entities split across kept sentences | 7 | 1 |
| Entities affected by dropped sentence regions | 786 | 125 |
| Entities outside sentence regions | 1,676 | 976 |

These are annotation counts, not independent statistical observations or
F1 scores. The new document view retains the eight cross-sentence entities
and the 3,563 annotations affected by dropped or absent sentence regions.
It also retains zero-width source tokens instead of making their references
appear missing.

Examples identified only by public corpus-relative headers and source IDs:

- `data/written/letters/solicitation-brochures/appalachian1.hdr`:
  `ne-n7` and `ne-n8`, exact same-type duplicate.
- `data/written/travel-guides/HistoryLasVegas.hdr`: `ne-n8` contains
  `ne-n9`, both location.
- `data/written/travel-guides/HistoryJerusalem.hdr`: organization `ne-n158`
  contains person `ne-n159`; organization `ne-n203` and location `ne-n202`
  have identical token membership.

The two held-out zero-width cases are in
`data/written/movie-script/pirates.hdr`:

- `ne-n473` references `penn-n6762`, linked to `seg-r11732` with original
  UTF-16 anchors `28597 28597`.
- `ne-n611` references `penn-n8784`, linked to `seg-r15272` with original
  UTF-16 anchors `37233 37233`.

Both referenced Penn nodes and quark regions exist in the pinned XML. These
are not missing-reference parser defects. The document view preserves them,
and the flat training adapter returns a typed refusal with document,
entity ID and original UTF-16 bounds.

## Remaining contract decision

A single flat BIO sequence cannot retain nested same-type entities or two
types on the same span. Removing the 42 duplicate pairs alone cannot resolve
this: 95 same-type nested pairs and 26 different-type identical-span pairs
remain across the two sides. No outermost-only, type-priority, deduplication
or document-exclusion policy is authorized by this receipt.

The actual `check` command now passes the former sentence-cut obstruction
and refuses `HistoryJerusalem.hdr`, `ne-n203`, at UTF-16 `16065..16080`
with `overlapping original annotations`.

Full-gold CRF training and its gold/majority/spaCy/Stanza gate therefore
remain blocked on an annotation/model-representation decision in #1730.
The reader and document grouping are usable independently of that decision.
No GPU run, reduced-data result or shipment through #1726 is claimed.
