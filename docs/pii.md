# PII detection

`turbo-pii` (`analysis/pii/`) finds personal and sensitive identifiers in
a text view and writes them as the `turbo:pii` layer: e-mail addresses,
phone numbers, payment cards, IBANs, ABA routing numbers, network
addresses, credentials, Bitcoin and Ethereum addresses, national
identifiers, device identifiers, vehicle identification numbers, parcel
tracking numbers, web addresses, US passport, driver's licence and ZIP
codes, and phone extensions. Each kind with a check digit or
checksum is validated with it:

| Type | Role |
|---|---|
| `PiiDetector` | Runs one or more `Scanner`s and orders mentions into `Pii` (`turbo:pii`) |
| `Pack`, `PiiDetector::pack` | Preconfigured scanner packs (`Payment`, `Contact`, `Network`, `Secrets`, `Crypto`, `UsIdentity`, `EuIdentity`, `BrIdentity`, `CaIdentity`, `Device`, `Logistics`, `Web`, `AllStructured`) |
| `Extractor`, `Scanner`, `CustomExtractor`, `PiiSink` | Built-in and caller-defined byte scanners and kind filters |
| `Pii`, `PiiMention`, `PiiKind`, `PiiKind::Custom` | Typed layer, mention accessor, and built-in/custom kind identifiers |
| `AllowList`, `DenyList`, `Pattern`, `PatternKind`, `Case`, `ListError` | Global and per-kind ignore lists, regex-free shape/glob patterns, and Aho-Corasick/fuzzy deny lists (`analysis/pii/src/lists/`) |
| `PiiEvidence`, `ChecksumAlgorithm`, `EvidenceList` | The typed facts a detector records for each candidate: shape, checksum (algorithm and result), label (span and phrase id), deny hit (list and term), rule |
| `PiiDetectors`, `PiiDetectorRef` | The per-layer table of detector ids and versions every mention refers to |
| `PiiScore`, `PiiScoreKind` | An optional model score with its kind, model and calibration flag; absent is not zero |
| `PiiConfidence` | `Format`, `Labelled`, `Checksum`, or `LabelledChecksum`, derived from the facts by `PiiConfidence::derive` |
| `PiiCandidates`, `PiiCandidate`, `PiiDisposition`, `RejectReason`, `AllowScope`, `AllowRule` | The `turbo:pii-candidates` layer: every span a detector considered and what became of it, with no values |
| `window`, `LabelPolicy`, `window::has_label` | UAX #29 context-word window helper, per-kind label policies, and caller overrides |
| `rewrite::mask`, `mask_all`, `mask_by_kind`, `MaskPolicy` | Byte-length-preserving character masking (with `keeping_format`, `keeping_leading`, `keeping_trailing`, `first_n`, `last_n`) |
| `rewrite::MaskedView`, `mask_view` | Derived `View` with bidirectional UTF-8 offset alignment |
| `rewrite::Pseudonymizer`, `HmacTokenizer`, `PiiRewrite` | Deterministic label replacement and HMAC-SHA-256 tokenization |
| `rewrite::FormatRedactor`, `StaticReplacer`, `Sha256Tokenizer`, `MapReplacer`, `Truncate`, `Abbreviator`, `ZipSafeHarbor`, `AgeSafeHarbor`, `DateRewrite` | Template redaction, static replacement, unkeyed SHA-256 hex digest, lookup-table replacement, partial character retention, initials abbreviation, HIPAA Safe Harbor ZIP generalization, HIPAA Safe Harbor age aggregation, and date truncation, shifting, relative dates and Safe Harbor date years |
| `rewrite::RewritePolicy`, `RewriteStrategy` | Composable per-`PiiKind` rewrite dispatcher with a default fallback strategy |
| `rewrite::PiiAuditReport` | Redacted mention counts, suppressed counts, and keyed token summaries |
| `checksum::luhn`, `mod97`, `mod11`, `icao9303`, `es`, `fr`, `bic`, `vin`, `parcel`, `verhoeff`, `damm`, `sha256`, `keccak`, `base58`, `bech32` | Checksum, national ID, and cryptographic validators |
| `nemotron`, `NEMOTRON_TYPES`, `resolve_nemotron_kind`, `resolve_model_with_rules` | Mapping of all 55 `Nemotron-PII` entity types to `PiiKind` (15 built-ins, 40 custom kinds) and rule-overlap precedence |
| `turbo_tagger::PiiTagger` | Neural `HEAD_TOKEN_TAGGER` (`SCHEME_BIO`) annotator writing `(Pii, PiiCandidates)` with calibrated `PiiScore` (`analysis/tagger/src/pii.rs`) |

The crate depends on `turbo-analysis`, `turbo-analysis-layers`,
`turbo-glossary` and `turbo-unicode` only. SHA-256, HMAC-SHA-256, Keccak-256, base58check,
bech32 and bech32m, mod 97 and Luhn are written in the crate, and the
scanners are hand-written byte scanners with no regular expressions.

## Use

```rust
use turbo_analysis::{Analyzer, Document};
use turbo_pii::{Pack, Pii, PiiDetector};

let analyzer = Analyzer::builder().add(PiiDetector::pack(Pack::AllStructured)).build()?;
let mut docs = vec![Document::new("Write to desk@example.com or call +44 20 7946 0958.")];
for result in analyzer.run(&mut docs) {
    result?;
}
let pii = docs[0].get::<Pii>().unwrap();
for m in pii.mentions() {
    println!("{}..{} {} {} {}", m.span.start, m.span.end, m.kind, m.confidence().name(), m.normalized);
}
```

`PiiDetector::layer(text, view)` and `PiiDetector::detect(text, &mut pii)`
run the detector outside an analyzer. A detector of several scanners
(`PiiDetector::new`) orders the candidates of all of them together, as
`CompositePiiExtractor` does. `Scanner::with_kinds` limits an extractor to
some of its kinds and refuses an empty set or a kind the extractor does
not find (`ScannerError`).

## Packs

| `Pack` | `PiiPacks` method | Scanners | Kinds |
|---|---|---|---|
| `Payment` | `payment()` | the cursor scanner limited to IBANs and cards, then banking | `iban`, `card`, `aba-routing` |
| `Contact` | `contact()` | the cursor scanner limited to e-mail and phone, then phone extension, then US address | `email`, `phone`, `phone-extension`, `street-address` |
| `Network` | `network()` | network | `ipv4`, `ipv6`, `mac` |
| `Secrets` | `secrets()` | secrets | `aws-access-key`, `github-token`, `jwt`, `url-credential` |
| `Crypto` | `crypto()` | crypto | `btc-address`, `eth-address` |
| `UsIdentity` | `usIdentity()` | US identity | `us-ssn`, `us-itin`, `us-ein`, `us-passport`, `us-drivers-licence`, `us-zip` |
| `EuIdentity` | `euIdentity()` | EU identity | `uk-nhs`, `de-steuer-id`, `es-dni`, `es-nie`, `es-cif`, `de-personalausweis`, `fr-nir`, `fr-siren`, `fr-siret`, `bic` |
| `BrIdentity` | `brIdentity()` | BR identity | `br-cpf`, `br-cnpj` |
| `CaIdentity` | `caIdentity()` | CA identity | `ca-sin` |
| `Device` | `device()` | device | `imei` |
| `Logistics` | (none) | logistics | `vin`, `tracking-number` |
| `Web` | (none) | web | `url` |
| `AllStructured` | `allStructured()` | cursor, secrets, crypto, network, US, EU, BR and CA identity, device, banking, logistics, web, phone extension, US address, age | every kind but `date`, `currency`, `state` and `county` |
| `Dates` | (none) | temporal | `date` |
| `Money` | (none) | money | `currency` |
| `Places` | (none) | US state, US county | `state`, `county` |

`Dates`, `Money` and `Places` are opt-in and not in `AllStructured`:
dates, amounts and places are in most texts, and only some of them are
a person's.

`PiiDetector::cursor()` is the default `CursorPiiExtractor`: e-mail,
phone, IBAN and card.

## The layer

`Pii`, in `turbo-analysis-layers`, has one item per mention with the
columns `spans` (UTF-8 byte offsets into the view, on character
boundaries), `kind`, `detector`, `evidence`, `score` and `normalized`,
plus the `custom_kinds` and `detectors` tables those columns index
(`PiiMention::confidence()` derives the tier from the facts). The
`normalized` column is the detected value itself, so it is a
`Sensitive<Strs>`: its `Debug`, like a `PiiMention`'s, prints
`<withheld>`, and reading it is spelled `expose()`. Outputs copy it only
where their disclosure decision allows values (docs/grpc-analysis.md, PII
output). Mentions are ordered
by start, then longer first, then by kind priority; equal mentions (span,
kind and normalized form) appear once, the first found kept. By default,
a mention strictly nested inside a longer mention (such as a phone or
card inside a grouped IBAN, or an NHS number inside an international
phone number) is dropped; `PiiDetector::keeping_nested` retains strictly
nested mentions when requested. Mentions of different kinds may still share an exact
span or cross each other: an IMEI after its label is also a Luhn-valid
card number, and `https://user:pass@example.com` has a URL credential
crossing an e-mail address. `Pii::check` holds a layer to this order.

The kinds, in id order (`PiiKind::TABLE`, `PiiKind::builtin_id`), with the
normalized form and the confidence each gets. Each row of the table has a
stable id (1 to 38), its name and a priority; the ids are the numbers of
the proto `PiiKindId` enum, the C `TURBO_PII_KIND_*` defines, the Java
`TurboPiiKind` enum and the binary codec, and never change. Priority
alone decides ordering and cluster winners; `PiiKind` has no `Ord`.
A C ABI PII row carries its kind as a number in `token_start` under
`TURBO_SPAN_PII_KIND` when the analyzer is created with
`TURBO_ANALYZER_PII_KIND_IDS`: the id for a built-in kind, and
`TURBO_PII_KIND_CUSTOM_BASE` (`1 << 16`, the codec's custom kind code)
plus the custom-table index for a custom kind. The Java bindings always
ask for it and expose it as `piiKindCode()`, `piiKind()` and
`piiCustomKindIndex()` (#1940).
Confidence ids are format 1, checksum 2, labelled checksum 3 and
labelled 4 everywhere; 0 is unspecified and refused.

| Kind | Normalized form | Confidence |
|---|---|---|
| `jwt` | as written | format |
| `aws-access-key` | as written | format |
| `github-token` | as written | format |
| `url-credential` | the `user:password` userinfo | format |
| `email` | the domain in lower case | format |
| `iban` | upper case, no separators | checksum (mod 97) |
| `imei` | the digits | labelled checksum (Luhn) |
| `card` | the digits | checksum (Luhn) |
| `btc-address` | bech32 in lower case, base58 as written | checksum (base58check, bech32, bech32m) |
| `eth-address` | EIP-55 case | checksum when written in mixed case, format when in one case |
| `mac` | lower case pairs split by `:` | format |
| `ipv6` | RFC 5952 | format |
| `ipv4` | dotted decimal | format |
| `us-ssn`, `us-itin` | `ddd-dd-dddd` | format |
| `uk-nhs` | the digits | checksum (mod 11) |
| `de-steuer-id` | the digits | checksum (ISO 7064 MOD 11,10 and the repetition rules) |
| `es-dni` | 8 digits and uppercase control letter | checksum (mod 23) |
| `es-nie` | `X`/`Y`/`Z`, 7 digits, and uppercase control letter | checksum (mod 23) |
| `es-cif` | organization letter, 7 digits, and control digit/letter | checksum (weighted digit/letter) |
| `de-personalausweis` | 10-character alphanumeric Ausweisnummer | checksum (ICAO Doc 9303 7, 3, 1 weights) |
| `fr-nir` | 15 characters without separators (`2A`/`2B` preserved) | checksum (mod 97 key `01..=97`) |
| `fr-siren` | 9 digits without separators | checksum (Luhn) |
| `fr-siret` | 14 digits without separators | checksum (Luhn or La Poste mod 5) |
| `br-cpf` | 11 digits without separators | checksum (double mod 11) |
| `br-cnpj` | 14 characters without separators | checksum (double mod 11, numeric or compact/punctuated alphanumeric) |
| `bic` | 8 or 11 uppercase characters after a `BIC`/`SWIFT` label | labelled (ISO 9362 + ISO 3166-1 alpha-2) |
| `ca-sin` | the digits | labelled checksum (Luhn) |
| `aba-routing` | the digits | checksum (3, 7, 1 weights) |
| `phone` | `+` and digits, or the digits | format |
| `vin` | 17 uppercase characters | checksum (ISO 3779 / 49 CFR 565 check character); labelled checksum near a `VIN` label |
| `tracking-number` | `1Z` and 16 characters, or the digits | checksum (UPS mod 10, FedEx Express mod 11, GS1 mod 10); labelled checksum near a carrier or tracking label |
| `us-ein` | `dd-ddddddd` | format (assigned IRS prefix); labelled near an `EIN` label |
| `url` | scheme and host in lower case, the rest as written | format |
| `us-passport` | 9 characters | labelled |
| `us-drivers-licence` | upper case, no separators | labelled |
| `us-zip` | `ddddd` or `ddddd-dddd` | format after a state code, labelled after a ZIP label |
| `phone-extension` | the digits | format |
| `age` | the number, no leading zeros | format |
| `street-address` | as written | format |

`PiiConfidence` is ordered `Format < Labelled < Checksum < LabelledChecksum`:
`Format` (`"format"`, C ABI `TURBO_PII_CONFIDENCE_FORMAT = 1`, Java
`TurboPiiConfidence.FORMAT(1)`) when only the shape backs the kind, `Labelled`
(`"labelled"`, `TURBO_PII_CONFIDENCE_LABELLED = 4`, `LABELLED(4)`) when a
context label within the kind's word window names a format-only mention,
`Checksum` (`"checksum"`, `TURBO_PII_CONFIDENCE_CHECKSUM = 2`, `CHECKSUM(2)`)
when a check digit or checksum over the value passes without a context label,
and `LabelledChecksum` (`"labelled-checksum"`,
`TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM = 3`, `LABELLED_CHECKSUM(3)`) when a
context label names the kind and its check digit passes. The tier is
never stored or set: `PiiConfidence::derive(facts)` reads it off the
evidence, a passed `Checksum` fact and a `Label` fact being the two
inputs of the table below, and nothing in detector policy reads the tier
(see "Evidence, detectors, scores and candidates").

| passed checksum fact | label fact | tier |
|---|---|---|
| no | no | `Format` |
| yes | no | `Checksum` |
| no | yes | `Labelled` |
| yes | yes | `LabelledChecksum` |

### Context-word window

`turbo_pii::window` (`analysis/pii/src/window.rs`, tested in
`analysis/pii/tests/window.rs`) checks whether any label term or
multi-word label phrase for a `PiiKind` appears within `N` UAX #29 word
tokens (`turbo_unicode::word_break`) before `span.start` or after
`span.end`. Word tokens are compared with ASCII case folding (`sin`,
`SIN`, `Sin`), non-word punctuation and whitespace segments (`:`, `#`,
`=`, `.`, parentheses, line breaks) are skipped, and multi-word phrases
(`"social insurance number"`, `"social insurance no"`,
`"social security number"`) match when their near boundary lies within
`N` words of the candidate span:

- **Per-kind defaults**: `window::default_labels(kind)` returns the built-in
  label list for each of the 38 built-in kinds (`&[]` for
  `PiiKind::Custom`). `window::default_window(kind)` is `1` for `ca-sin`,
  `imei`, and `bic`, `2` for `us-passport` and `us-drivers-licence`, and `window::DEFAULT_WINDOW_WORDS` (`4`) for every
  other kind. `window::default_policy(kind)` is `LabelPolicy::Required` for
  `ca-sin`, `imei`, `bic`, `us-passport` and `us-drivers-licence` (the
  last two with a window of `2` words), `LabelPolicy::Boosts` for `vin`,
  `tracking-number`, `us-ein` and `us-zip` (a label raises the tier and
  never drops a candidate), and `LabelPolicy::None` for the other 29
  built-in kinds and custom kinds, so existing detector outputs stay unchanged
  unless a caller opts in. When no caller override is configured
  on `ca-sin`, `imei`, or `bic`, `PiiDetector` uses
  `window::has_strict_prefix_label` (the strict prefix rule: a label
  before `span.start` separated only by whitespace and `:`, `#`, or `=`,
  plus `.` for `ca-sin` and `bic`) so default `CaIdentity`, `Device`,
  `EuIdentity`, and `AllStructured` outputs remain identical; configuring
  `with_label_window`, `with_extra_label`, or `with_label_policy` on
  `ca-sin`, `imei`, or `bic` switches that kind onto the general
  bidirectional `N`-word UAX #29 window.
- **Policies and caller overrides**: `PiiDetector::label_required(kind)`
  (`LabelPolicy::Required`) rejects unlabelled candidates of `kind`
  (recorded in `turbo:pii-candidates` as `RejectedByRule { LabelRequired }`)
  and records a `Label` fact on labelled ones;
  `PiiDetector::label_boosts(kind)` (`LabelPolicy::Boosts`) keeps
  unlabelled candidates and records a `Label` fact on labelled ones (so
  their tier derives to `Labelled` or `LabelledChecksum`). The label
  search is the detector's alone: a scanner or custom extractor cannot
  record a `Label` fact, so a candidate marked "labelled" by an extractor
  never passes a `Required` policy (`analysis/pii/tests/window.rs`);
  `PiiDetector::with_label_policy(kind, policy)` sets any `LabelPolicy`
  (including relaxing `ca-sin` or `imei` to `Boosts` or `None`);
  `PiiDetector::with_label_window(kind, n)` overrides the word window
  width `N`; and `PiiDetector::with_extra_label(kind, label)` /
  `with_extra_labels` append caller terms or multi-word phrases. Non-default
  overrides fold deterministically into `PiiDetector::config_hash()`.
- **Known limitation**: the context window checks word distance only and
  does not parse clause or field structure. When a label inside the
  `N`-word window belongs to a different nearby number (for example
  `"Passport 123456789, order 987654321"` when `"987654321"` lies within
  `N` words of `"Passport"`), the second number also matches the window.
  Callers can narrow the window for that kind with
  `PiiDetector::with_label_window(kind, 1)`.

## Evidence, detectors, scores and candidates

Every mention records why it was found, by whom, and (when a model
scored it) how strongly, and every candidate a detector considered is
kept with the decision taken on it (#1903). Facts are primary: the
confidence tier, the audit report and every policy decision read them.

- **Evidence** (`PiiEvidence`, packed per mention in `EvidenceList`):
  `Shape` (the text has the kind's shape), `Checksum { algorithm, passed }`
  (`ChecksumAlgorithm`: Luhn 1, mod 97 2, mod 11 3, mod 23 4, Spanish CIF
  5, ISO 7064 mod 11,10 6, ICAO 9303 7, Base58Check 8, Bech32 9, EIP-55
  10, ABA weights 11, other 12, La Poste SIRET mod 5 13, VIN 14,
  UPS mod 10 15, FedEx mod 11 16, GS1 mod 10 17), `Label { span, phrase }` (the label
  phrase's span in the layer's view and its index in the kind's phrase
  table: built-in phrases first, then `with_extra_label` phrases in
  registration order), `DenyHit { list, term }` and `Rule { rule }`. Each
  mention holds at least one fact; a failed checksum is recorded as a
  fact too (`passed: false`). Label spans are projected with the layer
  (`turbo:pii` on a derived view carries them in that view's offsets).
  Fact kinds have stable tags (shape 1, checksum 2, label 3, deny hit 4,
  rule 5) in the binary codec; an unknown tag, algorithm, score kind or
  disposition on import is an error, never skipped.
- **Detectors** (`PiiDetectors`): a per-layer table of
  `(id, version)` pairs, at most 1024 entries of 128 bytes each. Built-in
  scanners are `turbo-pii:<extractor>` (`turbo-pii:cursor`,
  `turbo-pii:iban`, ...) at `BUILTIN_VERSION` (the crate version); deny
  lists are `DENY_LIST_DETECTOR` (`turbo-pii:deny-list`); a
  `CustomExtractor` names itself with `id()` and `version()` (default
  `"0"`). Every mention's `detector` column refers to the table.
- **Scores** (`PiiScore`): optional per mention; `value`, `kind`
  (`Probability` 1 in `[0, 1]`, `Logit` 2, `Heuristic` 3 in `[0, 1]`),
  `model` (a detector ref) and `calibrated`. A mention with no score is
  `None`, which is not a score of zero.
  `PiiDetector::with_score_threshold(min)` rejects scored candidates
  below `min` (`RejectedByRule { Threshold }`) and leaves unscored ones
  alone.
- **Candidates** (`PiiCandidates`, `turbo:pii-candidates`, written next
  to `turbo:pii` by the annotator): one item per candidate any detector
  produced, with the same kind, detector, evidence and score columns and
  a `PiiDisposition`: `Kept { mention }` (its row in `turbo:pii`),
  `AllowedByRule { scope, rule }` (`AllowScope::Global` or `Kind`,
  `AllowRule::Term(i)` or `Pattern(i)` by registration index),
  `NestedIn { mention }`, `RejectedByRule { reason }` (`LabelRequired`
  or `Threshold`) or `Repeats { mention }`. The layer has no value or
  normalized column by construction, so a suppressed candidate never
  carries text. `PiiAuditReport::with_candidates` counts suppressed
  candidates per kind from it, and `PiiCandidates::check_against` holds
  mention references to the `turbo:pii` layer of the same run.
- **On the wire**: `PiiMention` carries `detector` (5), `evidence` (6)
  and `score` (7) next to its kind, tier and normalized form, and
  `TypedAnnotation.pii_candidate` (99) carries a `PiiCandidateAnnotation`;
  the typed validator refuses a mention whose tier is not the one its
  evidence derives to, an unknown fact, algorithm, score kind, scope or
  reason, a missing detector, and a candidate whose mention index lies
  outside the `turbo:pii` layer of the same instance. The C ABI borrows
  `turbo_pii_span`, `turbo_pii_fact` and (under
  `TURBO_ANALYZER_PII_DETAIL`) `turbo_pii_candidate` slices from
  `turbo_analysis_result` alongside `turbo_span`, and the Java bindings
  read them as `piiDetector()`, `piiEvidence()`, `piiScore()` and
  `piiCandidates()`, with no value or label text by default (#1941).

## Per-document risk

The `turbo-pii:risk` step (aliases `turbo:pii-risk`, `pii-risk`;
`turbo_pii::PiiRiskScorer`) reads the view and its `turbo:pii` layer and
writes `turbo:pii-risk` (`PiiRisk`): one score and level per document, so
the indexing pipeline can index a document as is, redact it, or
quarantine it. It is computed from the layer alone and holds kinds,
counts and weights, never a value, so it stays in a findings-only answer.

Each kind has a severity, and each severity a weight:

| Severity | Weight | Built-in kinds |
| --- | --- | --- |
| critical | 10 | jwt, aws-access-key, github-token, url-credential, card, us-ssn, us-itin, uk-nhs, de-steuer-id, es-dni, es-nie, de-personalausweis, fr-nir, br-cpf, ca-sin, us-passport, us-drivers-licence |
| high | 5 | iban |
| medium | 2 | email, phone, street-address, imei, mac, ipv4, ipv6, btc-address, eth-address, vin, every custom kind |
| low | 0.5 | es-cif, fr-siren, fr-siret, br-cnpj, bic, aba-routing, us-ein, tracking-number, url, us-zip, phone-extension, age |

`PiiSeverityTable` overrides any built-in kind (`with`), a custom kind by
name (`with_custom`) and the custom default (`with_custom_default`). The
table's digest is the step's configuration hash and is recorded on the
layer, so a cached risk is never reused under another table.

Each mention has a confidence weight, a documented heuristic over its
evidence facts and never a probability: 1 for a deny list hit or a label
with a passed checksum, 0.9 for a passed checksum, 0.85 for a label, and
0.75 for the shape alone or a rule. A failed checksum counts for nothing.

A kind's weighted count `c` is the sum of its mentions' confidence
weights, and its contribution is `severity weight x log2(1 + c)`: one
fully supported mention scores its weight, three score twice it, seven
three times, so no kind dominates by volume. The score is the sum of the
contributions in kind order. Every contribution is positive and grows
with its count, so adding a mention never lowers the score, and the same
layer and table always give the same bits.

| Level | Score |
| --- | --- |
| NONE | 0 (no mention) |
| LOW | above 0, below 3 |
| MEDIUM | from 3, below 6 |
| HIGH | from 6, below 9 |
| CRITICAL | from 9 |

So one Luhn-checked card is critical (9.26), one unlabelled SSN high
(8.07) and two critical (13.2), one e-mail address low (1.61), ten high
(6.17), and fifty Luhn-checked company numbers stay low (2.76). The layer also
reports the mention count, the view's length in Unicode scalar values,
and the density in mentions per 1,000 characters.

Each kind also carries a suggested rewrite: a `[%t]` label for
credentials, contact, network, URL and custom kinds; the last four kept
for cards and IBANs; a format-keeping mask for identity, licence, vehicle,
parcel, postal and business numbers and phone extensions.
None needs a key, a dictionary or per-document numbering.
`turbo_kserve::pii_policy::suggested_policy` turns a projected
`turbo:pii-risk` layer into a `RedactionPolicy` message (id
`turbo-pii:risk-suggested`, one enabled rule per kind found, rule ids
`suggested:<kind>`, a `*` mask by default) that a caller can review,
change and send back as `pii_redaction`.

On the wire, `TypedAnnotation.pii_risk` (100) carries a
`PiiRiskAnnotation` with `PiiRiskLevel`, and one `PiiRiskKind` per kind
with its `PiiSeverity` and `PiiSuggestedRewrite`; custom kinds index the
layer's `custom_pii_kinds`. The layer is document-scoped: its one
annotation has no span. The typed validator refuses a level its score
does not give, an unknown or unspecified enum, kinds out of order or
outside the table, a mention count that is not their sum, a non-finite
number, a span and a second annotation.

## Neural BIO PII head

`turbo_tagger::PiiTagger` (`analysis/tagger/src/pii.rs`, step id
`turbo-tagger-pii`) runs a bundle's `HEAD_TOKEN_TAGGER` (`SCHEME_BIO`,
layer `turboembed:pii`) over `(View, Tokens, Option<Subwords>)` and writes
`(Pii, PiiCandidates)`. The default-tier zoo head
`turbo-en-nemotron-pii-modernbert-base-1` (`COMPONENT_PII_DETECTOR`) is
trained in pure Rust (`turbo-trainer`) on `nvidia/Nemotron-PII` over
`answerdotai/ModernBERT-base` (`trainer/recipes/src/nemotron_pii.rs`) to
cover contextual PII that format and checksum scanners cannot recognize
(person names, medical record and health-plan numbers, free-form
addresses, credentials, occupations, and identifiers without check
digits).

```rust
use turbo_analysis::{Analyzer, Document};
use turbo_analysis_embed::EncoderOptions;
use turbo_pii::{Pack, Pii, PiiDetector};
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

let tagger = PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())?
    .with_temperature(NEMOTRON_PII_TEMPERATURE)?
    .with_score_threshold(0.50)?
    .with_rules(PiiDetector::pack(Pack::AllStructured));

let analyzer = Analyzer::builder()
    .add(SimpleTokenizer::default())
    .add(tagger)
    .build()?;
let mut docs = vec![Document::new("Patient Jane Doe, SSN 123-45-6789, MRN 884210.")];
for r in analyzer.run(&mut docs) {
    r?;
}
let pii = docs[0].get::<Pii>().unwrap();
```

### Label mapping (`turbo_pii::nemotron`)

All 55 `Nemotron-PII` entity types map deterministically through
`NEMOTRON_TYPES` (`analysis/pii/src/nemotron.rs`):

- **15 built-in `PiiKind` variants**: `age` (`Age`), `bank_routing_number`
  (`AbaRouting`), `county` (`County`), `credit_debit_card` (`Card`), `date`
  (`Date`), `email` (`Email`), `ipv4` (`Ipv4`), `ipv6` (`Ipv6`),
  `mac_address` (`Mac`), `phone_number` (`Phone`), `ssn` (`UsSsn`), `state`
  (`State`), `street_address` (`StreetAddress`), `swift_bic` (`Bic`), and
  `url` (`Url`).
- **40 kebab-case `CustomKinds` entries** (registered automatically in
  canonical order): `account-number`, `api-key`, `biometric-identifier`,
  `blood-type`, `certificate-license-number`, `city`, `company-name`,
  `coordinate`, `country`, `customer-id`, `cvv`, `date-of-birth`,
  `date-time`, `device-identifier`, `education-level`, `employee-id`,
  `employment-status`, `fax-number`, `first-name`, `gender`,
  `health-plan-beneficiary-number`, `http-cookie`, `language`, `last-name`,
  `license-plate`, `medical-record-number`, `national-id`, `occupation`,
  `password`, `pin`, `political-view`, `postcode`, `race-ethnicity`,
  `religious-belief`, `sexuality`, `tax-id`, `time`, `unique-id`,
  `user-name`, and `vehicle-identifier` (40 custom + 15 built-in = 55
  total; structured kinds outside `Nemotron-PII`'s 55 types such as `iban`
  are covered directly by `PiiDetector` packs).

If any custom kind is promoted to a built-in `PiiKind` in a future schema
revision, `resolve_nemotron_kind` automatically resolves its kebab-case
wire name via `PiiKind::parse` first.

### Rule-overlap precedence, `PiiScore`, and policy integration

- **Rules win on overlap**: `PiiTagger::with_rules(PiiDetector)` runs the
  configured scanner pack alongside the BIO head in one `Annotator` step
  (respecting `Analyzer`'s single-producer invariant on `turbo:pii`).
  Whenever a neural span overlaps an admitted rule mention, the rule
  mention wins so checksum and format guarantees (`Checksum`, `Format`,
  `LabelledChecksum`) are never diluted by a neural span:
  - Exact `(span, kind)` twins appear in `turbo:pii-candidates` with
    `PiiDisposition::Repeats` pointing to the kept rule mention.
  - Neural spans strictly inside a kept rule mention record
    `PiiDisposition::NestedIn` pointing to that rule mention.
  - Enclosing, partially overlapping, or cross-kind neural spans (`postcode`
    on a `us-zip` span) yield to the kept rule mention.
- **Scores and temperature calibration**: Rule mentions record
  `score: None`; neural mentions record
  `Some(PiiScore { value, kind: PiiScoreKind::Probability, model, calibrated })`
  with mean token probability in `[0.0, 1.0]` (`Some(0.0)` is distinct from
  `None`), detector id `turbo-tagger-pii` on `mention.detector`, and model
  id/version in `score.model` (`PiiDetectors`). Calling
  `PiiTagger::with_temperature(T)` scales token logits by `1 / T` (fitted on
  held-out `dev` via `turbo-train-head calibrate`, never `test`) and sets
  `calibrated: true`.
- **Policy integration (`#1882`)**: `RedactionPolicy` rules in
  `turbo-kserve` evaluate `ScoreAtLeast` (`PiiScoreKind::Probability`,
  `min`, `calibrated`), `Detector` (`turbo-tagger-pii`), and
  `ChecksumPassed` uniformly across rule and neural mentions in the same
  `turbo:pii` layer.

## Masking and labels

`rewrite::mask` overwrites the characters of each span with a policy;
`mask_by_kind` uses `MaskPolicy::for_kind` (cards
and IBANs keep separators and their last four letters or digits, contact,
network, national and device identifiers and routing numbers keep
separators, credentials and wallet addresses are masked whole), and
`mask_all` masks every character. Each span is masked against the
original text, in order, and what a later policy keeps readable does not
undo an earlier mask, so overlapping mentions are both masked.

The mask character is one ASCII byte (`MaskPolicy::new` refuses any other,
`MaskError::NotAscii`) and a masked character becomes as many mask bytes as
it has UTF-8 bytes, so the masked text has the byte length of the original
and every byte offset stays valid.

`Pseudonymizer` replaces each mention with a numbered label (`EMAIL-1`),
the same label for the same kind and value within one text.
`HmacTokenizer` replaces it with the kind in capitals, `-` and the first
hexadecimal digits of the HMAC-SHA-256 of the kind name, a zero byte and
the normalized value, under a key of at least 32 bytes; the label is the
same in every text for one key. Both return a `PiiRewrite`: the new text
as a `View` derived from the one read, written with `ViewBuilder` so its
alignment maps both ways, the labels as a layer of it, and its
`turbo:pii-edits` layer (see "The redacted view" below); `to_derived` and
`to_original` map a span either way through the alignment; overlapping mentions are coalesced by
default into their union span under the kind of highest priority, while
`refusing_overlaps` and invalid or empty spans are refused
(`RewriteError`). `PiiAuditReport` counts mentions by kind, with keyed
tokens in place of values. `analysis/pii/tests/rewrite_audit.rs` checks,
for every pack over every text of the `pii` input set, that no mention's
covered text or normalized value appears in the report's `Display` or
`Debug` output.

The key comes from the caller and is at least 32 bytes. `HmacTokenizer`
copies it and leaves it out of its `Debug` output. Tokens made with one
key are linkable across texts, so they need the access control of the
values they stand for.

### Additional rewrite strategies and per-kind composition

Every strategy below works standalone via `.rewrite(&view, &pii)` (with
`rewrite::source_view(text)` for a text outside a document) and inside
a composable per-`PiiKind` `RewritePolicy` (`RewritePolicy::new(default).with(kind, strategy)`),
producing a `PiiRewrite` whose derived view maps UTF-8 byte spans from
the source text into the rewritten text and back
(`analysis/pii/tests/rewrite_strategies.rs` and
`analysis/pii/tests/rewrite_audit.rs`):

- **Format-string redaction (`FormatRedactor`, `FormatError`):** expands a
  caller template supporting `%t` (uppercase kind name, such as `EMAIL` or
  `US-SSN`, matching `Pseudonymizer`), `%k` (lowercase kind name, such as
  `email` or `us-ssn`), `%v` (1-based per-kind pseudonym number, reusing
  the same number for equal `(kind, normalized)` within one rewrite), and
  `%%` (literal `%`), for example `{{{REDACTED-%t}}}` or `[%t-%v]`.
- **Static replace (`StaticReplacer`):** replaces mentions with fixed text
  (default `"[REDACTED]"`, or overridden per `PiiKind` via
  `StaticReplacer::new(default).with(kind, text)`), producing identical
  output within a document and across documents.
- **Unkeyed SHA-256 (`Sha256Tokenizer`, `Sha256Replacer`):** replaces each
  mention with the 64 lowercase hexadecimal digits of
  `checksum::sha256::digest` over its normalized UTF-8 value (or a 4-to-64
  hex digit prefix via `Sha256Tokenizer::with_length`, with optional
  `with_kind_prefix` or `with_prefix`). **Security warning:** because
  unkeyed SHA-256 uses no secret key, digests are linkable across all
  documents and datasets and can be reversed by brute-force search or
  precomputed tables for low-entropy or bounded-domain identifiers (such
  as 9-digit US SSNs, phone numbers, or known e-mail lists). Use
  `HmacTokenizer` whenever a secret key can be held.
- **Map replace (`MapReplacer`, `MapTarget`, `MapError`):** replaces
  mentions from a caller-provided lookup table (constructed inline with
  `MapReplacer::new` or loaded from a caller TSV or delimited file with
  `from_tsv`, `from_delimited`, `from_tsv_file` or `from_delimited_file`,
  never bundled), with a `case_sensitive` flag, target selection
  (`MapTarget::NormalizedOrSurface`, `Normalized`, or `Surface`), and a
  configurable fallback `RewriteStrategy` (`with_fallback`) for values not
  present in the table.
- **Truncate / keep N (`Truncate`, `KeepSide`, and `MaskPolicy`):**
  `Truncate::keep_start(n)` (`KeepSide::Start`) and
  `Truncate::keep_end(n)` / `Truncate::last_n(n)` (`KeepSide::End`) keep
  `n` Unicode scalar values (`char`, preserving multi-byte UTF-8
  boundaries) at the start or end of a mention and replace the remaining
  characters with a configurable mask character (`with_mask`, default
  `'*'`), optionally preserving non-alphanumeric separators
  (`keeping_format`). When a mention has `<= n` characters (or `<= n`
  letters or digits with `keeping_format`), all of its characters are
  masked so that a short secret is never left in the clear. For
  byte-length-preserving in-place masking and `MaskedView`, `MaskPolicy`
  also provides `keeping_leading(n)`, `keeping_trailing(n)`,
  `MaskPolicy::first_n(n, mask)`, `MaskPolicy::last_n(n, mask)` and
  `MaskPolicy::rewrite`; note that in-place `MaskPolicy` leaves spans with
  `<= n` letters or digits unmasked (matching `MaskPolicy::last_four`), so
  callers who need short values masked should use `Truncate::keep_start(n)`
  or `Truncate::last_n(n)`.
- **Abbreviate (`Abbreviator`):** reduces a mention to the initials of its
  Unicode words (`"John Fitzgerald Kennedy"` to `"J.F.K."` with
  `Abbreviator::dotted()` or `"JFK"` with `Abbreviator::compact()`, or
  custom punctuation via `Abbreviator::with_punctuation`), with optional
  hyphen preservation (`preserving_hyphens`, turning `"Jean-Luc Picard"`
  into `"J.-L.P."` or `"J-LP"`) and case preservation (`preserving_case`).
  `Abbreviator` is intended for name or multi-word phrase kinds and should
  not be used as a default strategy on numeric kinds (for example,
  `123-45-6789` would abbreviate to `1.4.6.`).
- **ZIP Safe Harbor (`ZipSafeHarbor`):** HIPAA Safe Harbor
  (45 CFR 164.514(b)(2)(i)(B)) for `us-zip`: keeps the first three digits
  of a ZIP code, writes `000` when that three-digit area holds 20,000
  people or fewer, and masks every other digit (`62704-1234` becomes
  `627**-****`, `03601` becomes `000**`). The restricted areas are the
  17 prefixes HHS published from the 2000 Census
  (`HHS_2000_RESTRICTED_PREFIXES`, `ZipSafeHarbor::hhs_2000()`) or a
  caller's own three-digit list (`ZipSafeHarbor::with_restricted`, for a
  later Census). A mention of another kind than `us-zip`, or that is not
  five ASCII digits optionally followed by `-` and four more, is masked
  whole, and so is a cluster of
  overlapping mentions, so the strategy never leaves another kind's value
  readable. Edits record `PiiStrategyKind::ZipSafeHarbor`
  (`PII_STRATEGY_ZIP_SAFE_HARBOR`); the gRPC arm is
  `PiiZipSafeHarborStrategy` (#1884).
- **Age Safe Harbor (`AgeSafeHarbor`):** HIPAA Safe Harbor
  (45 CFR 164.514(b)(2)(i)(C)) for `age`: keeps an age of 89 or under as
  written and writes every older age as `90+`, the single category the
  rule allows (`aged 93` becomes `aged 90+`, `34-year-old` stays). The
  `age` mention is the number alone, so the words around it stay. Leading
  zeros do not make an age younger (`093` becomes `90+`). A mention of
  another kind than `age`, digits or not, or that is not one or more ASCII
  digits, is masked whole with the mask character
  (`*` by default), and so is a cluster of overlapping mentions. Edits
  record `PiiStrategyKind::AgeSafeHarbor` (`PII_STRATEGY_AGE_SAFE_HARBOR`);
  the gRPC arm is `PiiAgeSafeHarborStrategy` (#1884).
- **Dates (`DateRewrite`):** for `date`, the mention's text is read again
  with the temporal scanner, so the value is the one the detector saw,
  then, by `DateMode`:
  - `Year`: the year alone (`July 14, 2026` becomes `2026`), which Safe
    Harbor (45 CFR 164.514(b)(2)(i)(C)) lets a record keep. Edits record
    `DateYear` (`PII_STRATEGY_DATE_YEAR`).
  - `Shift(DateShift)`: moved by years and months as one count of months,
    then days, then an optional keyed offset, and written in the form it
    was written in: month names full or three letters in their case,
    ordinals (`14th` becomes `2nd`), zero-padded days and separators stay.
    A day past the end of the new month becomes its last day (January 31
    plus one month is the end of February, a leap day plus a year is
    February 28), and a month mention moves as its first day does, or one
    month the days' way when they would leave it in its own month
    (`July 2026` under a 10-day shift becomes `August 2026`). A shift
    whose net effect would write a value back as it was (no offset, or a
    fixed shift that cancels the keyed one) masks it whole, so no date
    under a shift stays readable.
    `DateShift::keyed(key, context, max_days)` adds 1 to `max_days` days,
    earlier or later, chosen by HMAC-SHA-256 of `context` under a key of
    at least 32 bytes: never zero, and the same for every date with that
    context, so intervals within a document or patient stay. Edits record
    `DateShift` (`PII_STRATEGY_DATE_SHIFT`).
  - `Relative(reference)`: relative to a reference date the caller passes,
    never the clock (`3 months ago`, `in 2 weeks`, `today`), counted in
    the largest whole unit that fits: days under a week, weeks under 31
    days, calendar months under a year, then years; a month mention in
    months, then years. The temporal scanner reads each phrase back with
    the same reference. Edits record `DateRelative`
    (`PII_STRATEGY_DATE_RELATIVE`).
  - `SafeHarbor(reference)`: the year alone, as `Year` writes it, except
    that every year 90 or more years before the reference year is written
    as one category, `1936 or earlier` for a 2026 reference
    (`safe_harbor_category`). Someone born in such a year can be older
    than 89 on the reference date, so Safe Harbor
    (45 CFR 164.514(b)(2)(i)(C)) counts the year as an element of a date
    indicative of an age over 89, which may be kept only aggregated into
    one category of age 90 or older. The cutoff is by year alone
    (`safe_harbor_cutoff`), so the category also takes a year whose
    people are 89 on part of it, and no year whose people may be 90 is
    left ungrouped.
    The reference is passed in, usually the release date of the record,
    never the clock. The rule does not know which date is a birth date,
    so it groups every old enough date; pair it with `AgeSafeHarbor` for
    the `age` kind. Edits record `DateSafeHarbor`
    (`PII_STRATEGY_DATE_SAFE_HARBOR`).

  A mention of another kind, a text that is not one whole absolute day or
  month mention (the scanner reads four-digit years only, so `7/14/26` or
  `July 14, 26`), a date shifted out of the years `0001` to `9999`, and a
  cluster of overlapping mentions are masked whole (`*` by default). The
  gRPC arm is `PiiDateStrategy`, with `year`, `shift` (a keyed offset
  names a registered key; `max_days` 0 means 30), `relative` and
  `safe_harbor` (each with a required `reference` as `yyyy-MM-dd`)
  (#1884).
- **Per-kind policy (`RewritePolicy`, `RewriteStrategy`):** assigns a
  `RewriteStrategy` per built-in or `PiiKind::Custom` kind with a default
  fallback strategy (`RewriteStrategy::default()`, and so
  `RewritePolicy::default()`, is a whole-value `'*'` mask, never a
  per-document pseudonym or an unkeyed SHA-256, issue #1882; a
  `MapReplacer` built without `with_fallback` still falls back to a
  pseudonym, and the typed redaction policy always sets that fallback), sharing per-document pseudonym counters across all
  mentions in a single `rewrite` call. When mentions overlap in a cluster,
  any strategy in the cluster configured with `refusing_overlaps` (including
  via a `MapReplacer` fallback) causes `RewritePolicy::rewrite` to return
  `RewriteError::Overlap`; when coalesced, a source character inside the
  union span is kept only when every overlapping mention covering that byte
  permits keeping it under its own strategy.

### Reversible protection (`reversible` feature)

With the `reversible` cargo feature of `turbo-pii`, two strategies
replace each mention with a value that a holder of the key can decrypt
back to the mention as it was written (its surface). The feature brings
in the RustCrypto `aes-gcm` crate and the `fpe` crate (both MIT or
Apache-2.0). The default build has no crypto crate.

- **AES-256-GCM tokens (`AesGcmEncryptor`):** the token is the kind in
  capitals, `:`, then the unpadded base64url of the 12-byte nonce, the
  ciphertext and the 16-byte tag, for example `EMAIL:...`. The associated
  data binds the kind as the token writes it, plus a caller context
  (`with_context`, such as a document id or a field name). A token moved
  to another kind or context, decrypted with another key, or changed in
  any character fails authentication (`CryptError::Authentication`).
  Every token draws a fresh nonce, so one value gives a different token
  each time and tokens are not linkable. `decrypt(token)` returns the
  kind and the value.
- **FF1 format-preserving encryption (`Ff1Encryptor`, NIST SP 800-38G):**
  encrypts the characters of an alphabet (`Ff1Alphabet::Digits`,
  `UpperAlphanumeric` or `Alphanumeric`, radix 10, 36 or 62) under a
  caller tweak (`with_tweak`). Every other character keeps its place, so
  `123-45-6789` encrypts to another `ddd-dd-dddd`. A value with a letter
  or digit outside the alphabet is refused (`CryptError::OutsideAlphabet`,
  which names its byte offset, never the character) rather than leaving
  that character in the clear. FF1 needs a domain of
  at least a million values, so a value with fewer than 6 digits, or 4
  letters and digits, is refused (`CryptError::TooFewNumerals`). One value
  under one key and tweak always encrypts the same way, so ciphertexts
  are linkable, as `HmacTokenizer` labels are.

Both take a 32-byte AES-256 key from the caller (`CryptError::KeyLength`
otherwise). The key is kept only in the cipher's key schedule, which is
zeroised on drop, and it never appears in `Debug` output. In a rewrite,
a mention that cannot be encrypted fails the rewrite with
`RewriteError::Encrypt`. The edits record the strategy as
`PiiStrategyKind::Aes256Gcm` (10) or `PiiStrategyKind::Ff1` (11), which
are `PII_STRATEGY_AES_256_GCM` and `PII_STRATEGY_FF1` on the wire.

With the feature, both are `RewriteStrategy` arms (`RewriteStrategy::AesGcm`
and `RewriteStrategy::Ff1`, from `AesGcmEncryptor` or `Ff1Encryptor` with
`.into()`), so a `RewritePolicy` can give each kind its own encryptor and
mask or tokenize the rest. Each mention is encrypted as written. When
mentions overlap, the merged run is encrypted once, as one value. Two
encryptors are equal, and fold the same into the policy's configuration
hash, when their key, context or tweak, alphabet and overlap rule match.
The key enters as an HMAC-SHA-256 under a fixed label, as an
`HmacTokenizer` key does, never as itself. A clone of an
`AesGcmEncryptor` draws a nonce seed of its own and is still equal,
since it decrypts the same tokens. A gRPC redaction policy cannot name
them yet: on the server, keys are to be registered key references,
never key bytes in a message.
`analysis/pii/tests/reversible.rs` runs the NIST FF1 AES-256 samples
through `Ff1Encryptor` and the round trip of both strategies for every
built-in kind and a custom one. `analysis/pii/tests/reversible_policy.rs`
runs them as policy strategies beside a mask. The encryptor's unit test
checks the cipher against NIST CAVP AES-256-GCM vectors.

### The redacted view

The server's `turbo-pii:redact` step writes the same view and edits from
a typed `RedactionPolicy` (docs/grpc-analysis.md, "The redact step"),
choosing each mention's strategy with `rewrite::rewrite_chosen` and a
`StrategyChooser`; its edits also carry the policy rule id and digest.
Policy dictionaries (`PiiDictionaryRef`) resolve through the vocabulary
store, pinned by content, and release audit events name each released
mention's covering rule, revision and digest (#1882 part C). The PII wire
messages are frozen under the proto compatibility policy from #1882 on:
see docs/grpc.md, "PII and result envelope freeze (#1882)".

`RedactedView` is the pipeline step for a `RewritePolicy`: it reads a
`View` and the `turbo:pii` layer over it and writes the rewritten text as
a `View` derived from it, with the `turbo:pii-edits` layer over that view,
both as the instance named `redacted` (`rewrite::REDACTED`). The step
writes the view first and the edits second; the edits are made over
`ViewId::PENDING` and moved onto the id the document gives the view as the
two are stored (`Layer::bind_pending_view`).

`turbo:pii-edits` (`PiiEdits`) has one item a replacement: its span in the
redacted view, its source span in the parent view, the covered mentions
of the pii layer as a list of rows (any subset, so overlapping members are
listed one by one), the rule of the policy that chose the strategy
(`PiiRule::Default` or `PiiRule::Kind`), the strategy kind
(`PiiStrategyKind`), and the caller's policy revision
(`RewritePolicy::with_revision`). It holds no original value. The
document refuses an edits layer whose spans disagree with the view's
alignment in either direction, or whose members are not exactly the
mentions under each source span. Across the rewrite a span maps source to
redacted to source as containment, not equality: a span touching a
replacement comes back as the whole mention it replaced
(`analysis/pii/tests/redacted_view.rs`, which builds view chains with
`ViewBuilder` for combining marks, ZWJ emoji, a split ligature, overlaps,
unequal lengths, and deletion and insertion anchors).
`Document::map_span_outcome` says whether a projected span is exact,
widened or deleted, and `Document::project_fragments` reports which of an
item's fragments coalesced, as a separate fact.

### The masked view

`MaskedView` is an annotator that reads a `View` and the `turbo:pii`
layer over it and writes the masked text as a `View` derived from it,
bound by `MaskedView::bound` as the instance named `masked`
(`rewrite::MASKED`):

```rust
use turbo_analysis::{Analyzer, Document};
use turbo_pii::rewrite::{MASKED, MaskedView};
use turbo_pii::{Pack, PiiDetector};

let analyzer = Analyzer::builder()
    .add(PiiDetector::pack(Pack::Contact))
    .step(MaskedView::by_kind('*')?.bound())
    .build()?;
let mut docs = vec![Document::new("Write to desk@example.com.")];
for result in analyzer.run(&mut docs) {
    result?;
}
assert_eq!(docs[0].get_key(&MASKED).unwrap().text(), "Write to ****@*******.***.");
```

`MaskedView::by_kind` masks each kind with `MaskPolicy::for_kind`,
`MaskedView::all` every mention with one policy, and `with` sets the
policy of one kind. `MaskedView::view` and `mask_view` make the view
outside an analyzer, and `mask_into` writes into a caller's
`ViewBuilder`. The text is the one `mask` writes for the same spans and
policies, overlapping spans included; the view is built with
`ViewBuilder` as it is written, with no copy of the text beyond the view
itself.

The view has the byte length of its parent, and each masked character
is its own edit, so the alignment maps exactly in UTF-8 bytes both ways:
every character boundary of the source maps to the same offset of the
view, a mask byte maps back to the one character it stands for (a byte
inside the four mask bytes of `𝕒` maps to `𝕒`), and adjacent mentions
keep the boundary between them. A character whose bytes already are the
mask (a `*` masked with `*`) is not an edit.
`analysis/pii/tests/masked_view.rs` checks every offset of every pair of
spans over multi-byte texts, and that the text equals `mask`.

## Vehicles, parcels, EINs and web addresses (#1878)

Four kinds added after the kind registry (#1901) take the next ids,
appended: `vin` 31, `tracking-number` 32, `us-ein` 33 and `url` 34. Their
checks record the checksum ids `vin` 14, `ups-mod10` 15, `fedex-mod11` 16
and `gs1-mod10` 17. The scanners are byte scanners with no regular
expressions:

- `vin` (`Logistics`, `analysis/pii/src/scan/logistics.rs`): 17 uppercase
  letters and digits at a word start, no `I`, `O` or `Q`, at least one
  letter and one digit, a tenth (model year) character that is not `U`,
  `Z` or `0`, and the ninth character the check character of
  `checksum::vin` (49 CFR 565.15 transliteration and weights, modulus 11,
  `X` for 10). VINs outside North America need not carry a check
  character, so they are not found.
- `tracking-number` (`Logistics`): compact, or in groups of four split by
  single spaces. UPS: `1Z` and 16 uppercase letters or digits, single
  spaces allowed, with the UPS check digit. FedEx Express 12 digits
  (weights 3, 1, 7, modulus 11, modulus 10) and FedEx Ground 15 digits
  (GS1 modulus 10), both only when the digits fail Luhn, since a
  Luhn-valid run is read as a card or an IMEI. USPS 20 digits starting
  `0` or `7` (GS1 over the digits as written, or with the `91` prefix the
  legacy numbers imply). 22 digits starting `91` to `95` (USPS, GS1 over
  21 digits) or `96` (FedEx Ground, GS1 over the 14-digit serial). A
  12-digit number passes the FedEx check about one time in ten, so an
  unlabelled 12-digit reference can be read as a tracking number;
  `PiiDetector::label_required(PiiKind::TrackingNumber)` admits only
  labelled ones.
- `us-ein` (`UsIdentity`, `analysis/pii/src/scan/us.rs`): `dd-ddddddd`
  at a number start and end whose prefix the IRS assigns (every prefix
  but 00, 07 to 09, 17 to 19, 28, 29, 49, 69, 70, 78, 79, 89, 96 and 97)
  and whose serial is not all zeros. A bare nine-digit run is not read
  as an EIN.
- `url` (`Web`, `analysis/pii/src/scan/url.rs`): `http://`, `https://`,
  `ftp://` or `ftps://` in any case, or `www.`, at a token start; a host
  of two or more labels ending in a registered top-level domain (the
  e-mail scanner's IANA table), or after a scheme a dotted IPv4 address;
  an optional port; RFC 3986 path, query and fragment characters.
  Trailing sentence punctuation, quotes and an unbalanced closing bracket
  are left out. A URL with userinfo is left to `url-credential`, so a
  credential is never hidden inside a longer `url` mention. Bare domains
  (`example.com`) are not URLs. A `url` mention holds any address,
  number or e-mail address inside it, which the default detector drops
  as nested (`keeping_nested` keeps them).

`vin`, `tracking-number` and `us-ein` default to `LabelPolicy::Boosts`:
a label in the word window (`VIN`, `vehicle identification number`;
`tracking`, `UPS`, `FedEx`, `USPS`, `shipment`, `waybill`; `EIN`,
`FEIN`, `employer identification number`, `tax id`) raises the tier and
an unlabelled mention is kept. Masking keeps separators of the three
number kinds and masks a `url` whole. `checksum::verhoeff` and
`checksum::damm` are reusable validators for kinds to come.

Evaluation: `analysis/pii/tests/us_identifiers.rs` has positive,
negative and check digit failure cases and single-digit mutation sweeps
per scanner; `analysis/pii/tests/us_identifiers_eval.rs` scores a seeded
synthetic set of 200 valid values and 200 near misses per kind (800
true, 0 missed; the 3 false mentions are near-miss digit runs that pass
Luhn as cards). In the mixed set the project-authored
`us-identifiers-memos.txt` adds 20 gold spans for these kinds, and the six real web
addresses already in the set are `url` gold: `vin` 4/0/0, `tracking-number`
9/0/0, `us-ein` 3/0/0, `url` 10/0/0 (true, false, missed), plus one
false `phone` on the grouped USPS near miss. The external partition was
annotated before `url` existed: its 496 web addresses are counted and
pinned but not scored, and it has one false `tracking-number` (a 12-digit
test card that fails Luhn and passes the FedEx check).

### Passports, driver's licences, ZIP codes and phone extensions

The label-dependent kinds take ids 35 to 38: `us-passport`,
`us-drivers-licence`, `us-zip` and `phone-extension`.

- `us-passport` (`UsIdentity`, `analysis/pii/src/scan/us_labelled.rs`):
  nine digits, or one capital letter and eight digits, with no separator.
  `LabelPolicy::Required` with a two-word window (`passport`, `passport
  number`, `passport no`, `passport card`).
- `us-drivers-licence` (`UsIdentity`): the union of the shapes the states
  issue: seven to thirteen digits, one or two capitals and four to
  fourteen digits, or two digits, three capitals and five digits; a
  hyphen may split two digits, and the normalized form drops it.
  `LabelPolicy::Required` with a two-word window (`driver's license` in
  its spellings, `driving licence`, `license number`, `DL`, `DLN`). The
  two-word window keeps a passport label from naming a licence a few
  words on, and the reverse.
- `us-zip` (`UsIdentity`): five digits, or five, a hyphen and four, not
  all zeros, read only one space after a state, territory or armed forces
  code in capitals (`Springfield, IL 62704`, format) or after a `ZIP`,
  `ZIP code`, `zipcode` or `postal code` label (labelled; the scanner
  reads the label as part of the shape, like `1Z` for UPS, and the
  `Boosts` policy records it). A bare five-digit number is not a ZIP code.
- `phone-extension` (new `PhoneExtension` extractor, in `Contact` and
  `AllStructured`): one to six digits after a phone the phone scan finds,
  introduced by `ext`, `ext.`, `extn`, `extension`, `x` or `#`. The
  mention is the digits alone, next to the phone mention. The extractor
  runs the phone scan itself to find the phones.

Every passport and licence shape is a candidate, so an unlabelled one is
in `turbo:pii-candidates` as `RejectedByRule { LabelRequired }`, as for
`ca-sin`. `PiiDetector::with_label_policy`, `with_label_window` and
`with_extra_label` change the rules. Masking keeps separators. In the
mixed evaluation set the memo document gains a travel and badging
section, and the seven real ZIP codes in the MASC letters are gold:
`us-passport` 2/0/0, `us-drivers-licence` 2/0/0, `us-zip` 10/0/0,
`phone-extension` 2/0/0. The seeded synthetic set has 200 labelled
values and 200 unlabelled ones per kind: all labelled ones found, no
unlabelled one read as one of these kinds; the false mentions are other
kinds on the same digits (Spanish CIFs, ABA numbers, cards).

### Ages and street addresses

`age` and `street-address` take ids 39 and 40. Both are rule scanners;
names, free-form addresses and other kinds with no fixed shape are left
to a trained head.

- `age` (new `Age` extractor, `AgePiiExtractor`, `turbo-pii:age`, in
  `AllStructured`; `analysis/pii/src/scan/age.rs`): a number from 0 to
  125 before `years old`, `year old`, `yrs old` or `yr old`, the pieces
  joined by spaces or hyphens (`34-year-old`), before `y/o` or `y.o.`,
  or after `age`, `age:`, `age of` or `aged`, in any case. The mention is
  the number alone and its normalized form has no leading zeros, so a
  rewrite acts on the number and keeps the words (`a **-year-old`). Not
  ages: a number inside a decimal or a grouped number, `aged` or `age`
  before a number of months, weeks, days or hours or a percentage,
  `years ago`, `years older`, `page 34`, `usage 34`, `ages 5 to 10`.
- `street-address` (new `UsAddress` extractor, `UsAddressPiiExtractor`,
  `turbo-pii:us-address`, in `Contact` and `AllStructured`;
  `analysis/pii/src/scan/address.rs`): a house number of one to six
  digits not starting with 0 (one capital letter may follow it, `221B`),
  an optional direction (`N`, `NW`, `North`), one to four capitalized
  words or ordinals (`42nd`), and a street suffix, every primary name and
  standard abbreviation of USPS Publication 28 Appendix C1
  (`analysis/pii/src/tables/street_suffixes.rs`, in any case once the
  word is capitalized); the last suffix in the run ends the street
  (`18 Mill Creek Road`). A street-only suffix (`St`, `Street`, `Ave`,
  `Avenue`, `Rd`, `Road`, `Blvd`, `Ct`, `Court`, `Loop` and the like) may
  be followed on the same line by a comma-less city (`12 Main Street
  Boston`, `12 Oak Court Springfield`, `1 Infinite Loop Cupertino`, not
  `the 12 Wall Street Journal`), whereas a general noun suffix (`Park`,
  `Point`, `Summit`, `Center`, `Plan`) may not be followed by unconsumed
  capitalized words (`3 Point Lane`, not `the 3 Five Point Plan` or
  `2020 Vision Center Plan`). When the house number is a four-digit year
  (`1900` to `2099`) and the suffix is a general noun (or `Court`, `Ct`
  or `Loop`, as in `2024 Supreme Court`, `2024 Supreme Ct` or `2023
  Feedback Loop`) rather than an unambiguous street-only suffix, a
  post-direction, a unit or a comma and capitalized city word must
  follow (`2016 Central Park West`, `2016 Central Park, New York`,
  `2024 Oak Ct, Springfield`, not `2016 Central Park`, `2024 Supreme
  Court`, `2024 Supreme Ct`, `2023 Feedback Loop` or `2024 Annual Bay
  Summit`). A direction may follow the suffix (`Main St NW`, or a
  quadrant after a `.` or a comma, `L St., NW`), then a unit: an
  optional comma, `Apt`, `Suite`, `Ste`, `Unit`, `Floor`, `Room`,
  `Building`, `Lot` and the like with an optional `.`, or `#`, then one
  to six letters, digits and hyphens. Post office boxes are the same
  kind: `PO`, `P.O.`, `P. O.`, `P O` or `Post Office`, then `Box` and one
  to ten digits, in any case. Every piece is on one line and joined by
  one horizontal space (`' '`, `'\t'`, `U+00A0` or `U+202F`), so a city,
  state and ZIP code after a comma, or the next line of a letter, stay
  outside the mention. Lower case street words,
  a house number inside a decimal, an ordinal alone (`5th Avenue`) and
  a number with no suffix after its words (`Table 4 Summary`, `Route
  66`, `the 3 Point Plan`) are not addresses.

`age` is low severity and `street-address` medium; both are suggested a
label rewrite and are masked keeping separators. Neither needs a label
(`LabelPolicy::None`); `age`, `aged`, `address`, `street address`,
`mailing address` and `home address` are their label phrases for the
context window. In the mixed evaluation set, project-authored clinic
intake memos (`intake-memos.txt`) carry seven ages and six addresses
with near misses, and the two addresses in the MASC letters are gold:
`age` 7/0/0, `street-address` 8/0/1 (the missed one is `P0 Box 1163`,
written with a zero). The external partition was annotated before
these kinds existed: its five postal addresses (RFC author and registry
blocks and one example) and one age are counted and pinned but not
scored.

### Dates and amounts of money

`date` and `currency` take ids 41 and 42. Their scanners read the text
with the numeric crate's own scanners (docs/temporal.md, docs/money.md),
so a date or an amount is found exactly where `turbo:temporal` or
`turbo:money` would find it.

- `date` (new `Temporal` extractor, `TemporalPiiExtractor`,
  `turbo-pii:temporal`, in `Dates`; `analysis/pii/src/scan/date.rs`):
  the absolute mentions of the temporal scanner that name a day or a
  month (`2026-07-14`, `July 4, 1952`, `14th March 2025`,
  `August 2026`). A year alone and a quarter are not date elements HIPAA
  counts, and a relative expression (`3 weeks ago`) needs a reference
  date the scanner never takes, so none is a mention; nor is a
  calendar-invalid one (`2026-13-40`). The normalized form is the value
  in ISO 8601 (`1952-07-04`, `2026-08`).
- `currency` (new `Money` extractor, `MoneyPiiExtractor`,
  `turbo-pii:money`, in `Money`; `analysis/pii/src/scan/money.rs`): the
  mentions of the money scanner with its default symbol table, amounts
  read in US notation (`$1,250.00`, `EUR 40`, `-$3`, `40 euros`). A bare
  number is never money. The normalized form is the ISO 4217 code and
  the amount (`USD 1250.00`, `EUR 40`).

Both are low severity, suggested a label rewrite and masked keeping
separators (`****-**-**`). Neither needs a label; `date`, `date of
birth`, `dob` and `born`, and `amount`, `balance`, `salary` and
`payment`, are their label phrases for the context window. They are the
kinds the date rewrites of #1884 act on.

A redaction policy over gRPC turns `Dates`, `Money` or `Places` on
itself: an enabled rule for `date`, `currency`, `state` or `county`
adds that pack's scanners to the `turbo-pii:pii` step, which otherwise
runs `AllStructured` (`RedactionPolicy::opted_in_packs`).

### States, counties and cities

`state`, `county` and `city` take ids 43, 44 and 45, in the opt-in
`Places` pack.

- `state` (new `UsState` extractor, `UsStatePiiExtractor`,
  `turbo-pii:us-state`; `analysis/pii/src/scan/state.rs`): the name of a
  state, the District of Columbia or an inhabited territory (`Illinois`,
  `New York`, `Puerto Rico`, `U.S. Virgin Islands`), as written or all in
  capitals, at word boundaries, single spaces between its words.
  `Washington, D.C.` and `Washington DC` are the District; with a word
  character joined to that suffix (`Washington DC_x`) the text names
  neither the District nor the state. `New York
  City` is a city and `Washington County` a county, so neither is a
  state; a county word that makes no county of the list leaves the state
  (`the Texas area`, `Illinois County`). A two-letter USPS code is also a common word (`IN`, `OR`, `ME`,
  `OK`), so it is a mention only in capitals and only directly after a
  state label (`State: OR`) or before a ZIP code that ends an address,
  by what comes before the code: a comma (`Springfield, IL 62704`), a
  city of that state in the bundled gazetteer and one space
  (`Springfield IL 62704 is`), or the start of a line when only blanks
  follow the ZIP code to the line's end or a clause's end (`IL 62704`,
  `IL 62704, USA`). After any other
  word (`Order IN 12345.`, `PART OR 90210`), or with a word after a line's
  code and number (`IN 12345 units`), the number is a count, not a
  state. The normalized form is the USPS code (`IL`).
- `county` (new `UsCounty` extractor, `UsCountyPiiExtractor`,
  `turbo-pii:us-county`; `analysis/pii/src/scan/county.rs`): a county or
  county equivalent of the Census Bureau's 2020 county list
  (`national_county2020.txt`, public domain, SHA-256
  `9f6e5f6eb6ac2f5e9a36d5fd01dec77991bddc75118f748a069441a4782970d6`),
  with its own last word: `County`, `Parish`, `Borough`, `Census Area`,
  `Municipality` or `Municipio` (`Cook County`, `Orleans Parish`,
  `Matanuska-Susitna Borough`, `Mayagüez Municipio`). Each word reads as
  the list writes it or in capitals, and the last also in small letters
  (`Cook county`); a right single quotation mark reads as an apostrophe.
  The independent cities (`Baltimore city`) are cities, not counties. A
  name shared by several states (`Washington County`) is one mention,
  normalized as the list spells it; a state named beside it (`Cook
  County, Illinois`) is a mention of its own.
  The 1,918 names are in `analysis/pii/src/tables/us_counties.txt`.
- `city` (new `City` extractor, `CityPiiExtractor`, `turbo-pii:city`;
  `analysis/pii/src/scan/city.rs`): a populated place of the bundled
  gazetteer (`turbo_geo::BundledGazetteer`, Natural Earth's 7,342
  populated places, public domain), matched by the gazetteer's own rule
  (case, accents and punctuation folded). Many city names are also
  common words or given names (`Mobile`, `Phoenix`, `Jackson`), so a name
  is a mention only in one of two settings:
  - directly before a comma, a space and a US state's or territory's
    name or USPS code, when the gazetteer holds a place of that name
    there (`Springfield, IL`, `Mobile, Alabama`, `San Juan, PR`; not
    `Mobile, IL`). The gazetteer files the territories under their own
    country codes, which are their USPS codes;
  - directly after a city label, as any place of that name: `city of
    residence`, `place of birth`, `city of` and `town of` with or without
    a colon, and `birthplace`, `city/town`, `city` and `town` only with
    one (`City: Phoenix`, `the city of Mobile`), as `the city Council`
    names none. A known name that a space and a capitalized word follow
    is the start of a longer name (`City: San Foo`), so it is no mention.

  A bare name (`Chicago is large`) is no mention. The name has at most
  five words with single spaces between them, and its first and last
  words start with a capital (`Rio de Janeiro`). The longest name wins,
  and the normalized form is the gazetteer's name (`Sao Paulo` for
  `São Paulo`). A city and the state after it are two mentions.

A place name starts and ends where no letter, digit, combining mark or
`_` touches it (`text::name_char`, `on_name_start`, `on_name_end`), at
both ends of a state name, a county name, a city name and a USPS code. So
`x_Illinois`, `Illinois_x`, `e` + U+0301 + `Illinois`, `Cook County_x`
and `Washington_County` hold no place, while `Illinois-based` and
`(Cook County)` do. The other scanners keep the shared letter-or-digit
rule (`on_word_start`, `on_end`), so their spans and pack snapshots stay
those of the port.

A county name is compared in Unicode normalization form C, as the
Census list is written. A decomposed spelling (`Doña Ana County` with
`n` + U+0303, `Mayagüez Municipio` with `u` + U+0308) is the same
county: its mention keeps the source's bytes and its value is the
list's spelling. A candidate that is ASCII or already NFC is compared in
place, with no copy. Every state name is ASCII, so the state scanner
needs no folding.

Both are low severity, suggested a label rewrite and masked keeping
spaces (`**** ******`). Neither needs a label; `state`, `state of
residence`, `county` and `county of residence` are their label phrases.
HIPAA Safe Harbor removes every geographic subdivision smaller than a
state, so `county` is an identifier and `state` is not; the state kind
is there for policies that want the whole place gone.

Scored on its own (`places_eval.rs`, gold in
`analysis/pii/tests/data/human-eval/labels-places.tsv`), the `Places`
pack over the ten documents of the human-written set and a
project-authored memo of places and near misses (`places-memos.txt`):

| Kind | Source | TP | FP | FN | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `state` | third-party MASC (`29`) + project-authored (`11`) | 40 | 0 | 0 | 1.000 | 1.000 |
| `county` | project-authored (`places-memos.txt`) | 9 | 0 | 0 | 1.000 | 1.000 |
| `city` | third-party MASC (`24`) + project-authored (`12`) | 9 | 0 | 27 | 1.000 | 0.250 |

The MASC documents name no county, so the county figures are only those
of the memo written for it. The `city` gold marks every reference to a
city, bare names in running prose included (`in Rouen`, `the Boston
area`), so its recall counts what the two settings above leave out: all
27 misses are names with no state after them and no label before them.

## Custom extractors and scanner rules

- Offsets are UTF-8 bytes on character boundaries.
- Masks preserve the UTF-8 byte length of the input (above).
- Caller-defined extractors implement `CustomExtractor` (with optional
  `config_hash` for stateful settings, or pass a scan function to
  `Scanner::custom_fn` / `PiiDetector::with_custom_fn`) and emit
  mentions of a `PiiKind::Custom(CustomKindId)` with their evidence
  facts (`PiiEvidence::Shape`, `PiiEvidence::Checksum`,
  `PiiEvidence::Rule`) and an optional `PiiScore` through `PiiSink`
  (`PiiSink::model` names the detector and version every mention carries;
  `try_add_candidate` takes a score). The sink refuses an empty fact list
  and the detector-owned `Label` and `DenyHit` facts
  (`SinkError::ForeignEvidence`), and a score it cannot hold
  (`analysis/pii/tests/custom_extractors.rs`). A custom kind indexes the
  `CustomKinds` table the detector is given with
  `PiiDetector::with_custom_kinds`; every `Pii` layer carries its own
  table, so there is no process-wide state. Custom kinds order after
  built-in kinds on an exact span tie, then by table index. A custom
  kind name must be 1 to 64 UTF-8 bytes (`CustomKinds::MAX_NAME_BYTES`)
  and not a built-in name; a table holds up to 256 names
  (`CustomKinds::MAX`), checked when the table is built. A kind outside
  the table is refused (`KindError`) by the detector builder,
  `PiiSink`, `Pii::check`, export, import and every rewrite. Caller kinds flow
  through `mask_all`, `mask_by_kind`, `MaskedView` (with
  `MaskedView::with` overrides), `Pseudonymizer`, `HmacTokenizer`,
  `PiiAuditReport` and binary graph export/import.
- By default, `PiiDetector` drops a mention whose span lies strictly
  inside a longer mention, so trailing groups of a grouped IBAN are not
  also reported as a phone or card and ten digits inside an
  international phone number are not also reported as an NHS number.
  `PiiDetector::keeping_nested` retains strictly nested mentions.

## Allow and deny lists

`AllowList` and `DenyList` (`analysis/pii/src/lists/`, tested in
`analysis/pii/tests/allow_deny.rs`) let a caller suppress known non-PII
values or flag domain-specific terms without writing a custom byte scanner:

- **Allow list (`AllowList`)**: applied after detection and nesting
  resolution. A mention whose covered text (`&text[start..end]`) or
  normalized form (`mention.normalized`) matches an active allow rule is
  dropped.
  - Scope is either global (`PiiDetector::with_allow_list`,
    `allow_term`, `allow_terms`, `allow_file`, `allow_pattern`,
    `allow_shape`, `allow_glob`) or scoped to a single `PiiKind`
    (`PiiDetector::with_kind_allow_list`, `allow_kind_term`,
    `allow_kind_terms`, `allow_kind_file`, `allow_kind_pattern`,
    `allow_kind_shape`, `allow_kind_glob`), leaving mentions of other
    kinds on the same or other spans untouched.
  - Terms and patterns compare case-sensitively (`Case::Sensitive`) by
    default, or case-insensitively (`Case::Insensitive`, via
    `AllowList::case_insensitive` / `AllowList::with_case` /
    `Pattern::case_insensitive`) using `turbo-unicode`'s per-scalar
    lowercase mapping. Duplicate terms and patterns are deduplicated on
    insertion and when changing case mode.
- **Regex-free pattern language (`Pattern`, `PatternKind`)**: patterns
  match a whole candidate value from start to end using a deterministic,
  non-backtracking two-pointer wildcard pass with no regular expressions:
  - `Pattern::shape(spec)` (`PatternKind::Shape`):
    - `d`: any ASCII decimal digit (`0..=9`)
    - `a`: any ASCII letter (`A..=Z`, `a..=z`)
    - `w`: any ASCII letter or digit (`A..=Z`, `a..=z`, `0..=9`)
    - `?`: any single Unicode scalar (`char`)
    - `*`: any sequence of zero or more characters
    - `\x`: literal character `x` (such as `\d`, `\a`, `\w`, `\?`, `\*`, `\\`)
    - any other character (including uppercase letters `A..=Z`, digits,
      and punctuation such as `-`, ` `, `@`, `.`): literal match
  - `Pattern::glob(spec)` (`PatternKind::Glob`):
    - `*`: any sequence of zero or more characters
    - `?`: any single Unicode scalar (`char`)
    - `\x`: literal character `x` (such as `\*`, `\?`, `\\`)
    - any other character (including `d`, `a`, `w`): literal match
- **Deny list (`DenyList`)**: terms that are always reported under a
  custom kind of the detector's table (`DenyList::new`,
  `DenyList::from_terms`, `DenyList::from_file`, attached via
  `PiiDetector::with_deny_list`) under the detector `turbo-pii:deny-list`
  with a `PiiEvidence::DenyHit { list, term }` fact naming the list's
  registration index and the term's index in it.
  - Exact terms are matched with `turbo-glossary`'s Aho-Corasick
    `TextMatcher` (`Case::Sensitive` or `Case::Insensitive`) at word
    boundaries (`text::on_word_start` and `text::on_end`), with overlaps
    resolved leftmost-longest by `turbo_glossary::select`. Duplicate terms
    are deduplicated on insertion and when changing case mode.
  - Optional fuzzy matching (`DenyList::with_max_edits`) admits word-token
    candidates within `0`, `1`, or `2` Levenshtein edits (insertions,
    deletions, or substitutions of one Unicode scalar). Edit distances
    above `2` are refused (`ListError::InvalidEditDistance`). To prevent
    short terms from matching arbitrary short words, a term word must have
    at least `2 * k + 2` characters to allow `k >= 1` edits (`>= 4`
    characters for `1` edit, `>= 6` characters for `2` edits; words
    shorter than `4` characters only match at `0` edits).
    Fuzzy terms are indexed by their first word (`analysis/pii/src/lists/fuzzy.rs`): a
    first word of up to 24 characters by the hashes of every string left
    after deleting up to its edit budget of characters, a longer one by
    length, so a text word is only checked against terms that share one of
    its own deletion hashes. Two words within `k` edits always share one,
    so the index never drops a match (#1935).
    `cargo run --release -p turbo-pii --example fuzzy_deny_bench` times the
    fuzzy scan against the exact scan.
  - `DenyList::capitalised_only` (`with_capitalised_only`,
    `capitalized_only`) requires a candidate in the text to start with an
    uppercase or titlecase letter (`Lu` or `Lt`), preventing fuzzy matches
    against common lowercase words.
- **File loading**: `AllowList::from_file`, `AllowList::with_file`,
  `AllowList::with_shapes_file`, `AllowList::with_globs_file`,
  `DenyList::from_file`, and `DenyList::with_file` read caller-supplied
  UTF-8 files one entry per line, stripping an optional UTF-8 BOM and
  ignoring blank lines and lines starting with `#`. Lists are loaded only
  from caller paths and are never bundled.
- **Configuration hash and audit reporting**:
  - `PiiDetector::config_hash` folds all global allow rules, per-kind
    allow rules, and deny lists (target kinds, terms, pattern syntax and
    tokens, case mode, max edit distance, and `capitalised_only`).
  - `PiiAuditReport` (`KindReport::suppressed`,
    `PiiAuditReport::total_suppressed`, `PiiAuditReport::suppressed`,
    `PiiAuditReport::with_suppressed`, `PiiDetector::audit`,
    `PiiDetector::detect_and_audit`) counts suppressed mentions by
    `PiiKind` without ever storing, tokenizing, or displaying the
    suppressed values (`analysis/pii/tests/rewrite_audit.rs`).

Scanner edge-case rules:

- The IPv6 filter refuses an address with fewer than two groups written,
  or with no group of three digits or more unless all eight are written,
  so `::1`, `::` and `a:b:c` are not found. This keeps times and C++
  namespaces out, by design.
- The TLD check of an e-mail domain folds ASCII case only.
- An Ethereum address in one case (all lower or all upper) has no EIP-55
  checksum to check, as the EIP says, and is reported with the format
  confidence. Its normalized form is the EIP-55 case.
- Phone prefixes are tried longest first, and a shorter prefix that fits
  a national length is reported when the whole does not:
  `+44 (0)20 7946 0123` gives `+44 (0)20 7946`, normalized `+440207946`,
  because the trunk `0` counts as a digit.
- Ten digits grouped `ddd ddd dddd` on the same span that pass the NHS
  mod 11 check are both an NHS number and a phone number in the wide
  pack.

## Fixtures

Every fixture is synthetic. Where a test or corpus fixture would otherwise
hold a value that could be real, it uses documented test values or
synthetic specimens:

- The P2PKH (`1...`) Bitcoin addresses of the test corpora
  (`analysis/pii/tests/data/golden-true-positives.tsv`,
  `analysis/pii/tests/data/golden-false-positives.txt`) and of the crypto
  tests are made up: a version byte over a hash of a fixed string, with
  its base58check checksum.
- Public DNS resolver addresses are replaced by documentation ranges
  (`192.0.2.0/24`, `2001:db8::/32`) or made up addresses, and real host
  names by `example.com`.
- An SSN known to have been published in an advertisement is left out.

## Snapshot tests

The snapshots are reviewed outputs of this crate, checked in next to its
tests. No other implementation produces or checks them.

- Inputs: `analysis/pii/tests/inputs/`, 74 texts written for the packs
  with only documented test values, example ranges and made up values.
- Detection snapshots: `analysis/pii/tests/snapshots/<pack>/<id>.tsv`,
  one directory per pack (`payment`, `contact`, `network`, `secrets`,
  `crypto`, `us-identity`, `eu-identity`, `br-identity`, `ca-identity`,
  `device`, `logistics`, `web`, `all-structured`), one row per `turbo:pii` mention with its `start`
  and `end` (UTF-8 byte offsets), kind and TSV-escaped normalized form.
- Rewrite snapshots: `analysis/pii/tests/rewrite_snapshots/<pack>/<id>.txt`,
  what `MaskedView`, `mask_all`, `Pseudonymizer`, `HmacTokenizer` and
  `PiiAuditReport` make of the pack's mentions under a fixed test key:
  the masked text, the text with every mention masked, both rewrites (or
  `OVERLAP` when both refuse overlapping mentions), their labels with the
  spans they replaced, and the audit report.

`analysis/pii/tests/goldens.rs` runs every pack with
`PiiDetector::keeping_nested` over the inputs, checks each layer with
`Layer::check`, that spans are in bounds and that kind names parse back,
and compares the rows with the detection snapshots.
`analysis/pii/tests/rewrite_goldens.rs` does the same for the rewrites,
and also checks that every rewrite label maps back to a source mention
and that both rewrites refuse the same overlaps. A missing snapshot fails
the test. After a reviewed change in behaviour, rewrite them with

```
TURBO_BLESS=1 cargo test -p turbo-pii --test goldens --test rewrite_goldens
```

The fuzz catalog (`analysis/fuzz/src/runs/catalog.rs`) runs every pack
natively over generated texts.

## Precision and recall

`analysis/pii/tests/precision_recall.rs` measures the widest pack
(`Pack::AllStructured`) on two sets: a synthetic set built to exercise
the scanners' rules, and a mixed evaluation set providing partial
human-written-text evidence in
`analysis/pii/tests/data/human-eval/` (`analysis/pii/tests/data/human-eval/README.md`).
A found mention is true when a label has its span and kind; every other
found mention is false. `PII_SHOW_FALSE=1` lists the false mentions.

### Synthetic set

`precision_and_recall_on_the_synthetic_set` runs on a labelled set made
from a fixed seed (40 values of every kind built to pass their checks,
each in a sentence, and near misses of the checksummed kinds with no
label) together with the checked-in true and false positive corpora
(`analysis/pii/tests/data/golden-true-positives.tsv` and
`analysis/pii/tests/data/golden-false-positives.txt`):
1917 texts, 1120 labels made in the test.

| | true | false | missed | precision | recall |
|---|---|---|---|---|---|
| all kinds | 1144 | 40 | 0 | 0.966 | 1.000 |
| `phone` | 122 | 40 | 0 | 0.753 | 1.000 |
| every other kind | 1022 | 0 | 0 | 1.000 | 1.000 |

The false mentions are a phone in each near miss NHS number (ten grouped
digits are a phone shape); phones and cards nested inside grouped IBANs
and NHS numbers nested inside US phone numbers are dropped by default.
The set is made to pass the checks the scanners make, so recall on it
says the scanners find what they are built to find, not how they do on
text written by people.

### Mixed evaluation set (third-party MASC, adapted specifications, and project-authored memos)

`precision_and_recall_on_human_written_text` scores the same pack on a
mixed nine-document evaluation set (`48,036` UTF-8 bytes, `131` gold
mentions across 28 of the 38 built-in kinds in
`analysis/pii/tests/data/human-eval/labels.tsv`; licences and SHA-256
digests in `analysis/pii/tests/data/human-eval/README.md` and
`docs/licences.md`). On its own, this mixed set is partial work for
#968; the external partition below completes its licensed
human-written-text requirement per kind. In the mixed set, only
`phone` is evaluated on unedited third-party prose
(`masc-*.escaped`), `btc-address`
and `eth-address` are evaluated on project-adapted specification summaries
embedding the canonical BIP-173/350 and ERC-55 test vectors, and the
remaining 17 kinds are evaluated on project-authored operational memos
(`operational-memos.txt`):

- Five third-party documents from MASC 3.0.0 (`34,969` bytes, CC BY 3.0
  US, `21` gold spans: `12` `phone` with `10` true, `1` false, `2` missed,
  precision `0.909`, recall `0.833`, `7` `us-zip`, all found, and `2`
  `street-address`, one found): three corporate and non-profit
  solicitation letters (`masc-att2.escaped`, `masc-hsus4.escaped`,
  `masc-united1.escaped`), one New York Times News Service travel
  dispatch (`masc-nyt6.escaped`), and one Legal Services Corporation
  protocol full of CFR, statutory and federal case citations
  (`masc-lsc-access.escaped`, `0` gold spans).
- Two project-adapted specification summaries (`4,360` bytes, Apache-2.0
  prose, `18` gold spans) embedding the canonical published checksum test
  vectors from ERC-55 (`analysis/pii/tests/data/human-eval/eip-55.txt`,
  `10` `eth-address` spans, CC0-1.0) and BIP-173 / BIP-350
  (`analysis/pii/tests/data/human-eval/bip-173-350.txt`, `8`
  `btc-address` spans, BSD-2-Clause), plus invalid mixed-case and
  Bech32/Bech32m near misses and the detector's all-zero burn-address
  exclusion.
- Project-authored incident, treasury, network, payroll/tax, clinical
  and mobile fleet memos (`analysis/pii/tests/data/human-eval/operational-memos.txt`,
  `6,072` bytes, Apache-2.0, `57` gold spans: `3` `phone` spans and every
  gold mention of the remaining 17 kinds) written with documentation,
  test, and synthetic checksum-valid specimen identifiers (RFC 2606,
  RFC 5737, RFC 3849, RFC 7042, RFC 7519, AWS documentation example keys,
  ISO 13616 / Bundesbank specimen IBANs, Federal Reserve routing numbers,
  NHS test numbers, ISO 7064 test Steuer-IDs, SIN `046 454 286`, and
  Luhn-valid test IMEIs), paired with mistyped check digits, order and
  invoice numbers, timestamps, software versions and statutory citations.
- Project-authored fleet, shipping and vendor memos
  (`analysis/pii/tests/data/human-eval/us-identifiers-memos.txt`, `2,635`
  bytes, Apache-2.0, `31` gold spans: `vin`, `tracking-number`, `us-ein`,
  `url`, `us-passport`, `us-drivers-licence`, `us-zip`, `phone` and
  `phone-extension`) with check-digit-valid VINs, the UPS sample number,
  checksum-valid parcel numbers and documentation domains, next to
  near misses; the six web addresses already in the other documents
  are `url` gold as well, and the seven ZIP codes in the MASC letters
  are `us-zip` gold.
- Project-authored clinic intake memos
  (`analysis/pii/tests/data/human-eval/intake-memos.txt`, `1,057` bytes,
  Apache-2.0, `14` gold spans: `7` `age` and `7` `street-address`) with
  made up ages, street addresses and post office boxes, next to near
  misses (`page 34`, `11 years ago`, `aged 3 months`, `Table 4 Summary`,
  `the 5th Avenue satellite`, `Route 9`).

| Kind | Source partition | true | false | missed | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| `jwt` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `aws-access-key` | project-authored (`operational-memos.txt`) | 2 | 0 | 0 | 1.000 | 1.000 |
| `github-token` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `url-credential` | project-authored (`operational-memos.txt`) | 2 | 0 | 0 | 1.000 | 1.000 |
| `email` | project-authored (`operational-memos.txt`) | 3 | 2 | 0 | 0.600 | 1.000 |
| `iban` | project-authored (`operational-memos.txt`) | 4 | 0 | 0 | 1.000 | 1.000 |
| `imei` | project-authored (`operational-memos.txt`) | 3 | 0 | 2 | 1.000 | 0.600 |
| `card` | project-authored (`operational-memos.txt`) | 4 | 2 | 0 | 0.667 | 1.000 |
| `btc-address` | adapted spec (`bip-173-350.txt`, BSD-2-Clause vectors) | 8 | 0 | 0 | 1.000 | 1.000 |
| `eth-address` | adapted spec (`eip-55.txt`, CC0-1.0 vectors) | 10 | 0 | 0 | 1.000 | 1.000 |
| `mac` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `ipv6` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `ipv4` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `us-ssn` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `us-itin` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `uk-nhs` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `de-steuer-id` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `ca-sin` | project-authored (`operational-memos.txt`) | 3 | 0 | 1 | 1.000 | 0.750 |
| `aba-routing` | project-authored (`operational-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `phone` (MASC only) | third-party human-written (`masc-*.escaped`, CC BY 3.0 US) | 10 | 1 | 2 | 0.909 | 0.833 |
| `vin` | project-authored (`us-identifiers-memos.txt`) | 4 | 0 | 0 | 1.000 | 1.000 |
| `tracking-number` | project-authored (`us-identifiers-memos.txt`) | 9 | 0 | 0 | 1.000 | 1.000 |
| `us-ein` | project-authored (`us-identifiers-memos.txt`) | 3 | 0 | 0 | 1.000 | 1.000 |
| `url` | project-authored (`us-identifiers-memos.txt`, `4`) + existing documents (`6`) | 10 | 0 | 0 | 1.000 | 1.000 |
| `us-passport` | project-authored (`us-identifiers-memos.txt`) | 2 | 0 | 0 | 1.000 | 1.000 |
| `us-drivers-licence` | project-authored (`us-identifiers-memos.txt`) | 2 | 0 | 0 | 1.000 | 1.000 |
| `us-zip` | third-party MASC (`7`) + project-authored (`3`) | 10 | 0 | 0 | 1.000 | 1.000 |
| `phone-extension` | project-authored (`us-identifiers-memos.txt`) | 2 | 0 | 0 | 1.000 | 1.000 |
| `age` | project-authored (`intake-memos.txt`) | 7 | 0 | 0 | 1.000 | 1.000 |
| `street-address` | third-party MASC (`2`) + project-authored (`7`) | 8 | 0 | 1 | 1.000 | 0.889 |
| `phone` (all) | third-party MASC (`12`) + project-authored (`3` + `2`) | 15 | 6 | 2 | 0.714 | 0.882 |
| **all kinds** | **mixed (`21` MASC + `18` adapted spec + `57` operational memos + `31` US identifier memos + `14` intake memos + `6` url in existing documents)** | **141** | **10** | **6** | **0.934** | **0.959** |

### External prose partition (whole third-party documents)

A separately scored partition of twenty-three whole third-party documents
(`496,469` bytes, `491` gold mentions,
`precision_and_recall_on_external_prose`, gold in
`analysis/pii/tests/data/human-eval/labels-external.tsv`, provenance in
that directory's `README.md`): RFC 7519, RFC 5737, RFC 3849 and RFC 7042
as retrieved from the RFC Editor (publication-era BCP 78 notices), nine Wikipedia
articles as wikitext at pinned revisions (CC BY-SA 4.0): "International
Bank Account Number", "Social Security number", "Social insurance
number", "NHS number", "International Mobile Equipment Identity",
"Routing transit number", "Individual Taxpayer Identification
Number", "Steuerliche Identifikationsnummer" and "Payment card
number" (the ITIN and Steuer-ID articles carry no gold themselves
and measure precision on identifier-dense prose; their positive
specimens are in the validator READMEs below, as `card` positives
are in the Test Payment Cards list), the BIP-173
and BIP-350 specification texts (BSD-2-Clause) with ERC-55
(CC0-1.0) at pinned source commits, the Everything curl URL
authentication chapter (CC BY 4.0), two AWS documentation
chapters (CC BY-SA 4.0) at pinned awsdocs commits, and the GitHub
Docs REST data file for the credentials-revoke endpoint (CC BY 4.0),
the Test Payment Cards gateway list (CC BY-SA 4.0), and the Uphold
`itin-validator` and Kontist `validate-steuerid` READMEs (both MIT)
at pinned commits.
These rows never pool with the mixed set above.

| Kind | Source partition | true | false | missed | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| `aba-routing` | external (Wikipedia routing wikitext) | 4 | 0 | 4 | 1.000 | 0.500 |
| `aws-access-key` | external (AWS EKS and Rekognition documentation chapters) | 4 | 0 | 0 | 1.000 | 1.000 |
| `btc-address` | external (BIP-173, BIP-350 mediawiki) | 8 | 0 | 3 | 1.000 | 0.727 |
| `ca-sin` | external (Wikipedia SIN wikitext) | 0 | 0 | 2 | N/A | 0.000 |
| `card` | external (Test Payment Cards gateway list) | 314 | 4 | 41 | 0.987 | 0.885 |
| `de-steuer-id` | external (Kontist validate-steuerid README) | 2 | 0 | 0 | 1.000 | 1.000 |
| `email` | external (RFC 7519, RFC 5737, RFC 3849, RFC 7042, BIP-173, BIP-350, ERC-55) | 22 | 2 | 1 | 0.917 | 0.957 |
| `eth-address` | external (ERC-55 markdown) | 14 | 0 | 0 | 1.000 | 1.000 |
| `github-token` | external (GitHub Docs REST credentials data) | 5 | 0 | 0 | 1.000 | 1.000 |
| `iban` | external (Wikipedia IBAN wikitext) | 4 | 0 | 0 | 1.000 | 1.000 |
| `imei` | external (Wikipedia IMEI wikitext) | 0 | 0 | 2 | N/A | 0.000 |
| `ipv4` | external (RFC 5737, AWS Rekognition CloudTrail chapter) | 10 | 3 | 0 | 0.769 | 1.000 |
| `ipv6` | external (RFC 3849) | 3 | 0 | 0 | 1.000 | 1.000 |
| `jwt` | external (RFC 7519) | 0 | 0 | 2 | N/A | 0.000 |
| `mac` | external (RFC 7042) | 29 | 0 | 0 | 1.000 | 1.000 |
| `phone` | external (RFC 5737, RFC 7042) | 5 | 15 | 0 | 0.250 | 1.000 |
| `url-credential` | external (Everything curl URL authentication chapter) | 1 | 0 | 0 | 1.000 | 1.000 |
| `uk-nhs` | external (Wikipedia NHS wikitext) | 1 | 1 | 3 | 0.500 | 0.250 |
| `tracking-number` | external (Test Payment Cards gateway list) | 0 | 1 | 0 | 0.000 | n/a |
| `us-itin` | external (Uphold itin-validator README) | 2 | 0 | 2 | 1.000 | 0.500 |
| `us-ssn` | external (Wikipedia SSN wikitext) | 3 | 0 | 0 | 1.000 | 1.000 |
| **all kinds** | **external (`491` gold; false mentions of kinds without gold included, `url` not annotated and not scored)** | **431** | **26** | **60** | **0.943** | **0.878** |

Zero prediction or gold support makes the corresponding precision or
recall **N/A**, including the card recall when no card is gold-labelled.
These metrics describe the documented annotation scope, not all possible
identifier formats.

What the external false positives and misses are:

- **`email` false positives (2) and miss (1)**: RFC 3849 prints
  `ietf-\n   ipr@ietf.org` over a display line break (bytes 7658..7679).
  The detector finds only the suffix `ipr@ietf.org` (7667..7679), so the
  full address is missed and the suffix is a false positive. The
  second false positive is `password@example.com`, the tail of the
  Everything curl credential URL whose userinfo carries the
  `url-credential` gold; the tail is not an address occurrence and
  is not labelled.

- **`phone` false positives (15)**: six five-group prefixes of EUI-64
  identifiers and SNAP header patterns in RFC 7042; the three NHS-shaped
  values in the NHS article the `uk-nhs` scanner rejects (below) plus
  the checksum example `943 476 5919` on the same span as its `uk-nhs`
  mention, and the Scottish CHI range endpoints `010 100 0000` and
  `311 299 9999`; the multiplier row `0 8 6 8 5 8 2 16 6` of the SIN
  checksum walkthrough; `5698 7654 32` inside the markup-split
  British IBAN example; and the digit run of an inline SVG icon path in the GitHub Docs
  credentials data.
- **`card` false positives (4) and misses (41)**: the false
  positives are the 15-digit grouped IMEI examples and a contiguous
  14-digit walkthrough prefix in the IMEI article (only complete
  IMEIs carry `imei` gold); the 14-digit archive timestamps in
  Wikipedia citation templates that were Luhn-valid by coincidence
  sit inside archive URLs, and the `url` mention now holds them; the Test Payment Cards list itself draws no
  false card mention. The misses are all in that list: 34 Pay360
  test numbers in the 9905 and 9900 ranges and two 1-series UATP
  numbers, whose first digit is outside the scanner's 2-to-6 or 81
  rule, and five numbers the gateways publish as test cards whose
  check digit fails Luhn.
- **`ipv4` false positives (3)**: the Simple Web Token version `0.9.5.1`
  in RFC 7519's references and the section number `2.3.2.1` twice in
  RFC 7042.
- **`uk-nhs` false positive (1)**: `0123456789` in ERC-55's hexadecimal
  digit alphabet, which satisfies the modulus-11 check. The NHS-valid
  `0110201035` in the SSN article's citations, and the article
  identifier `2101-153765` that was a `phone` false positive in the
  Steuer-ID article, are both inside citation URLs, and the `url`
  mention now holds them.
- **`tracking-number` false positive (1)**: `491748000000` in the Test
  Payment Cards list, a 12-digit expiry-driven test number that fails
  Luhn and passes the FedEx Express check by coincidence (about one
  12-digit run in eleven does). `label_required(PiiKind::TrackingNumber)`
  removes it.
- **`btc-address` misses (3)**: the witness version 1, 2 and 16
  addresses of BIP-173's own valid list. BIP-173 encodes every
  version with the original bech32 checksum; BIP-350 replaced it
  with bech32m for version 1 and up, and the scanner follows
  BIP-350, so the three BIP-173-era encodings fail its checksum
  split. The document publishes them as valid mainnet addresses,
  so they are gold occurrences in a form the scanner does not take.
  Testnet addresses in both BIPs are not labelled.
- **`aba-routing` misses (4)**: the dashed forms `0260-0959-3` and
  `2260-7352-3`, which the scanner does not take, and the historical
  check-image MICR example `129131673` (twice), whose check digit does
  not satisfy the modern ABA checksum.
- **`ca-sin` misses (2)** and **`imei` misses (2)**: the labelled
  scanners need their keyword in the accepted forms; the SIN examples
  carry no adjacent `SIN` label and both IMEI examples use the
  2-6-6-1 grouped presentation after the words "IMEI code", which the
  scanner does not take (it reads them as `card` instead).
- **`uk-nhs` misses (3)**: the article's given example `987 654 4321`
  fails the modulus-11 check digit, as does the reserved test-range
  start `999 000 0000`; `999 999 9999` passes the checksum but is
  refused by the scanner's repeated-digit defences.
- **`jwt` misses (2)**: the signed and unsecured examples in RFC 7519
  sections 3.1 and 6.1 are line-wrapped for display, with separating
  periods on separate lines. They are the evaluated subset; Appendix A
  also publishes complete encrypted/nested JWTs (JWE serializations),
  excluded by this partition's annotation policy. The scanner requires
  contiguous nonempty segments; wrapping and the unsecured example's
  empty signature are unsupported. This subset's recall is `0.000`.
- **`us-itin` misses (2)**: the compact `900700000` form printed
  twice by the Uphold README. The scanner takes hyphen- and
  space-grouped ITINs, so both dashed occurrences are found and
  both compact occurrences are missed.

#### External coverage for #968

The external partition establishes positive third-party coverage for
every built-in kind: `aba-routing`, `aws-access-key`, `btc-address`,
`ca-sin`, `card`, `de-steuer-id`, `email`, `eth-address`,
`github-token`, `iban`, `imei`, `ipv4`, `ipv6`, `jwt` (recall
`0.000` on the signed/unsecured subset, which is display-wrapped),
`mac`, `uk-nhs`, `url-credential`, `us-itin`, `us-ssn`, and further
third-party `phone`. The ITIN and Steuer-ID Wikipedia articles and
IRS Publication 1915 print format patterns and algorithms but no
complete specimen, so those documents remain precision evidence;
the positive specimens are in the MIT validator READMEs above. The
`btc-address` and `eth-address` coverage is the whole BIP-173,
BIP-350 and ERC-55 specification texts, superseding the adapted
summaries as the crypto evidence.

Future corpus additions still need attributable, licensed source
passages with appropriate context and per-occurrence gold labels,
including scanner misses and negative examples. Preserve the fixture
policy for safe specimen values; where a source requires replacement
of a value, record that transformation and retain the source context
rather than authoring replacement prose. Pin the source version,
licence, text digest and final UTF-8 annotation offsets. The mixed
metrics remain separate from measurements on external prose.

What the false positives and misses are:

- **`phone` false positives (5)**: `2002-07-31 13` in the NYT wire
  timestamp `2002-07-31 13:47` (`masc-nyt6.escaped`), where the date and
  two-digit hour before `:47` form ten digits separated by hyphens and a
  space; `0532 0130 00` inside the rejected near-miss IBAN
  `DE88 3704 0044 0532 0130 00` (`operational-memos.txt`), because when
  the outer IBAN fails mod 97 no `iban` mention is emitted to suppress
  its trailing ten grouped digits; `12 134 567 895`
  (`operational-memos.txt`), where an 11-digit grouped German Steuer-ID
  shares the exact span of an 11-digit domestic phone candidate (default
  nesting suppression drops strictly nested spans, not identical spans);
  `943 476 5919` (`operational-memos.txt`), where the space-grouped UK
  NHS number also matches the ten-digit phone rule on the same span; and
  `943 476 5918` in `NHS 943 476 5918` (`operational-memos.txt`), where
  the mistyped NHS check digit is rejected by `uk-nhs`, leaving ten
  grouped digits.
- **`card` false positives (2)**: both occurrences of the 15-digit GSMA
  test IMEI `490154203237518` (`operational-memos.txt`) begin with `49`
  and pass Luhn, matching the 13-to-19-digit Visa card rule on the same
  span as `imei` (and once without an `IMEI` label in a prose back-reference).
- **`email` false positives (2)**: in the `https://` and `ftp://`
  credential URLs of `operational-memos.txt`, the `password@host` tails
  (`stage-pass99@artifacts.example.com` and `mirror-key42@ftp.example.org`)
  match `email` crossing `url-credential`.
- **`phone` misses (2)**: `973.504.6215` and `(919) 807.2214` in the
  state charity-registration disclosures of `masc-hsus4.escaped` use `.`
  as a digit group separator, whereas the phone scanner accepts spaces,
  hyphens and parentheses only.
- **`ca-sin` (1) and `imei` (2) misses**: `SIN (ON) 046 454 286` and
  `IMEI (primary) 001970012345677` in `operational-memos.txt` place a
  parenthesised qualifier word between the keyword label and the digits,
  putting the label at word distance 2 outside `ca-sin` and `imei`'s
  default 1-word window (`window::default_window`); and the second
  occurrence of `490154203237518` in
  `"Note that 490154203237518 begins with..."` (`operational-memos.txt`)
  has no `IMEI` keyword in its window.
