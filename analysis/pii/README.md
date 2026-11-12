# turbo-pii

Regex-free personal and sensitive identifier detection, risk scoring, and view-aligned redaction on the `turbo-analysis` `Document` graph.

`turbo-pii` (`analysis/pii/`) scans UTF-8 text with hand-written byte scanners and in-crate checksum/cryptographic validators, records structured findings and candidate dispositions as typed document layers, and optionally rewrites mentions into derived `View`s whose `Alignment` chains map spans in both directions across UTF-8 bytes, UTF-16 code units, and Unicode code points.

Full scanner rules, per-pack tables, and corpus evaluation figures live in `docs/pii.md`. Server policy, disclosure, and audit wire contracts live in `docs/grpc-analysis.md`.

## Architecture: Annotations, Rewritten Views, and Value Release

PII handling in TurboEmbed separates three concerns that are easy to conflate: **annotating spans**, **producing scrubbed text for downstream stages**, and **releasing protected values from a graph**.

### 1. Detection is an annotation layer, not secrecy from the caller

`PiiDetector` never mutates the input `View`. It writes findings onto the `turbo:pii` layer (`Pii`), candidate decisions onto `turbo:pii-candidates` (`PiiCandidates`), and optional risk summaries onto `turbo:pii-risk` (`PiiRisk`).

Over gRPC (`InspectDocumentGraph`, `RoundTripDocumentGraph`, `AnalyzeDocumentProgressive`, and sink items), responses for documents with `turbo:pii` findings carry **findings only** by default (`PiiValue.withheld`): each mention's kind, confidence tier, optional evidence, spans, and provenance.

- **Why values are withheld when the caller sent the text:** A caller that just submitted raw text to normalize or analyze already holds that plaintext in memory and slices it directly using the returned spans. Default withholding is not hiding the caller's own input from the caller; it keeps raw source text, normalized strings, and graph bytes out of default responses, logs, caches, and downstream consumers that did not ask for cleartext (`PiiOutputOptions.full_text`).
- **Annotation-first pipelines:** When an upstream caller owns the source text and only needs span boundaries and entity types, running `PiiDetector` alone (without a rewrite step) gives typed span annotations with zero key management.

### 2. Rewriting builds a derived `View` for downstream data hygiene

In ingestion pipelines (`ingest -> normalize -> chunk -> embed / index / store`), the caller typically drops raw text and hands normalized output to downstream stages. If raw PII remained in the text alongside span annotations, every downstream consumer (chunkers, tokenizers, embedding models, BM25/vocabulary builders, loggers) would have to strip those spans itself or risk baking raw identifiers into vector embeddings and indexes.

Rewrites solve that downstream hygiene problem while keeping structured metadata in annotation layers:

- **Derived `View` (`masked` or `redacted`):** `MaskedView` and `RedactedView` (server step `turbo-pii:redact`) write scrubbed or transformed text into a child `View` with a complete stepwise `Alignment` back to its parent view. Downstream chunking, tokenization, vocabulary, and embedding steps run over that derived view, and callers can request `PiiOutputOptions.rewritten_text` without disclosing `full_text`.
- **Edit provenance (`turbo:pii-edits`) and dry-run plans (`turbo:pii-plan`):** `RedactedView` pairs the derived view with `turbo:pii-edits` (`PiiEdits`), recording each replacement's derived span, parent source span, covered `turbo:pii` member indices, `PiiRule`, `PiiStrategyKind`, policy revision, and canonical SHA-256 policy digest, with **no original values**. Dry runs write `turbo:pii-plan` (`PiiPlan`) on the source view with no text copy.
- **Keeping text clean for embeddings vs inline tokens:** Non-cryptographic strategies (`MaskPolicy`, `StaticReplacer`, `FormatRedactor`, `Truncate`, `Abbreviator`, `ZipSafeHarbor`, `AgeSafeHarbor`, `DateRewrite`) keep `.text` clean for tokenizers and embedding models while `turbo:pii` and `turbo:pii-edits` carry the typed metadata. `Ff1Encryptor` preserves character class and separators in place (such as `ddd-dd-dddd`) for downstream consumers that parse fixed formats. `AesGcmEncryptor` writes self-contained `KIND:<base64url>` ciphertext tokens inline when the text itself serves as the reversible archive; when clean text plus separate encrypted values are preferred, mask the view and attach encrypted values on the `turbo:pii` annotations via server ciphertext release (`PiiValueCiphertext`).

### 3. Shared-key threat model: why deterministic transforms differ

Even when a caller already knows the text it sent in the current request, strategies that use a **server-held secret key** fall into two cryptographic classes with different risk profiles against *previously stored* documents:

| Strategy class | Implementations | Reversible? | Cross-document linkable? | Lookup / guessing oracle risk if exposed to untrusted callers? |
|---|---|---|---|---|
| **Keyless scrubs & generalizations** | `MaskPolicy`, `StaticReplacer`, `FormatRedactor`, `Pseudonymizer` (per-doc), `Truncate`, `Abbreviator`, `MapReplacer`, `ZipSafeHarbor`, `AgeSafeHarbor`, unkeyed `DateRewrite` | No | No | **None.** Output depends only on the current document or public rules. |
| **Unkeyed cryptographic digest** | `Sha256Tokenizer` | No (by key), guessable on small domains | Yes (globally) | **Offline brute-forceable** without the server on bounded domains (9-digit SSNs, phone numbers). Never a default. |
| **Randomized authenticated encryption** (`reversible` feature) | `AesGcmEncryptor` (`aes_256_gcm`) | Yes, offline with 32-byte key | No (fresh 12-byte nonce per mention) | **None.** Encrypting candidate inputs yields fresh ciphertexts each time and reveals nothing about stored ciphertexts. |
| **Deterministic keyed transforms** | `HmacTokenizer` (`keyed_token`), `Ff1Encryptor` (`ff1`, `reversible` feature), keyed `DateShift` | `ff1`: Yes, offline with 32-byte key; `HmacTokenizer`: No | Yes (per key and tweak/context) | **Online confirmation / dictionary oracle.** Same `(key, tweak, input)` yields the same output, so an untrusted caller able to query it can batch-test candidate values (for example partial SSNs) against stored redacted records without holding the key. |

### 4. Redaction encryption keys vs audited value release on the server

`turbo-kserve` keeps pipeline redaction encryption separate from privileged value release (`docs/grpc-analysis.md`):

- **Redaction encryption (`--pii-encryption-key ID:VERSION:PATH`):** Supplies 32-byte AES-256 keys for `RedactionPolicy` `aes_256_gcm` and `ff1` arms during `turbo-pii:redact`. The server only encrypts into the `redacted` view and never decrypts; possession of the key file outside the server controls who can read original values later.
  - Loading `--pii-encryption-key` does **not** require `--pii-grant`, mTLS, or an audit endpoint, and never enables value release on its own.
  - **Inline-policy guard (`NamedOnlyEncryption`):** When a server runs without `--pii-grant` (so callers need not be mTLS-authenticated), grant-less encryption keys resolve **only** for operator-loaded `--pii-policy` named policies. Caller-supplied inline `pii_redaction` policies referencing an encryption key are refused (`INVALID_ARGUMENT`), preventing an unauthenticated caller from supplying an inline `ff1` arm with a chosen tweak and using the server as an encryption oracle.
  - **Ingress guidance for deterministic named policies:** Because a named policy that configures `ff1` still computes a deterministic function under its configured tweak, non-mTLS listeners serving named `ff1` policies must sit behind trusted network ingress (localhost sidecar, service mesh, or private network), not an untrusted public edge.
- **Audited value release (`InspectDocumentGraph` + `--pii-grant`):** Populates `PiiMention.value` (`PiiValueToken`, `PiiValueCiphertext` under context `turbo-kserve.pii-release.v1`, or `PiiValueClear`) directly on the `turbo:pii` annotation layer when inspecting a graph. Value release always requires `--pii-grant`, mTLS client identity, `--pii-audit-endpoint`, policy release-ceiling checks (`PiiReleaseCeiling`), and a confirmed synchronous `RecordPiiAccess` audit event before any value leaves the server. Ciphertext release shares the `--pii-encryption-key` key set (selecting the active release key via `--pii-encryption-key-active`), while `--pii-grant` gates value release itself; `--pii-token-key` and both `*-active` flags remain gated on `--pii-grant`.

## Document Layers and Views

| Artifact | Kind | Native `View` | Carries raw/normalized PII values? | Purpose |
|---|---|---|---|---|
| `turbo:pii` (`Pii`) | Layer | Scanned view | In-process `normalized` column; **withheld by default** on wire (`PiiValue.withheld`) | Ordered `PiiMention` spans, `PiiKind` (built-in or per-layer `CustomKinds`), `PiiDetectorRef`, typed `PiiEvidence`, and optional `PiiScore` |
| `turbo:pii-candidates` (`PiiCandidates`) | Layer | Scanned view | No | Every candidate span considered and its `PiiDisposition` (accepted, `RejectReason`, or `AllowRule`) |
| `turbo:pii-risk` (`PiiRisk`) | Layer | Scanned view | No | Per-document risk score, `PiiRiskLevel` (`NONE`..`CRITICAL`), per-kind counts, severities, and suggested rewrites |
| `turbo:pii-plan` (`PiiPlan`) | Layer | Scanned view | No | Dry-run preview mapping 1:1 with `turbo:pii`: chosen `PiiRule`, `rule_id`, `PiiStrategyKind`, revision, and policy digest |
| `masked` (`MaskedView`) | Derived `View` | Child of scanned view | Masked characters only | Byte-length-preserving character masking with 1:1 character boundary `Alignment` |
| `redacted` (`RedactedView`) | Derived `View` | Child of scanned view | Rewritten replacements / ciphertexts only | General policy rewrite (`RewritePolicy`) with full stepwise UTF-8 `Alignment` |
| `turbo:pii-edits` (`PiiEdits`) | Layer | `redacted` view | No | One record per replacement: derived span, parent source span, covered `turbo:pii` member rows, `PiiRule`, `PiiStrategyKind` (`1..=17`), revision, and policy digest |

## Crate Layout

- `src/detector.rs`, `src/detector/*`, `src/hits.rs`: `PiiDetector`, `Pack` presets (`Payment`, `Contact`, `Network`, `Secrets`, `Crypto`, `UsIdentity`, `EuIdentity`, `BrIdentity`, `CaIdentity`, `Device`, `Logistics`, `Web`, `Places`, `AllStructured`), nesting resolution, `PiiSink`, and custom extractor registration (`CustomExtractor`).
- `src/scan/`: Hand-written regex-free byte scanners for structured identifiers, credentials, payment cards, IBANs, crypto addresses, national IDs, phones, emails, URLs, VINs, parcel tracking numbers, dates, money, ages, street addresses, US states, counties, and cities.
- `src/checksum/`: In-crate validators for Luhn, ISO 7064 mod 97 and mod 11, ICAO 9303, Spanish/French/German national ID checks, BIC, North American VIN (49 CFR 565.15), UPS/FedEx/GS1 parcel check digits, Verhoeff, Damm, SHA-256, Keccak-256 (EIP-55), Base58Check, and Bech32/Bech32m (BIP-173/BIP-350).
- `src/window.rs`: UAX #29 word-boundary context windows and `LabelPolicy` (`None`, `Boosts`, `Required`).
- `src/lists/`: Global and per-kind `AllowList`s, regex-free `Pattern` (`Shape` and `Glob`), and Aho-Corasick / bounded-Levenshtein fuzzy `DenyList`s (`examples/fuzzy_deny_bench.rs`).
- `src/risk.rs`: `PiiRiskScorer` writing `turbo:pii-risk`.
- `src/rewrite/`: `MaskPolicy`, `MaskedView`, `RedactedView`, `RewritePolicy`, `RewriteStrategy`, `Pseudonymizer`, `HmacTokenizer`, `FormatRedactor`, `StaticReplacer`, `Sha256Tokenizer`, `MapReplacer`, `Truncate`, `Abbreviator`, `ZipSafeHarbor`, `AgeSafeHarbor`, `DateRewrite`, `PiiAuditReport`, and (behind `--features reversible`) `AesGcmEncryptor` and `Ff1Encryptor`.

## Dependencies and Cargo Features

- Default build has **no external cryptography or regex crates**: depends only on workspace crates (`turbo-analysis`, `turbo-analysis-layers`, `turbo-glossary`, `turbo-geo`, `turbo-numeric`, `turbo-sha256`, `turbo-unicode`).
- `reversible` feature: enables `AesGcmEncryptor` and `Ff1Encryptor` via `aes-gcm` and `fpe` (allowlisted in `bench/tests/tree.rs` under #1885; surfaced in `turbo-kserve` via `--features pii-reversible`).

## Testing

```sh
# Core detector, checksum, list, view-alignment, and rewrite suites
cargo test -p turbo-pii

# Reversible AES-256-GCM (NIST CAVP) and FF1 (NIST SP 800-38G) round-trip and policy suites
cargo test -p turbo-pii --features reversible --test reversible --test reversible_policy

# Workspace hygiene and dependency/em-dash/path tree checks
cargo test -p turbo-bench --test tree
```

Key integration suites in `analysis/pii/tests/`:
- `goldens.rs` and `rewrite_goldens.rs`: deterministic pack detection and rewrite snapshots across 74 synthetic inputs (`TURBO_BLESS=1` updates snapshots after a reviewed behaviour change).
- `precision_recall.rs`, `us_identifiers_eval.rs`, `places_eval.rs`: synthetic, mixed MASC/memo, and whole-document external prose evaluation partitions (`tests/data/human-eval/README.md`).
- `masked_view.rs`, `redacted_view.rs`, `xray_masked.rs`: bidirectional UTF-8 / UTF-16 / code-point span projection across multi-byte scalars, ZWJ emoji, overlaps, and unequal replacement lengths.
- `rewrite_strategies.rs`, `rewrite_audit.rs`, `chosen_rewrite.rs`, `zip_safe_harbor.rs`, `age_safe_harbor.rs`: strategy semantics and verification that `PiiAuditReport` never leaks raw or normalized mention text in `Display` or `Debug`.
