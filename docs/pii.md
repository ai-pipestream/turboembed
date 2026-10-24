# PII detection

`turbo-pii` (`analysis/pii/`) finds personal and sensitive identifiers in
a text view and writes them as the `turbo:pii` layer: e-mail addresses,
phone numbers, payment cards, IBANs, ABA routing numbers, network
addresses, credentials, Bitcoin and Ethereum addresses, national
identifiers and device identifiers. Each kind with a check digit or
checksum is validated with it. It is a port of the `opennlp.tools.pii`
package of the OpenNLP fork the other ports follow (commit `cdf33c6`):

| Rust | OpenNLP |
|---|---|
| `PiiDetector` | `CompositePiiExtractor` and `PiiAnnotator` |
| `Pack`, `PiiDetector::pack` | `PiiPacks` |
| `Extractor`, `Scanner` | the extractor classes (`CursorPiiExtractor`, `NetworkPiiExtractor` and the others) and their type selection |
| `Pii`, `PiiMention`, `PiiKind` | `PiiMention`, the type strings, `PiiTypePriority` |
| `PiiConfidence` | none (below) |
| `rewrite::mask`, `mask_all`, `mask_by_kind`, `MaskPolicy` | `Masker`, `MaskPolicy`, `MaskPolicies` |
| `rewrite::MaskedView`, `mask_view` | `Masker` with `MaskPolicies.byType()`, as a view |
| `rewrite::Pseudonymizer`, `HmacTokenizer`, `PiiRewrite` | `Pseudonymizer`, `HmacTokenizer`, `PiiRewrite` |
| `rewrite::PiiAuditReport` | `PiiAuditReport` |
| `checksum::luhn`, `mod97`, `sha256`, `keccak`, `base58`, `bech32` | `Luhn`, the IBAN check, `MessageDigest` and `Mac`, `Keccak256`, `Base58Check`, `Bech32` |

The crate depends on `turbo-analysis`, `turbo-analysis-layers` and
`turbo-unicode` only. SHA-256, HMAC-SHA-256, Keccak-256, base58check,
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
    println!("{}..{} {} {} {}", m.span.start, m.span.end, m.kind, m.confidence.name(), m.normalized);
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
| `Contact` | `contact()` | the cursor scanner limited to e-mail and phone | `email`, `phone` |
| `Network` | `network()` | network | `ipv4`, `ipv6`, `mac` |
| `Secrets` | `secrets()` | secrets | `aws-access-key`, `github-token`, `jwt`, `url-credential` |
| `Crypto` | `crypto()` | crypto | `btc-address`, `eth-address` |
| `UsIdentity` | `usIdentity()` | US identity | `us-ssn`, `us-itin` |
| `EuIdentity` | `euIdentity()` | EU identity | `uk-nhs`, `de-steuer-id` |
| `CaIdentity` | `caIdentity()` | CA identity | `ca-sin` |
| `Device` | `device()` | device | `imei` |
| `AllStructured` | `allStructured()` | cursor, secrets, crypto, network, US, EU and CA identity, device, banking | every kind |

`PiiDetector::cursor()` is the default `CursorPiiExtractor`: e-mail,
phone, IBAN and card.

## The layer

`Pii`, in `turbo-analysis-layers`, has one item per mention with the
columns `spans` (UTF-8 byte offsets into the view, on character
boundaries), `kind`, `confidence` and `normalized`. Mentions are ordered
by start, then longer first, then by kind rank; equal mentions (span,
kind and normalized form) appear once, the first found kept. By default,
a mention strictly nested inside a longer mention (such as a phone or
card inside a grouped IBAN, or an NHS number inside an international
phone number) is dropped; `PiiDetector::keeping_nested` keeps it for
OpenNLP parity. Mentions of different kinds may still share an exact
span or cross each other: an IMEI after its label is also a Luhn-valid
card number, and `https://user:pass@example.com` has a URL credential
crossing an e-mail address. `Pii::check` holds a layer to this order.

The kinds, in rank order (`PiiKind::ALL`, `PiiKind::rank`), with the
normalized form and the confidence each gets:

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
| `ca-sin` | the digits | labelled checksum (Luhn) |
| `aba-routing` | the digits | checksum (3, 7, 1 weights) |
| `phone` | `+` and digits, or the digits | format |

`PiiConfidence` is not in OpenNLP, which reports none. It is read from
what the scanner checked, so it never changes which mentions are found:
`Format` when only the shape backs the kind, `Checksum` when a check digit
or checksum over the value passes, `LabelledChecksum` when a label names
the kind as well. The goldens hold no confidence.

## Masking and labels

`rewrite::mask` overwrites the characters of each span with a policy;
`mask_by_kind` uses `MaskPolicy::for_kind` (`MaskPolicies.forType`: cards
and IBANs keep separators and their last four letters or digits, contact,
network, national and device identifiers and routing numbers keep
separators, credentials and wallet addresses are masked whole), and
`mask_all` masks every character. Each span is masked against the
original text, in order, and what a later policy keeps readable does not
undo an earlier mask, so overlapping mentions are both masked.

OpenNLP writes one mask character per UTF-16 unit. Here the mask is one
ASCII byte (`MaskPolicy::new` refuses any other, `MaskError::NotAscii`)
and a masked character becomes as many mask bytes as it has UTF-8 bytes,
so the masked text has the byte length of the original and every byte
offset stays valid. Every detected mention is ASCII, so the two write the
same text for them; they differ only for spans over non-ASCII text, which
a caller can pass to `mask`.

`Pseudonymizer` replaces each mention with a numbered label (`EMAIL-1`),
the same label for the same kind and value within one text.
`HmacTokenizer` replaces it with the kind in capitals, `-` and the first
hexadecimal digits of the HMAC-SHA-256 of the kind name, a zero byte and
the normalized value, under a key of at least 32 bytes; the label is the
same in every text for one key. Both return a `PiiRewrite` with the new
text, the labels as a layer of it, and `map_offset` and `map_span` from
the original to the new text; both refuse overlapping mentions
(`RewriteError`). `PiiAuditReport` counts mentions by kind, with keyed
tokens in place of values. `analysis/pii/tests/rewrite.rs` checks, for
every pack over every text of the `pii` input set, that no mention's
covered text or normalized value appears in the report's `Display` or
`Debug` output.

The key comes from the caller and is at least 32 bytes. `HmacTokenizer`
copies it and leaves it out of its `Debug` output. Tokens made with one
key are linkable across texts, so they need the access control of the
values they stand for.

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

## Differences from OpenNLP

- Offsets are UTF-8 bytes; OpenNLP's are UTF-16 units. They differ after
  any non-ASCII character (after U+10428 the byte offset is 4 more, the
  unit offset 2 more). The goldens record UTF-8 offsets, so the
  comparison is exact.
- Masks keep the byte length, not the UTF-16 length (above).
- A detector is built from the extractor classes the package has.
  OpenNLP's `PiiExtractor` interface also takes extractors written by a
  caller with type strings of their own; there is no such extension here,
  so the Java tests of custom types (`PseudonymizerIdentity` with custom
  types, the audit escaping of custom type names, the read counts of a
  custom `CharSequence`) are not ported. Their ordering and masking cases
  are ported with built-in kinds.
- `PiiConfidence` is added (above).
- By default, `PiiDetector` drops a mention whose span lies strictly
  inside a longer mention, so trailing groups of a grouped IBAN are not
  also reported as a phone or card and ten digits inside an
  international phone number are not also reported as an NHS number.
  `PiiDetector::keeping_nested` keeps nested mentions as OpenNLP does;
  the golden and differential parity checks use it.

## OpenNLP behaviour kept

These are kept for parity; none is clearly wrong, so the port matches
them.

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

Every fixture is synthetic. Where an OpenNLP test or corpus held a value
that could be real, it is replaced, and the tests say so:

- The legacy Bitcoin addresses of the golden corpora
  (`analysis/pii/tests/data/golden-true-positives.tsv`,
  `analysis/pii/tests/data/golden-false-positives.txt`) and of the crypto
  tests are made up: a version byte over a hash of a fixed string, with
  its base58check checksum.
- Public DNS resolver addresses are replaced by documentation ranges
  (`192.0.2.0/24`, `2001:db8::/32`) or made up addresses, and real host
  names by `example.com`.
- An SSN known to have been published in an advertisement is left out.

## Golden files

JVM goldens come from the goldens tool (`analysis/goldens-tool`), as for
the other OpenNLP ports ([analysis.md](analysis.md#golden-files)).

- Inputs: the `pii` set, `analysis/goldens/data/inputs/pii/`, 61 texts
  written for the packs with only documented test values, example ranges
  and made up values.
- Goldens: `analysis/goldens/data/pii/<pack>/pii/<id>.golden`, one
  directory per pack (`payment`, `contact`, `network`, `secrets`,
  `crypto`, `us-identity`, `eu-identity`, `ca-identity`, `device`,
  `all-structured`), from `PiiAnnotator` over the `PiiPacks` extractor.
  Layer `turbo:pii` with the columns `start`, `end` (UTF-8 byte offsets),
  `type` and `normalized`; the `config pack` record names the pack.
- Rewrite goldens: `pii-rewrite/<pack>/pii/<id>.golden` under the same
  data directory, what `Masker` (with `MaskPolicies.byType()` and with
  `#`), `Pseudonymizer`, `HmacTokenizer` and `PiiAuditReport` make of
  the pack's mentions under the test key in the `config key` record: the
  masked text's edits as `turbo:view`, the text with every mention
  masked, both rewrites or the exception each threw, their labels with
  the spans they replaced, and the audit report by type and as text.
  `analysis/goldens-tool/README.md` lists the columns. A character
  OpenNLP masks is written as one mask byte per UTF-8 byte, as here.

Make them in the recipe container with

```
cargo run -p turbo-goldens-tool -- generate
```

`analysis/pii/tests/goldens.rs` runs every pack with
`PiiDetector::keeping_nested` over the set and compares the layer with
the goldens row for row; while the `pii` golden directory is not on disk
it says so and passes. The differential fuzzer covers the ten packs in
the same parity mode (`analysis/fuzz/src/runs/pii.rs`), and the `pii`
set is among the sets it writes generated texts into.

`analysis/pii/tests/rewrite_goldens.rs` runs every pack (with
`keeping_nested`) and `MaskedView` over the set and compares the masked
view's edits, the fully masked text, both rewrites (or that both sides
refuse overlapping mentions; the Java message gives UTF-16 offsets, so
only the refusal is compared), the labels with the spans they replaced
and the offset map at them, and the audit report, exactly. While the
`pii-rewrite` golden directory is not on disk it says so and passes. The
differential fuzzer leaves these runs out (`NOT_COVERED` in
`analysis/fuzz/src/runs/catalog.rs`), since they compare texts and
reports rather than layers.

## Precision and recall

`analysis/pii/tests/precision_recall.rs` measures the widest pack
(`Pack::AllStructured`) on two sets: a synthetic set built to exercise
the scanners' rules, and a human-written evaluation set in
`analysis/pii/tests/data/human-eval/` (`analysis/pii/tests/data/human-eval/README.md`).
A found mention is true when a label has its span and kind; every other
found mention is false. `PII_SHOW_FALSE=1` lists the false mentions.

### Synthetic set

`precision_and_recall_on_the_synthetic_set` runs on a labelled set made
from a fixed seed (40 values of every kind built to pass their checks,
each in a sentence, and near misses of the checksummed kinds with no
label) together with OpenNLP's golden true and false positive corpora:
1112 texts, 720 labels made in the test.

| | true | false | missed | precision | recall |
|---|---|---|---|---|---|
| all kinds | 744 | 40 | 0 | 0.949 | 1.000 |
| `phone` | 122 | 40 | 0 | 0.753 | 1.000 |
| every other kind | 622 | 0 | 0 | 1.000 | 1.000 |

The false mentions are a phone in each near miss NHS number (ten grouped
digits are a phone shape); phones and cards nested inside grouped IBANs
and NHS numbers nested inside US phone numbers are dropped by default.
The set is made to pass the checks the scanners make, so recall on it
says the scanners find what they are built to find, not how they do on
text written by people.

### Human-written text

`precision_and_recall_on_human_written_text` scores the same pack on
eight human-written documents (`45,345` UTF-8 bytes, `87` gold mentions
across all 20 built-in kinds in `analysis/pii/tests/data/human-eval/labels.tsv`;
licences and SHA-256 digests in `analysis/pii/tests/data/human-eval/README.md`
and `docs/licences.md`):

- Five documents from MASC 3.0.0 (CC BY 3.0 US): three corporate and
  non-profit solicitation letters (`masc-att2.escaped`,
  `masc-hsus4.escaped`, `masc-united1.escaped`), one New York Times News
  Service travel dispatch (`masc-nyt6.escaped`), and one Legal Services
  Corporation protocol full of CFR, statutory and federal case citations
  (`masc-lsc-access.escaped`).
- Specification prose and published checksum test vectors from EIP-55
  (`analysis/pii/tests/data/human-eval/eip-55.txt`, CC0-1.0) and
  BIP-173 / BIP-350 (`analysis/pii/tests/data/human-eval/bip-173-350.txt`,
  BSD-2-Clause), including invalid mixed-case and Bech32/Bech32m near
  misses.
- Project-authored incident, treasury, network, payroll/tax, clinical
  and mobile fleet memos (`analysis/pii/tests/data/human-eval/operational-memos.txt`,
  Apache-2.0) written with published documentation and specimen ranges
  (RFC 2606, RFC 5737, RFC 3849, RFC 7042, RFC 7519, AWS documentation
  example keys, ISO 13616 / Bundesbank specimen IBANs, Federal Reserve
  specimen routing numbers, the NHS `999` test range, BZSt test
  Steuer-IDs, CRA specimen SIN `046 454 286`, and GSMA / 3GPP TS 34.108
  test IMEIs), paired with mistyped check digits, order and invoice
  numbers, timestamps, software versions and statutory citations.

| Kind | true | false | missed | precision | recall |
|---|---:|---:|---:|---:|---:|
| `jwt` | 3 | 0 | 0 | 1.000 | 1.000 |
| `aws-access-key` | 2 | 0 | 0 | 1.000 | 1.000 |
| `github-token` | 3 | 0 | 0 | 1.000 | 1.000 |
| `url-credential` | 2 | 0 | 0 | 1.000 | 1.000 |
| `email` | 3 | 2 | 0 | 0.600 | 1.000 |
| `iban` | 4 | 0 | 0 | 1.000 | 1.000 |
| `imei` | 3 | 0 | 1 | 1.000 | 0.750 |
| `card` | 4 | 2 | 0 | 0.667 | 1.000 |
| `btc-address` | 8 | 0 | 0 | 1.000 | 1.000 |
| `eth-address` | 10 | 0 | 0 | 1.000 | 1.000 |
| `mac` | 3 | 0 | 0 | 1.000 | 1.000 |
| `ipv6` | 3 | 0 | 0 | 1.000 | 1.000 |
| `ipv4` | 3 | 0 | 0 | 1.000 | 1.000 |
| `us-ssn` | 3 | 0 | 0 | 1.000 | 1.000 |
| `us-itin` | 3 | 0 | 0 | 1.000 | 1.000 |
| `uk-nhs` | 3 | 0 | 0 | 1.000 | 1.000 |
| `de-steuer-id` | 3 | 0 | 0 | 1.000 | 1.000 |
| `ca-sin` | 3 | 0 | 1 | 1.000 | 0.750 |
| `aba-routing` | 3 | 0 | 0 | 1.000 | 1.000 |
| `phone` | 14 | 4 | 2 | 0.778 | 0.875 |
| **all kinds** | **83** | **8** | **4** | **0.912** | **0.954** |

What the false positives and misses are:

- **`phone` false positives (4)**: `2002-07-31 13` in the NYT wire
  timestamp `2002-07-31 13:47` (`masc-nyt6.escaped`), where the date and
  two-digit hour before `:47` form ten digits separated by hyphens and a
  space; `0532 0130 00` inside the rejected near-miss IBAN
  `DE88 3704 0044 0532 0130 00` (`operational-memos.txt`), because when
  the outer IBAN fails mod 97 no `iban` mention is emitted to suppress
  its trailing ten grouped digits; `12 134 567 895`
  (`operational-memos.txt`), where an 11-digit grouped German Steuer-ID
  shares the exact span of an 11-digit domestic phone candidate (default
  nesting suppression drops strictly nested spans, not identical spans);
  and `943 476 5918` in `NHS 943 476 5918` (`operational-memos.txt`),
  where the mistyped NHS check digit is rejected by `uk-nhs`, leaving
  ten grouped digits.
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
- **`ca-sin` and `imei` misses (1 each)**: `SIN (ON) 046 454 286` and
  `IMEI (primary) 001970012345677` in `operational-memos.txt` place a
  parenthesised qualifier between the keyword label and the digits,
  outside the `:`, `=`, `#`, `-` and whitespace separators the labelled
  scanners skip.

## Parity checks

Besides the goldens, a scratch differential (a JVM oracle over the same
OpenNLP commit and a Rust one, kept out of the tree) compared every pack
and the default cursor extractor on more than 500,000 generated texts, and the
masking, pseudonymizing, keyed labels, offset maps and audit reports on
20,000, with no difference. A second one ran the generator's
`PiiRewriteGolden` over 20,000 texts (a quarter from the fuzz corpus,
the rest values of the `pii` set run together, cut and mixed with
non-ASCII text) with every pack, and compared its 200,000 goldens with
the masked view, the rewrites, the labels and the audit report: no
difference over 955,007 masked characters, 102,992 labels, 4,562
refused rewrites and 60,496 audited mentions.
