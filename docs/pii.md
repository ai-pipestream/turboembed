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
kind and normalized form) appear once, the first found kept. Mentions of
different kinds may share a span or overlap: an IMEI after its label is
also a Luhn-valid card number, and the trailing groups of a grouped IBAN
can be a phone number. `Pii::check` holds a layer to this order.

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
tokens in place of values.

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
- Ten digits grouped `ddd ddd dddd` that pass the NHS mod 11 check are
  both an NHS number and a phone number in the wide pack, including in a
  US phone number whose digits pass by chance.

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

Make them in the recipe container with

```
cargo run -p turbo-goldens-tool -- generate
```

`analysis/pii/tests/goldens.rs` runs every pack over the set and compares
the layer with the goldens row for row; while the `pii` golden directory
is not on disk it says so and passes. The differential fuzzer covers the
ten packs (`analysis/fuzz/src/runs/pii.rs`), and the `pii` set is among
the sets it writes generated texts into.

## Precision and recall

`analysis/pii/tests/precision_recall.rs` measures the widest pack on a
labelled set made from a fixed seed (40 values of every kind built to
pass their checks, each in a sentence, and near misses of the
checksummed kinds with no label) together with OpenNLP's golden true and
false positive corpora: 1112 texts, 720 labels made in the test. A found
mention is true when a label has its span and kind; every other found
mention is false, values found inside other values included.

| | true | false | missed | precision | recall |
|---|---|---|---|---|---|
| all kinds | 745 | 88 | 0 | 0.894 | 1.000 |
| `phone` | 123 | 80 | 0 | 0.606 | 1.000 |
| `card` | 42 | 4 | 0 | 0.913 | 1.000 |
| `uk-nhs` | 41 | 4 | 0 | 0.911 | 1.000 |
| every other kind | 539 | 0 | 0 | 1.000 | 1.000 |

The false mentions are a phone in each near miss NHS number (ten grouped
digits are a phone shape), phones and cards inside grouped IBANs, and NHS
numbers inside US phone numbers whose digits pass mod 11. The set is made
to pass the checks the scanners make, so recall on it says the scanners
find what they are built to find, not how they do on text written by
people. `PII_SHOW_FALSE=1` lists the false mentions.

## Parity checks

Besides the goldens, a scratch differential (a JVM oracle over the same
OpenNLP commit and a Rust one, kept out of the tree) compared every pack
and the default cursor extractor on more than 500,000 generated texts, and the
masking, pseudonymizing, keyed labels, offset maps and audit reports on
20,000, with no difference.
