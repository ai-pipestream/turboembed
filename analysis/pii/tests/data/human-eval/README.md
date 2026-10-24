# Human-written PII evaluation set

Eight documents (`45,401` UTF-8 bytes) and their span-level annotations
(`labels.tsv`, `87` gold mentions across all 20 built-in `PiiKind`
variants), scored by `precision_and_recall_on_human_written_text` in
`analysis/pii/tests/precision_recall.rs` (`docs/pii.md`,
`docs/licences.md`).

The evaluation set is a mixed corpus:
- Five third-party MASC 3.0.0 documents (`34,969` bytes, `12` gold spans,
  all `phone`, across `masc-att2`, `masc-hsus4`, `masc-united1`, and
  `masc-nyt6`, plus `masc-lsc-access` with `0` gold spans as a statutory
  and CFR citation false-positive stress document).
- Two project-adapted specification summaries (`4,360` bytes, `18` gold
  spans: `10` `eth-address` in `eip-55.txt` and `8` `btc-address` in
  `bip-173-350.txt`) embedding the canonical test vectors from ERC-55
  (CC0-1.0) and BIP-173 / BIP-350 (BSD-2-Clause).
- One project-authored operational memo collection (`operational-memos.txt`,
  `6,072` bytes, `57` gold spans: `3` `phone` spans and every gold
  mention of the remaining 17 built-in kinds).

## Documents and provenance

| ID | File | Bytes | UTF-8 SHA-256 | Source and licence |
|---|---|---:|---|---|
| `masc-att2` | `masc-att2.escaped` | 2,980 | `f8f2cfd3ca288dcb9728092365b5c6d315de7b8b399f9f34bfc1da15302750ee` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/att2.txt` (AT&T solicitation letter, CC BY 3.0 US). |
| `masc-hsus4` | `masc-hsus4.escaped` | 6,933 | `88f06a9dc3c1338f01959b2599c0a3b6b10058c77a1c5e5811c5ff9f0aa0d700` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/hsus4.txt` (Humane Society solicitation letter and state charity-registration notices, CC BY 3.0 US). |
| `masc-united1` | `masc-united1.escaped` | 2,279 | `f9f0a966371aadd4ea29166109700d0ff77cc9e6d2899da90b204d945756ffaf` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/united1.txt` (United Airlines Mileage Plus letter, CC BY 3.0 US). |
| `masc-nyt6` | `masc-nyt6.escaped` | 9,095 | `d380182bfb346fb3e73bcf2189964dbab1adf1b269abe5b4c77ff0befba3bb2f` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/newspaper/nyt/NYTnewswire6.txt` (New York Times News Service travel dispatch, CC BY 3.0 US). |
| `masc-lsc-access` | `masc-lsc-access.escaped` | 13,682 | `ec16a1e1e8d0deca3db0d6e25800e815e518441c29b7187b4bed6fd2773275f3` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/govt-docs/LSC-Protocol_Regarding_Access.txt` (Legal Services Corporation protocol with CFR and statutory citations, CC BY 3.0 US). |
| `eip-55` | `eip-55.txt` | 2,164 | `99d519e93281cb8634d25262f7bd230f9deb68f5d54e3cf0e990acecc27ca666` | Project-adapted summary (Apache-2.0) embedding the canonical test vectors from [ERC-55](https://eips.ethereum.org/EIPS/eip-55) by Vitalik Buterin and Alex Van de Sande (2016-01-14, CC0-1.0), plus invalid mixed-case checksum near misses and the detector's all-zero burn-address exclusion. |
| `bip-173-350` | `bip-173-350.txt` | 2,196 | `1aebb37a80e7c451b85ee5c611c4267964d7d7c1d77f3f5bfaa7ae92c2b2359a` | Project-adapted summary (Apache-2.0) embedding the canonical test vectors from [BIP-173](https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki) and [BIP-350](https://github.com/bitcoin/bips/blob/master/bip-0350.mediawiki) by Pieter Wuille and Greg Maxwell (BSD-2-Clause), plus synthetic Base58Check vectors and invalid Bech32/Bech32m near misses. |
| `operational-memos` | `operational-memos.txt` | 6,072 | `aa4d465ba9de775c575090984359a19f50308c2b88e5634a87cde0f803d907a9` | Project-authored incident, treasury, network, tax/payroll, clinical and mobile fleet memos (Apache-2.0) using documentation, test, and synthetic checksum-valid specimen identifiers (RFC 2606, RFC 5737, RFC 3849, RFC 7042, RFC 7519, AWS documentation example keys, ISO 13616 / Bundesbank specimen IBANs, Federal Reserve routing numbers, NHS test numbers, ISO 7064 test Steuer-IDs, SIN `046 454 286`, and Luhn-valid test IMEIs). |

The five MASC documents are stored in `turbo_goldens::escape` encoding
(`.escaped`, identical to their files in `analysis/goldens/data/inputs/masc/`)
because the original MASC files use CRLF line endings; `precision_recall.rs`
unescapes them before hashing and scoring so byte offsets match the exact
MASC 3.0.0 bytes.

## Annotation format

`labels.tsv` has five tab-separated columns:

```text
doc_id	start_byte	end_byte	kind	covered_text
```

Offsets are UTF-8 byte indices into the unescaped document text. Each gold
mention represents an actual occurrence of that `PiiKind` in the prose
(including formatting variants that the current scanners do not yet match,
such as dot-separated US telephone numbers `973.504.6215` and parenthesised
qualifier labels `SIN (ON) 046 454 286`), while secondary overlapping kinds
produced only as side effects of another kind (such as an `email` span inside
`https://user:pass@host/path`, a `card` match on a 15-digit IMEI, or a `phone`
match on a German Steuer-ID) are not in `labels.tsv` and are counted as false
positives.
