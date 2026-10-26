# Mixed PII evaluation set (partial work for #968)

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

This corpus does not complete #968's requirement for licensed human-written
text with labelled spans per kind. Within the mixed set only `phone` has
unedited third-party prose coverage; the separate external partition below
adds unedited third-party coverage for `email`, `iban`, `ipv4`, `ipv6`,
`jwt`, `mac` and further `phone`. The adapted crypto summaries and authored
memos remain useful, separately labelled test partitions; they do not
establish external human-written prose coverage for their kinds. The test
function's historical name does not change that scope.

Outstanding coverage on external text is listed in `docs/pii.md` under
"Outstanding work for #968". New sources must preserve the policy for safe specimen
values, record provenance and redistribution terms, and pin the actual source
text and gold byte offsets. The original issue requirements remain open;
neither source-partition labels nor an agent's scope decision waive them.

## Documents and provenance

| ID | File | Bytes | UTF-8 SHA-256 | Source and licence |
|---|---|---:|---|---|
| `masc-att2` | `masc-att2.escaped` | 2,980 | `f8f2cfd3ca288dcb9728092365b5c6d315de7b8b399f9f34bfc1da15302750ee` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/att2.txt` (AT&T solicitation letter, CC BY 3.0 US). |
| `masc-hsus4` | `masc-hsus4.escaped` | 6,933 | `88f06a9dc3c1338f01959b2599c0a3b6b10058c77a1c5e5811c5ff9f0aa0d700` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/hsus4.txt` (Humane Society solicitation letter and state charity-registration notices, CC BY 3.0 US). |
| `masc-united1` | `masc-united1.escaped` | 2,279 | `f9f0a966371aadd4ea29166109700d0ff77cc9e6d2899da90b204d945756ffaf` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/letters/solicitation-brochures/united1.txt` (United Airlines Mileage Plus letter, CC BY 3.0 US). |
| `masc-nyt6` | `masc-nyt6.escaped` | 9,095 | `d380182bfb346fb3e73bcf2189964dbab1adf1b269abe5b4c77ff0befba3bb2f` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/newspaper/nyt/NYTnewswire6.txt` (New York Times News Service travel dispatch, CC BY 3.0 US). |
| `masc-lsc-access` | `masc-lsc-access.escaped` | 13,682 | `ec16a1e1e8d0deca3db0d6e25800e815e518441c29b7187b4bed6fd2773275f3` | [MASC 3.0.0](https://anc.org/data/masc/) `data/written/govt-docs/LSC-Protocol_Regarding_Access.txt` (Legal Services Corporation protocol with CFR and statutory citations, CC BY 3.0 US). |
| `eip-55` | `eip-55.txt` | 2,164 | `5359191d91cf2c6e707bc4a790ef90155da0499e788b386805ba10022ef6b848` | Project-adapted summary (Apache-2.0) embedding the canonical test vectors from [ERC-55](https://eips.ethereum.org/EIPS/eip-55) by Vitalik Buterin and Alex Van de Sande (2016-01-14, CC0-1.0), plus invalid mixed-case checksum near misses and the detector's all-zero burn-address exclusion. |
| `bip-173-350` | `bip-173-350.txt` | 2,196 | `a8d1e9b7621ff57df9a51e67fe636cd08cd7bd14ed1ef7014473a0402a9d14ae` | Project-adapted summary (Apache-2.0) embedding the canonical test vectors from [BIP-173](https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki) and [BIP-350](https://github.com/bitcoin/bips/blob/master/bip-0350.mediawiki) by Pieter Wuille and Greg Maxwell (BSD-2-Clause), plus synthetic Base58Check vectors and invalid Bech32/Bech32m near misses. |
| `operational-memos` | `operational-memos.txt` | 6,072 | `260fc461b579030279741b25a1b1c7f03c4fa5c3929fb5585bf12ee38dc3fe9c` | Project-authored incident, treasury, network, tax/payroll, clinical and mobile fleet memos (Apache-2.0) using documentation, test, and synthetic checksum-valid specimen identifiers (RFC 2606, RFC 5737, RFC 3849, RFC 7042, RFC 7519, AWS documentation example keys, ISO 13616 / Bundesbank specimen IBANs, Federal Reserve routing numbers, NHS test numbers, ISO 7064 test Steuer-IDs, SIN `046 454 286`, and Luhn-valid test IMEIs). |

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

## External prose partition (#968)

Five further whole third-party documents (`213,306` UTF-8 bytes, `69` gold
mentions in `labels-external.tsv`, same five columns except that newlines
inside covered text are escaped as `\n`), scored separately by
`precision_and_recall_on_external_prose` so this partition never pools with
the mixed set above. The documents are committed byte for byte as retrieved
(no escaping, no edits); the SHA-256 of each file is pinned in the test.

| ID | File | Bytes | UTF-8 SHA-256 | Source and licence |
|---|---|---:|---|---|
| `rfc7519` | `rfc7519.txt` | 63,039 | `fecd930e9ccf2276b95c0017c6c4ff5d09352e4bc3c7629946447894e0f97248` | [RFC 7519](https://www.rfc-editor.org/rfc/rfc7519.txt), JSON Web Token (JWT), IETF Trust, May 2015 (BCP 78; RFC text may be copied and furnished). Gold: `7` `email`, `2` `jwt` (both complete example JWTs, line-wrapped for display in the RFC, labelled over their whole wrapped spans). Encoded JOSE header and payload parts printed on their own are not JWTs and are not labelled; the Appendix A encrypted and nested examples are JWE compact serializations and are not labelled `jwt`. |
| `rfc5737` | `rfc5737.txt` | 7,036 | `9d16217614a74a9b064eff900e5cd07a525793cf8227dca1267100e880f0c410` | [RFC 5737](https://www.rfc-editor.org/rfc/rfc5737.txt), IPv4 Address Blocks Reserved for Documentation, IETF Trust, January 2010. Gold: `8` `ipv4` (the address part of each documentation prefix; the `/24` or `/16` length suffix is outside the span), `3` `email`, `2` `phone` (authors' address blocks). |
| `rfc3849` | `rfc3849.txt` | 7,872 | `b0741fc609be565b356c62d2f9b9ef30469f3fe7bde386c2690e173751bb1941` | [RFC 3849](https://www.rfc-editor.org/rfc/rfc3849.txt), IPv6 Address Prefix Reserved for Documentation, IETF Trust, July 2004. Gold: `3` `ipv6` (each `2001:DB8::` address part of the documentation prefix), `4` `email`. |
| `rfc7042` | `rfc7042.txt` | 53,488 | `8ad7b042253d1204cef2a5eddd6682072a5d4f9edc3b749df15df7bccdeca1df` | [RFC 7042](https://www.rfc-editor.org/rfc/rfc7042.txt), IANA Considerations and IETF Protocol and Documentation Usage for IEEE 802 Parameters, IETF Trust, October 2013. Gold: `29` `mac` (whole EUI-48 dash forms only, including range endpoints; OUIs, EUI-64 identifiers, and wildcard patterns such as `02-00-5E-FE-xx-xx-xx-xx` are not 48-bit MAC occurrences and are not labelled), `4` `email`, `3` `phone` (authors' addresses and the example application's telephone field). |
| `wikipedia-iban` | `wikipedia-iban.wikitext` | 81,871 | `bdb116257bb8d25897a11b61de009167c673856ce0458183d83d336d73d8e7b1` | Wikipedia article "International Bank Account Number", wikitext at revision `1376647818` (2026-09-25), CC BY-SA 4.0, retrieved through the MediaWiki action API with the revision pinned. Gold: `4` `iban` (the complete Irish and Burundi specimen IBANs in human-readable and machine-readable form). The worked British example is split across colour templates in the wikitext and is not one contiguous occurrence; format patterns in the country table are not IBAN occurrences. |

Annotation method: every occurrence of every built-in kind in these five
documents was enumerated by independent shape searches over the pinned text
and read in context before labelling; spans the search shapes could not
produce (the display-wrapped JWTs, address parts of CIDR prefixes) were
located from the documents' own example sections. Values are as their
sources publish them: documentation address blocks, specimen IBANs, the
RFC example JWT, and the RFC authors' public contact addresses. Nothing was
rewritten, so no replacement transformation needs recording. Detector output
played no part in choosing gold spans.
