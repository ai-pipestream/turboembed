# Extracted PDF artifact evaluation

This small, offline evaluation uses pages 2 and 3 of two ACL papers. Both
page pairs retain the complete `pdftotext -layout` output, including columns,
line wrapping, page numbers, form feeds and font mapping damage. It contains
21,819 UTF-8 bytes and 2,687 whitespace-delimited tokens. No damage was injected.
The PDFs are fetched only by the optional reproduction script.

## Sources and licence

The text excerpts remain under **CC-BY-4.0**, not the repository's code licence.
See the [ACL copyright policy](https://aclanthology.org/faq/copyright/) for ACL
materials from 2016 onward and the [licence](https://creativecommons.org/licenses/by/4.0/).
Attribution:

- Sunil Kumar Sahu, Fenia Christopoulou, Makoto Miwa and Sophia Ananiadou.
  2019. *Inter-sentence Relation Extraction with Document-level Graph
  Convolutional Neural Network*. ACL, pages 4309-4316.
  [Paper and PDF](https://aclanthology.org/P19-1423/), doi:10.18653/v1/P19-1423.
  Excerpt: PDF pages 2-3, printed pages 4310-4311.
- Lieke Gelderloos, Grzegorz Chrupała and Afra Alishahi. 2020. *Learning to
  Understand Child-directed and Adult-directed Speech*. ACL, pages 1-6.
  [Paper and PDF](https://aclanthology.org/2020.acl-main.1/), doi:10.18653/v1/2020.acl-main.1.
  Excerpt: PDF pages 2-3, printed pages 2-3.

Changes to the publications: text extraction, page selection and reversible
escaping only. Labels are separate annotations; no endorsement is implied.
The code and annotations use the repository licence.

## Reproduction

Poppler `pdftotext` **26.07.0**, options `-layout -f 2 -l 3 -enc UTF-8`.
The script pins PDF and extracted-text SHA-256, requires this extractor
version and checks the checked-in escaped text byte for byte. It never
generates labels or updates an expected result. From the repository root:

```sh
python3 analysis/normalize/data/extracted-pdf/reproduce.py target/pdf-reproduction
cargo test --locked -p turbo-normalize --test extracted_pdf -- --nocapture
```

| Document | PDF SHA-256 | Extracted UTF-8 SHA-256 |
| --- | --- | --- |
| P19-1423 | f9ee341ae788143c48a5d424812ebd40c8f1312b7eaffcbd52c0aa3fdb79f63c | 4a4c955ebcaac4f100a3eee0d9755b0febaa72394695e172f422f60715956f1c |
| 2020.acl-main.1 | bd2680310de89488ca086aa8378ed095ecf5e4b56588812f62b11667de66d70e | 61d8ea51cba932c21984731f870eb08efa34e280b91d20c81487577a5d8c86c8 |

Each `.escaped` file is one field in `turbo_goldens::escape` format followed
by a newline. Backslashes, C0 controls and U+2014 are escaped; decoding yields
the exact extracted bytes above. Test hashes guard offsets against drift.

## Annotation policy

The gold set was selected by inspecting the extraction before running the
detector. Inspection of the rendered third page of P19-1423 confirms that
U+0010/U+0011 and U+F8EB/U+F8ED/U+F8F6/U+F8F8 represent mathematical parentheses
in the printed equations, not intentional control text or portable Unicode.
There are eleven maximal control runs and four private-use runs. The second
paper contributes a negative sample with no gold artifact spans. Labels use
half-open UTF-8 byte offsets into the decoded page pair, with adjacent damage
of the same kind forming one span.

All other positions are negatives **for the detector's eleven kinds**. This
does not label the extraction as perfect: interleaved columns, line-break
hyphenation, flattened equations and missing URL underscores are outside this
taxonomy. They are not silently counted as detector successes. Neither page
pair has damaged ASCII word cores or encoded binary payloads for NoiseScorer;
that auxiliary evaluation is negative-only. Acronyms, scientific terms and
source spelling mistakes are legitimate text, not extraction damage.

Exact kind and both boundaries must match for a true positive. Unmatched
predictions are false positives and unmatched labels false negatives. Precision
is TP/(TP+FP), recall TP/(TP+FN); a zero denominator is **N/A**, not 1. The
optional all-kinds configuration is reported separately from the default.

This is a four-page convenience sample, labelled in one pass, not a held-out
representative benchmark or an independently double-annotated corpus. Most
kinds have no positive support. In particular, an unpaired surrogate cannot
occur in Rust UTF-8 text. Do not tune thresholds from these pages or generalize
the two supported kinds to OCR, other scripts, encodings or PDF producers.
The metrics and limitations are in `docs/normalizers.md`.
