# Extracted PDF artifact evaluation

This offline evaluation retains eleven complete extracted pages from four
CC-BY-4.0 papers: 50,994 UTF-8 bytes and 6,555 whitespace-delimited tokens.
No damage was injected and no source text was repaired. The gold labels
were selected from the extraction and rendered source pages before running
the detector on the new samples. Labels remain separate from predictions.

## Sources and licence

The excerpts remain under **CC-BY-4.0**, not the repository's code licence.
The code and annotations use the repository licence. Retain these source
attributions and transformation notices when redistributing the data:

- Sunil Kumar Sahu, Fenia Christopoulou, Makoto Miwa and Sophia Ananiadou.
  2019. *Inter-sentence Relation Extraction with Document-level Graph
  Convolutional Neural Network*. ACL, pages 4309-4316.
  [Paper and PDF](https://aclanthology.org/P19-1423/), doi:10.18653/v1/P19-1423.
  PDF pages 2-3, printed pages 4310-4311; Poppler extraction.
- Lieke Gelderloos, Grzegorz Chrupała and Afra Alishahi. 2020. *Learning to
  Understand Child-directed and Adult-directed Speech*. ACL, pages 1-6.
  [Paper and PDF](https://aclanthology.org/2020.acl-main.1/), doi:10.18653/v1/2020.acl-main.1.
  PDF pages 2-3, printed pages 2-3; Poppler extraction.
- Zhuoran Li, Chunming Hu, Xiaohui Guo, Junfan Chen, Wenyi Qin and Richong
  Zhang. 2022. *An Unsupervised Multiple-Task and Multiple-Teacher Model
  for Cross-lingual Named Entity Recognition*. ACL, pages 170-179.
  [Paper and PDF](https://aclanthology.org/2022.acl-long.14/),
  doi:10.18653/v1/2022.acl-long.14. PDF page 5, printed page 174; MuPDF
  extraction. The summation symbol in the training-loss equation becomes
  one replacement character in this extractor's output.
- Justin Bunker, Mark Girolami, Hefin Lambley, Andrew M. Stuart and
  T. J. Sullivan. 2025. *Autoencoders in Function Space*. Journal of
  Machine Learning Research 26(165):1-54.
  [Original publication](https://jmlr.org/papers/v26/25-0035.html).
  PDF pages 2-7; Poppler extraction. Its first page explicitly states
  CC-BY 4.0 and requires attribution linking to the original publication.

ACL materials from 2016 onward use CC-BY-4.0 under the
[ACL copyright policy](https://aclanthology.org/faq/copyright/).
[Licence text](https://creativecommons.org/licenses/by/4.0/).
Changes to every publication are page selection, text extraction and
reversible escaping only; no author endorsement is implied.

## Reproduction

The Poppler samples require `pdftotext` **26.07.0**, with
`-layout -enc UTF-8 -f FIRST -l LAST`. The replacement-character sample
requires MuPDF `mutool` **1.27.0**, with `draw -q -F txt` and page 5.
These are two real extraction pipelines; their outputs are not pooled as
if extractor identity were interchangeable.

From the repository root, with `curl`, `pdftotext` and `mutool` on PATH:

```sh
cargo run --locked -p turbo-normalize --example reproduce_extracted_pdf -- target/pdf-reproduction
cargo test --locked -p turbo-normalize --test extracted_pdf -- --nocapture
```

The Rust helper requires the pinned extractor versions, checks each PDF
and extracted-text SHA-256, and compares the decoded fixture byte for byte.
It never writes labels or checked-in fixtures. Existing cached PDFs must
match their hash before extraction; missing PDFs are fetched from the
publisher URLs. There is no Python helper or added Rust dependency.

| Fixture | PDF SHA-256 | Extracted UTF-8 SHA-256 |
| --- | --- | --- |
| P19-1423 | f9ee341ae788143c48a5d424812ebd40c8f1312b7eaffcbd52c0aa3fdb79f63c | 4a4c955ebcaac4f100a3eee0d9755b0febaa72394695e172f422f60715956f1c |
| 2020.acl-main.1 | bd2680310de89488ca086aa8378ed095ecf5e4b56588812f62b11667de66d70e | 61d8ea51cba932c21984731f870eb08efa34e280b91d20c81487577a5d8c86c8 |
| 2022.acl-long.14-mupdf | b88c8b84ca06bb72965981c29f2c04236c9f8006c9b89cfe7d653134fb1a1b88 | ff84db26480ba60169e7e8c8a1803d2a297de852638a5a8f79609e540230c8b6 |
| jmlr-25-0035 | e5b2e098ca661463f69e4328bb74ad6bf1a0402d4eec757ebcafabc747d70936 | f902f5558cea9702dd1184e240f71109605b9fe9c1372be649e64c5d45becd83 |

Each `.escaped` file is one `turbo_goldens::escape` field followed by a
newline. Backslashes, C0 controls and U+2014 are escaped. Decoding yields
the exact extracted bytes above; tests hash the decoded text before
reading byte offsets. The fixture id includes `mupdf` for the second
extraction pipeline.

## Annotation policy and limits

Labels are maximal adjacent runs of one kind with half-open UTF-8 byte
offsets. The original ACL samples contain eleven control runs and four
private-use runs in mathematical equations; the second paper is a negative
sample. Rendered ACL page 174 confirms that the new MuPDF U+FFFD replaces
a printed summation sign. The JMLR extract adds fifteen control runs in
mathematical expressions and six running headers, three of each alternating
header. Bare page numbers are not labelled as repeated-line artifacts.

The three occurrences of `u∼Υ z∼Qθ` in the JMLR extract are mathematical
notation, not running headers or damaged text. They stay negative gold
positions even though the optional repeated-line rule reports them. Its
measured precision is therefore 6/9, not 1.0. No labels were changed to
fit those predictions and no production detector options were tuned.

All other positions are negatives for the detector's eleven kinds.
Column interleaving, line-break hyphenation, flattened equations and
missing URL underscores lie outside this taxonomy. Exact kind and both
boundaries must match; unmatched predictions are false positives and
unmatched labels are false negatives. A zero denominator is **N/A**.
Default and all-kinds results are separate; default disables RepeatedLine
and misses its six gold spans. NoiseScorer's negative-only evaluation
continues to cover just the two original ACL samples.

Four kinds now have positive support: ReplacementChar, Control, PrivateUse
and RepeatedLine. **Mojibake and ZeroWidth still have no positive support.**
Other absent kinds remain unmeasured, and an unpaired surrogate cannot
occur in Rust UTF-8 text. This convenience sample was labelled in one
pass; it is not independently double-annotated or representative of PDF
and OCR quality. Further licensed damaged extracts and independent
annotation review are required to complete issue #915's coverage work.
