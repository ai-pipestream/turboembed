# Reference implementation inventory (#392)

Read-only audit of the build host reference collection on 2026-10-22.
All 28 tier-1 entries named in #392 are present, and their checked-out
commits match the host inventory. This is a code-reference inventory,
not approval to redistribute every fixture, model or dataset in those
repositories. Preserve upstream copyright and licence notices when
porting permitted material.

## Checked entries

The licence links pin the files inspected at each recorded revision.
Apache projects may include third-party notices and per-file exceptions;
check those before copying a particular file. The host inventory says
every clone is shallow, but Git reports full history for Lucene and
OpenNLP. The table records the observed state.

| Entry | Revision | Licence evidence | Shallow |
|---|---|---|---|
| [spaCy](https://github.com/explosion/spaCy) | `c2dabfce56ad2991685ec85783cd59637a5d7b8f` | [MIT](https://github.com/explosion/spaCy/blob/c2dabfce56ad2991685ec85783cd59637a5d7b8f/LICENSE) | yes |
| [spacy-lookups-data](https://github.com/explosion/spacy-lookups-data) | `180c5da7bead0535aee0b493910f8520bac7d966` | [MIT](https://github.com/explosion/spacy-lookups-data/blob/180c5da7bead0535aee0b493910f8520bac7d966/LICENSE) | yes |
| [stanza](https://github.com/stanfordnlp/stanza) | `c9a2158a87502dd63f757ab1aa933230b0acbce7` | [Apache-2.0](https://github.com/stanfordnlp/stanza/blob/c9a2158a87502dd63f757ab1aa933230b0acbce7/LICENSE) | yes |
| [nltk](https://github.com/nltk/nltk) | `501f2f36419d47ed21bb9f10c5ee8a41c994183c` | [Apache-2.0](https://github.com/nltk/nltk/blob/501f2f36419d47ed21bb9f10c5ee8a41c994183c/LICENSE.txt) | yes |
| [lucene](https://github.com/apache/lucene) | `4965e8d4d960445a0522fae512c60c6d8f11fc29` | [Apache-2.0](https://github.com/apache/lucene/blob/4965e8d4d960445a0522fae512c60c6d8f11fc29/LICENSE.txt) | **no** |
| [icu](https://github.com/unicode-org/icu) | `ad484f3d6c036ba06193ac7c3bc2233a53ab68b8` | [Unicode-3.0](https://github.com/unicode-org/icu/blob/ad484f3d6c036ba06193ac7c3bc2233a53ab68b8/LICENSE) | yes |
| [icu4x](https://github.com/unicode-org/icu4x) | `51f138436f50f8110f854252f9a506ccfe06ef1e` | [Unicode-3.0](https://github.com/unicode-org/icu4x/blob/51f138436f50f8110f854252f9a506ccfe06ef1e/LICENSE) | yes |
| [cldr-json](https://github.com/unicode-org/cldr-json) | `91c267402229a59e3ef2774544f001bf959e8809` | [Unicode-3.0](https://github.com/unicode-org/cldr-json/blob/91c267402229a59e3ef2774544f001bf959e8809/LICENSE) | yes |
| [tokenizers-v0.22](https://github.com/huggingface/tokenizers) | `f383101a26663708484cac0727792aad74f78234` | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/LICENSE) | yes |
| [tokenizers-v0.23](https://github.com/huggingface/tokenizers) | `301277072263fb3830eb66eec7e47adadf76f1dc` | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/301277072263fb3830eb66eec7e47adadf76f1dc/LICENSE) | yes |
| [sentencepiece](https://github.com/google/sentencepiece) | `e0cce7d37b065b5140349dbe12c6bcf6192fdd78` | [Apache-2.0](https://github.com/google/sentencepiece/blob/e0cce7d37b065b5140349dbe12c6bcf6192fdd78/LICENSE) | yes |
| [BlingFire](https://github.com/microsoft/BlingFire) | `18e9a19e586095fb60d629fa850fb610d5bca605` | [MIT](https://github.com/microsoft/BlingFire/blob/18e9a19e586095fb60d629fa850fb610d5bca605/LICENSE) | yes |
| [syntok](https://github.com/fnl/syntok) | `371e6ca0d4e307271eaf439f912fad94efdaa8e9` | [MIT](https://github.com/fnl/syntok/blob/371e6ca0d4e307271eaf439f912fad94efdaa8e9/LICENSE) | yes |
| [pySBD](https://github.com/nipunsadvilkar/pySBD) | `5905f13be4fc95f407b98392e0ec303617a33d86` | [MIT](https://github.com/nipunsadvilkar/pySBD/blob/5905f13be4fc95f407b98392e0ec303617a33d86/LICENSE) | yes |
| [pragmatic_segmenter](https://github.com/diasks2/pragmatic_segmenter) | `358fe97e62902cd86fa6b85063c3bf91aa6c156a` | [MIT](https://github.com/diasks2/pragmatic_segmenter/blob/358fe97e62902cd86fa6b85063c3bf91aa6c156a/LICENSE.txt) | yes |
| [sentencex](https://github.com/santhoshtr/sentencex) | `e4aaeb8346118e837139958dac8ae8390d7a8f12` | [MIT](https://github.com/santhoshtr/sentencex/blob/e4aaeb8346118e837139958dac8ae8390d7a8f12/LICENSE) | yes |
| [wtpsplit](https://github.com/segment-any-text/wtpsplit) | `3ef928c8e163d11dc6f6336ad610bff2ffe0bda4` | [MIT](https://github.com/segment-any-text/wtpsplit/blob/3ef928c8e163d11dc6f6336ad610bff2ffe0bda4/LICENSE) | yes |
| [cld3](https://github.com/google/cld3) | `b48dc46512566f5a2d41118c8c1116c4f96dc661` | [Apache-2.0](https://github.com/google/cld3/blob/b48dc46512566f5a2d41118c8c1116c4f96dc661/LICENSE) | yes |
| [GlotLID](https://github.com/cisnlp/GlotLID) | `d63cc2f6db97de90b6877ccbe0253161f08fdb14` | [Apache-2.0 plus training-data notice; code only](https://github.com/cisnlp/GlotLID/blob/d63cc2f6db97de90b6877ccbe0253161f08fdb14/LICENSE) | yes |
| [fastText](https://github.com/facebookresearch/fastText) | `1142dc4c4ecbc19cc16eee5cdd28472e689267e6` | [MIT](https://github.com/facebookresearch/fastText/blob/1142dc4c4ecbc19cc16eee5cdd28472e689267e6/LICENSE) | yes |
| [duckling](https://github.com/facebook/duckling) | `59a13ff87b1aa8be6b93d387244f8636b26185c5` | [BSD-3-Clause](https://github.com/facebook/duckling/blob/59a13ff87b1aa8be6b93d387244f8636b26185c5/LICENSE) | yes |
| [Recognizers-Text](https://github.com/microsoft/Recognizers-Text) | `da7edcff59f669b2a460ab9d400e36298f0d658e` | [MIT](https://github.com/microsoft/Recognizers-Text/blob/da7edcff59f669b2a460ab9d400e36298f0d658e/LICENSE) | yes |
| [presidio](https://github.com/microsoft/presidio) | `4f0e236e31a57767958a3eec2de3fc58dffc3034` | [MIT](https://github.com/microsoft/presidio/blob/4f0e236e31a57767958a3eec2de3fc58dffc3034/LICENSE) | yes |
| [python-ftfy](https://github.com/rspeer/python-ftfy) | `74dd0452b48286a3770013b3a02755313bd5575e` | [Apache-2.0](https://github.com/rspeer/python-ftfy/blob/74dd0452b48286a3770013b3a02755313bd5575e/LICENSE.txt) | yes |
| [twitter-text](https://github.com/twitter/twitter-text) | `30e2430d90cff3b46393ea54caf511441983c260` | [Apache-2.0](https://github.com/twitter/twitter-text/blob/30e2430d90cff3b46393ea54caf511441983c260/LICENSE) | yes |
| [Sudachi](https://github.com/WorksApplications/Sudachi) | `434f31d8af6ef99eee97c57317e5f097dfdfe561` | [Apache-2.0](https://github.com/WorksApplications/Sudachi/blob/434f31d8af6ef99eee97c57317e5f097dfdfe561/LICENSE-2.0.txt) | yes |
| [dkpro-core](https://github.com/dkpro/dkpro-core) | `ebc0bb37d03dea94b5d90119d9e435c84f1b1caa` | [Apache-2.0 in 105 remaining module licence files](https://github.com/dkpro/dkpro-core/blob/ebc0bb37d03dea94b5d90119d9e435c84f1b1caa/dkpro-core-api-anomaly-asl/LICENSE.txt) | yes |
| [opennlp](https://github.com/apache/opennlp) | `b453b9ee3f2856a1dbf323ad13c8465a1203d3f0` | [Apache-2.0](https://github.com/apache/opennlp/blob/b453b9ee3f2856a1dbf323ad13c8465a1203d3f0/LICENSE) | **no** |

## Checkout limits checked

- ICU uses sparse paths `icu4c/source/data/brkitr`,
  `icu4c/source/data/misc` and `icu4c/source/data/translit`.
- CLDR JSON uses `cldr-json/cldr-core` and `cldr-json/cldr-segments-full`.
- Tokenizers has separate v0.22.2 and v0.23.3 entries; do not substitute
  a newer main branch without checking offset behaviour.
- DKPro has none of the four excluded GPL module directories:
  `dkpro-core-corenlp-gpl`, `dkpro-core-gpl`, `dkpro-core-io-tgrep-gpl`
  and `dkpro-core-sfst-gpl`. The 105 remaining module `LICENSE.txt`
  files contain Apache-2.0 terms (two distinct licence-file hashes).
  The root licence delegates to those module files.
- GlotLID has no files ending in `.bin`, `.safetensors`, `.onnx` or
  `.pt` in this checkout. Its licence explicitly disclaims assurance
  about training-data rights. Code approval does not approve weights.
- No top-level directory exactly matches a name on the issue's
  do-not-read list, compared without case. This check inspected names
  only, not excluded project contents; it does not prove that no
  dependency or nested file includes material from those projects.

## Remaining work before closing #392

1. Correct the host inventory's shallow-clone claim for Lucene and
   OpenNLP, or have the collection owner replace those checkouts with
   shallow copies while preserving any local work. This audit changed
   no reference checkout or host inventory file.
2. Recover `beyond-opennlp/rust-crates.md` and
   `beyond-opennlp/other-languages.md`. Neither survey is available at
   the location supplied in #392. The explicit tier-1 list above is
   covered, but the Rust "copy candidates" cannot be checked for
   completeness without that source list. Do not infer that every
   installed Rust crate was an approved candidate.
3. Before copying data or model assets from a reference, inspect the
   asset-specific terms. This pass checked source licence files and
   checkout metadata, not every nested asset in 28 repositories.

The do-not-read list remains: CoreNLP, HeidelTime, YAKE, pke, gensim,
GATE, LanguageTool, Moses, KenLM, polyglot, KoNLPy, Elasticsearch,
Maverick, ReLiK and REBEL. Do not clone or read those implementations.
Do not add models or data under CC BY-SA, LGPL or non-commercial terms
to this reference collection. See [licences](licences.md) for the
separate policy governing product datasets and checkpoints.
