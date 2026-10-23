# Rust reference survey

This replacement survey is grounded in repository issues rather than an
assumed copy of the missing external file. The issue index below records
208 distinct crates.io names from issue bodies available on 2026-10-23.
The selected references have freshly checked registry metadata and, where
available, a pinned upstream source licence. A package mention is evidence
of consideration, not approval to clone, read, copy or add a dependency.

## Copy candidates and comparison oracles

**Candidate** means a source-code reference worth a file-level audit, not
blanket copying permission. **Oracle** means an external comparison or test
role. **Hold** means the evidence is insufficient for source copying.
The selection below is a new prioritization of issue evidence, not a claim
that these were exactly the lost survey's candidates.

Registry versions are explicit snapshots, not minimum dependency versions.
The licence links identify inspected upstream revisions, which are not
asserted to be the commits that produced those crate releases. Registry
SPDX declarations and a root licence do not cover every nested asset.
Keep notices, verify the selected file's lineage, and pin its commit before
copying. Current runtime-dependency policy still applies; no regex engine,
libtorch, JVM or FFI runtime is added by this survey.

| Crate / registry version | Registry licence declaration | Pinned source licence | Role and limits | Issue evidence |
|---|---|---|---|---|
| [ptb-reader 0.9.1](https://crates.io/crates/ptb-reader/0.9.1) | `BSD-3-Clause` | [LICENSE at b975a841e78d](https://github.com/sjmielke/ptb-reader-rust/blob/b975a841e78d464db8bbaa8862003777e8422ba5/LICENSE) | **Candidate**: Bracket scanner and parser tests; corpus text remains separately licensed. | [#211](https://github.com/ai-pipestream/turboembed/issues/211), [#238](https://github.com/ai-pipestream/turboembed/issues/238) |
| [conllu 0.8.0](https://crates.io/crates/conllu/0.8.0) | `MIT OR Apache-2.0` | Unavailable; registry only | **Hold**: Dependency-reader validation. Registry licence only: the declared upstream repository could not be resolved. | [#211](https://github.com/ai-pipestream/turboembed/issues/211), [#235](https://github.com/ai-pipestream/turboembed/issues/235), [#236](https://github.com/ai-pipestream/turboembed/issues/236) |
| [udgraph 0.8.0](https://crates.io/crates/udgraph/0.8.0) | `MIT OR Apache-2.0` | Unavailable; registry only | **Hold**: Root/cycle checks. Same unresolved upstream as conllu; no source-copy approval. | [#235](https://github.com/ai-pipestream/turboembed/issues/235), [#236](https://github.com/ai-pipestream/turboembed/issues/236) |
| [conllx 0.12.1](https://crates.io/crates/conllx/0.12.1) | `Apache-2.0` | [LICENSE at 6c7d8dbae81f](https://github.com/danieldk/conllx-rs/blob/6c7d8dbae81f96d66ef0d8fb8aa9eca95d919c6c/LICENSE) | **Candidate**: Reader edge cases; preserve OpenNLP column semantics. | [#234](https://github.com/ai-pipestream/turboembed/issues/234) |
| [crfs 0.4.1](https://crates.io/crates/crfs/0.4.1) | `MIT` | [LICENSE at a8b5e9914629](https://github.com/messense/crfs-rs/blob/a8b5e9914629983b0cc0e6080e7e53a060673024/LICENSE) | **Candidate**: CRF layout and Viterbi reference; not a substitute for JVM scoring/training parity. | [#173](https://github.com/ai-pipestream/turboembed/issues/173), [#206](https://github.com/ai-pipestream/turboembed/issues/206), [#229](https://github.com/ai-pipestream/turboembed/issues/229) |
| [liblbfgs 0.1.0](https://crates.io/crates/liblbfgs/0.1.0) | `MIT` | [LICENSE at 3671d41fd87f](https://github.com/ybyygu/rust-lbfgs/blob/3671d41fd87f01392b51a462a49addd66e9c48c5/LICENSE) | **Candidate**: OWL-QN pseudo-gradients and orthant projection tests; preserve the JVM line search. | [#228](https://github.com/ai-pipestream/turboembed/issues/228) |
| [libsvm-rs 0.9.1](https://crates.io/crates/libsvm-rs/0.9.1) | `BSD-3-Clause` | [LICENSE at 858df3f97a1a](https://github.com/ricardofrantz/libsvm-rs/blob/858df3f97a1a300a263a33ce4e8b56ba187748c4/LICENSE) | **Candidate**: Prediction and model-format logic; BSD notices and score tolerances still apply. | [#214](https://github.com/ai-pipestream/turboembed/issues/214) |
| [rust-bert 0.23.0](https://crates.io/crates/rust-bert/0.23.0) | `Apache-2.0` | [LICENSE at 6db859ef097e](https://github.com/guillaume-be/rust-bert/blob/6db859ef097edfdda338004b4d60deebf6a3ab66/LICENSE) | **Candidate**: Subword label aggregation and source offsets; no libtorch dependency or model download. | [#207](https://github.com/ai-pipestream/turboembed/issues/207), [#208](https://github.com/ai-pipestream/turboembed/issues/208) |
| [charabia 0.10.0](https://crates.io/crates/charabia/0.10.0) | `MIT` | [LICENSE at bc63b9860434](https://github.com/meilisearch/charabia/blob/bc63b9860434f06cf78b51bb0eab00aaefb69f15/LICENSE) | **Candidate**: Language-selected normalizer chain; dictionaries and FST data are not approved by the code licence. | [#184](https://github.com/ai-pipestream/turboembed/issues/184) |
| [text-splitter 0.33.0](https://crates.io/crates/text-splitter/0.33.0) | `MIT` | [LICENSE.txt at bdfe0fbe2cae](https://github.com/benbrandt/text-splitter/blob/bdfe0fbe2caefbb855dd65933fbc5d01b588a2e7/LICENSE.txt) | **Candidate**: Semantic level ordering and greedy baseline; test corpus provenance is separate. | [#167](https://github.com/ai-pipestream/turboembed/issues/167) |
| [tokenizations 0.4.2](https://crates.io/crates/tokenizations/0.4.2) | `MIT` | [LICENSE at bcb27e901c2a](https://github.com/tamuhey/tokenizations/blob/bcb27e901c2a9413b45dfd030c6ff8764d753188/LICENSE) | **Candidate**: Alignment algorithm; exact spans can instead use a sorted interval merge. | [#166](https://github.com/ai-pipestream/turboembed/issues/166) |
| [unicode-security 0.1.2](https://crates.io/crates/unicode-security/0.1.2) | `MIT/Apache-2.0` | [LICENSE-APACHE at 1b4a80b48147](https://github.com/unicode-rs/unicode-security/blob/1b4a80b48147f5f323fb2dfc784bcf2a2b4bbc4a/LICENSE-APACHE) | **Candidate**: Table generator and skeleton oracle; pin the Unicode data release and its notices. | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| [text2num 2.8.0](https://crates.io/crates/text2num/2.8.0) | `MIT` | [LICENSE at 8dc343634e1f](https://github.com/allo-media/text2num-rs/blob/8dc343634e1ffe18a2bd1e651817226a46a6045d/LICENSE) | **Candidate**: Number grammar and test logic; word tables need per-file provenance and Java parity checks. | [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| [interim 0.2.1](https://crates.io/crates/interim/0.2.1) | `MIT` | [LICENSE.txt at a0887dc6278c](https://github.com/conradludgate/interim/blob/a0887dc6278c7f6d06c490a6399ce44b74e776cd/LICENSE.txt) | **Candidate**: Relative-date grammar with a supplied base date; whole-string behavior differs from span extraction. | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| [twitter-text 0.3.1](https://crates.io/crates/twitter-text/0.3.1) | `Apache-2.0` | [LICENSE at 7e1956b9bb91](https://github.com/sayrer/twitter-text/blob/7e1956b9bb914b3c1ffec25da1650024f5623336/LICENSE) | **Candidate**: Social-text extraction rules and conformance logic; Rust port is distinct from the multilingual upstream. | [#182](https://github.com/ai-pipestream/turboembed/issues/182) |
| [elinor 0.4.0](https://crates.io/crates/elinor/0.4.0) | `MIT OR Apache-2.0` | [LICENSE-APACHE at 69cce1abf17c](https://github.com/kampersanda/elinor/blob/69cce1abf17cf89b6bd9895027ce24d9e3465c7b/LICENSE-APACHE) | **Oracle**: IR metrics and paired-test cross-checks; a dev/test role, not an automatic dependency. | [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#172](https://github.com/ai-pipestream/turboembed/issues/172) |
| [lindera 6.2.0](https://crates.io/crates/lindera/6.2.0) | `MIT` | [LICENSE at 049d87e9a71e](https://github.com/lindera/lindera/blob/049d87e9a71e15c139a0488f731adc50b7d6a7a3/LICENSE) | **Oracle**: Lattice formats, positions and readings; dictionaries need independent licences. | [#191](https://github.com/ai-pipestream/turboembed/issues/191), [#192](https://github.com/ai-pipestream/turboembed/issues/192), [#354](https://github.com/ai-pipestream/turboembed/issues/354) |
| [vibrato 0.5.2](https://crates.io/crates/vibrato/0.5.2) | `MIT OR Apache-2.0` | [LICENSE-APACHE at 6d80c52a2e5c](https://github.com/daac-tools/vibrato/blob/6d80c52a2e5c200fc0c7091840487ee55faf9aaa/LICENSE-APACHE) | **Oracle**: MeCab-format behavior and throughput; no dictionary approval implied. | [#191](https://github.com/ai-pipestream/turboembed/issues/191), [#354](https://github.com/ai-pipestream/turboembed/issues/354) |
| [lingua 1.8.0](https://crates.io/crates/lingua/1.8.0) | `Apache-2.0` | [LICENSE at 086bc1023433](https://github.com/pemistahl/lingua-rs/blob/086bc102343361140686491e64a4ba13501404d7/LICENSE) | **Oracle**: Short-text and mixed-language accuracy; verify model/data notices before any redistribution. | [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#239](https://github.com/ai-pipestream/turboembed/issues/239), [#389](https://github.com/ai-pipestream/turboembed/issues/389) |
| [whatlang 0.18.0](https://crates.io/crates/whatlang/0.18.0) | `MIT` | [LICENSE at 7ddb6bbd158a](https://github.com/greyblake/whatlang-rs/blob/7ddb6bbd158a27844378dd6df0f33f473cd3d731/LICENSE) | **Oracle**: Script and language filtering; model/feature tables require source-level checks. | [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#239](https://github.com/ai-pipestream/turboembed/issues/239), [#389](https://github.com/ai-pipestream/turboembed/issues/389) |
| [whichlang 0.1.1](https://crates.io/crates/whichlang/0.1.1) | `MIT` | [LICENSE at 6c1c8b63ea61](https://github.com/quickwit-oss/whichlang/blob/6c1c8b63ea61720306a19bee1bb555ec4646b635/LICENSE) | **Oracle**: Compact language classifier comparison; audit embedded model provenance separately. | [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#389](https://github.com/ai-pipestream/turboembed/issues/389) |
| [daachorse 5.1.0](https://crates.io/crates/daachorse/5.1.0) | `MIT OR Apache-2.0` | [LICENSE-APACHE at 7e94aa16917f](https://github.com/daac-tools/daachorse/blob/7e94aa16917fd97c19f77f5791117bd1ba4018f4/LICENSE-APACHE) | **Candidate**: Double-array matching and bounded traversal; preserve token-id/span semantics. | [#177](https://github.com/ai-pipestream/turboembed/issues/177), [#185](https://github.com/ai-pipestream/turboembed/issues/185), [#186](https://github.com/ai-pipestream/turboembed/issues/186), [#192](https://github.com/ai-pipestream/turboembed/issues/192), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#202](https://github.com/ai-pipestream/turboembed/issues/202), [#385](https://github.com/ai-pipestream/turboembed/issues/385) |
| [fst 0.4.7](https://crates.io/crates/fst/0.4.7) | `Unlicense/MIT` | [UNLICENSE at 5907b4739793](https://github.com/BurntSushi/fst/blob/5907b4739793b3d5d7061eaa3f85274e09769d6a/UNLICENSE) | **Candidate**: Compact phrase dictionaries and lookup tests; retain the chosen licence notice. | [#174](https://github.com/ai-pipestream/turboembed/issues/174), [#175](https://github.com/ai-pipestream/turboembed/issues/175), [#189](https://github.com/ai-pipestream/turboembed/issues/189), [#198](https://github.com/ai-pipestream/turboembed/issues/198), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#201](https://github.com/ai-pipestream/turboembed/issues/201), [#216](https://github.com/ai-pipestream/turboembed/issues/216), [#221](https://github.com/ai-pipestream/turboembed/issues/221), [#224](https://github.com/ai-pipestream/turboembed/issues/224), [#385](https://github.com/ai-pipestream/turboembed/issues/385) |
| [aho-corasick 1.1.5](https://crates.io/crates/aho-corasick/1.1.5) | `Unlicense OR MIT` | [UNLICENSE at 6c0abf5681bf](https://github.com/BurntSushi/aho-corasick/blob/6c0abf5681bfc30bb9d8f7f52b68a350b436fffa/UNLICENSE) | **Candidate**: Multi-pattern matching algorithms; does not override the no-runtime-regex rule. | [#177](https://github.com/ai-pipestream/turboembed/issues/177), [#185](https://github.com/ai-pipestream/turboembed/issues/185), [#186](https://github.com/ai-pipestream/turboembed/issues/186), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#202](https://github.com/ai-pipestream/turboembed/issues/202), [#385](https://github.com/ai-pipestream/turboembed/issues/385) |
| [encoding_rs 0.8.42](https://crates.io/crates/encoding_rs/0.8.42) | `(Apache-2.0 OR MIT) AND BSD-3-Clause` | [LICENSE-APACHE at 9db0e447f07c](https://github.com/hsivonen/encoding_rs/blob/9db0e447f07c34b9969a258a2a1237cc6cf341d5/LICENSE-APACHE) | **Candidate**: Incremental decoder accounting and alignment; BSD data notices are additional to MIT/Apache choices. | [#187](https://github.com/ai-pipestream/turboembed/issues/187), [#233](https://github.com/ai-pipestream/turboembed/issues/233), [#234](https://github.com/ai-pipestream/turboembed/issues/234), [#237](https://github.com/ai-pipestream/turboembed/issues/237), [#238](https://github.com/ai-pipestream/turboembed/issues/238), [#384](https://github.com/ai-pipestream/turboembed/issues/384) |
| [chardetng 1.0.0](https://crates.io/crates/chardetng/1.0.0) | `Apache-2.0 OR MIT` | [LICENSE-APACHE at 4a53a99e07e4](https://github.com/hsivonen/chardetng/blob/4a53a99e07e418ceb91e0049580e6bbcb2df0591/LICENSE-APACHE) | **Oracle**: Encoding detector tests and accuracy; no claim of source alignment from detection alone. | [#187](https://github.com/ai-pipestream/turboembed/issues/187), [#384](https://github.com/ai-pipestream/turboembed/issues/384) |
| [nlprule 0.6.4](https://crates.io/crates/nlprule/0.6.4) | `MIT OR Apache-2.0` | [LICENSE-APACHE at ebb54617ea5f](https://github.com/bminixhofer/nlprule/blob/ebb54617ea5f3378c0dc63486631724b7e97e13c/LICENSE-APACHE) | **Code-only hold**: Issues #174/#176 name permissive feature/beam logic, but LanguageTool-derived binaries are excluded. Per-file lineage must be checked before copying. | [#174](https://github.com/ai-pipestream/turboembed/issues/174), [#175](https://github.com/ai-pipestream/turboembed/issues/175), [#176](https://github.com/ai-pipestream/turboembed/issues/176), [#209](https://github.com/ai-pipestream/turboembed/issues/209) |

The [machine-readable evidence](rust-reference-evidence.json) records these
27 package versions and archive SHA-256 identifiers from registry metadata,
along with the inspected source-licence pins and licence-file hashes. Those
archive hashes identify metadata; this survey did not fetch model weights,
corpora or package archives. It inspected licence files, not all source files.

## Exclusions and asset boundaries

- Do not copy `lt-chunk` or `confusables`; the issue evidence identifies
  LGPL/GPL source restrictions. The broader do-not-read list in the
  [other-language survey](other-languages.md) also applies to wrappers.
- `kenlm-rs` and `udpipe-rs` do not make their wrapped implementations
  permissible. `nlprule` code and its LanguageTool-derived binaries must
  not be conflated. It is held out of the source-copy set pending lineage
  verification, even though its registry code licence is permissive.
- Hunspell wrappers, hyphenation patterns, dictionary packages, stopword
  lists, GeoNames tables, Public Suffix List data and neural checkpoints
  have independent terms. A permissive Rust package licence is not an
  asset licence. Do not import LGPL, CC BY-SA or non-commercial assets
  into the reference collection.
- Unicode and ICU4X release versions can differ from the JDK tables used
  for parity. A newer table is not automatically a correct replacement.
  Exact offsets and Java semantics take priority over API similarity.
- General trainers (`linfa`, `smartcore`, `argmin`) may provide numerical
  cross-checks, but cannot establish OpenNLP feature or score parity.
  The source issue states the intended role and numerical caveats.

## Complete issue-reference index

The following is an index, not an allowlist or a licence audit. It retains
both positive and negative issue mentions so exclusions are not erased.
It excludes no crate merely because its issue is closed. Names written
without crates.io URLs are covered by the selected table and the
other-language survey where explicit issue evidence supports them.

| Crate mentioned | Source issues |
|---|---|
| `aho-corasick` | [#177](https://github.com/ai-pipestream/turboembed/issues/177), [#185](https://github.com/ai-pipestream/turboembed/issues/185), [#186](https://github.com/ai-pipestream/turboembed/issues/186), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#202](https://github.com/ai-pipestream/turboembed/issues/202) |
| `any_ascii` | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#179](https://github.com/ai-pipestream/turboembed/issues/179) |
| `approx` | [#153](https://github.com/ai-pipestream/turboembed/issues/153) |
| `argmin` | [#228](https://github.com/ai-pipestream/turboembed/issues/228) |
| `base64` | [#188](https://github.com/ai-pipestream/turboembed/issues/188) |
| `base64-simd` | [#188](https://github.com/ai-pipestream/turboembed/issues/188) |
| `bech32` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `bm25` | [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#171](https://github.com/ai-pipestream/turboembed/issues/171) |
| `bpe` | [#194](https://github.com/ai-pipestream/turboembed/issues/194) |
| `bs58` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `candle-core` | [#168](https://github.com/ai-pipestream/turboembed/issues/168) |
| `candle-crf` | [#206](https://github.com/ai-pipestream/turboembed/issues/206) |
| `candle-nn` | [#205](https://github.com/ai-pipestream/turboembed/issues/205), [#206](https://github.com/ai-pipestream/turboembed/issues/206) |
| `candle-transformers` | [#207](https://github.com/ai-pipestream/turboembed/issues/207), [#208](https://github.com/ai-pipestream/turboembed/issues/208) |
| `card-validate` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `caseless` | [#157](https://github.com/ai-pipestream/turboembed/issues/157) |
| `celes` | [#221](https://github.com/ai-pipestream/turboembed/issues/221), [#223](https://github.com/ai-pipestream/turboembed/issues/223) |
| `cesu8` | [#155](https://github.com/ai-pipestream/turboembed/issues/155) |
| `charabia` | [#184](https://github.com/ai-pipestream/turboembed/issues/184) |
| `chardetng` | [#187](https://github.com/ai-pipestream/turboembed/issues/187) |
| `charset-normalizer-rs` | [#187](https://github.com/ai-pipestream/turboembed/issues/187) |
| `chrono` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `chrono-english` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `chunk` | [#167](https://github.com/ai-pipestream/turboembed/issues/167) |
| `cld2` | [#164](https://github.com/ai-pipestream/turboembed/issues/164) |
| `comrak` | [#165](https://github.com/ai-pipestream/turboembed/issues/165) |
| `confusables` | [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `conll` | [#234](https://github.com/ai-pipestream/turboembed/issues/234) |
| `conllu` | [#211](https://github.com/ai-pipestream/turboembed/issues/211), [#235](https://github.com/ai-pipestream/turboembed/issues/235), [#236](https://github.com/ai-pipestream/turboembed/issues/236) |
| `conllx` | [#234](https://github.com/ai-pipestream/turboembed/issues/234) |
| `contractions` | [#185](https://github.com/ai-pipestream/turboembed/issues/185) |
| `country-boundaries` | [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `crawdad` | [#191](https://github.com/ai-pipestream/turboembed/issues/191) |
| `crfs` | [#173](https://github.com/ai-pipestream/turboembed/issues/173), [#206](https://github.com/ai-pipestream/turboembed/issues/206), [#229](https://github.com/ai-pipestream/turboembed/issues/229) |
| `csv` | [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#202](https://github.com/ai-pipestream/turboembed/issues/202), [#218](https://github.com/ai-pipestream/turboembed/issues/218), [#221](https://github.com/ai-pipestream/turboembed/issues/221) |
| `daachorse` | [#177](https://github.com/ai-pipestream/turboembed/issues/177), [#185](https://github.com/ai-pipestream/turboembed/issues/185), [#186](https://github.com/ai-pipestream/turboembed/issues/186), [#192](https://github.com/ai-pipestream/turboembed/issues/192), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#202](https://github.com/ai-pipestream/turboembed/issues/202) |
| `data-url` | [#188](https://github.com/ai-pipestream/turboembed/issues/188) |
| `dateparser` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `decancer` | [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `deunicode` | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#179](https://github.com/ai-pipestream/turboembed/issues/179) |
| `dimensioned` | [#219](https://github.com/ai-pipestream/turboembed/issues/219) |
| `duckling` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `ecb-rates` | [#218](https://github.com/ai-pipestream/turboembed/issues/218) |
| `elinor` | [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#172](https://github.com/ai-pipestream/turboembed/issues/172) |
| `email_address` | [#181](https://github.com/ai-pipestream/turboembed/issues/181), [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `emojis` | [#183](https://github.com/ai-pipestream/turboembed/issues/183) |
| `emojito` | [#183](https://github.com/ai-pipestream/turboembed/issues/183) |
| `encoding_rs` | [#187](https://github.com/ai-pipestream/turboembed/issues/187), [#233](https://github.com/ai-pipestream/turboembed/issues/233), [#234](https://github.com/ai-pipestream/turboembed/issues/234), [#237](https://github.com/ai-pipestream/turboembed/issues/237), [#238](https://github.com/ai-pipestream/turboembed/issues/238) |
| `fancy-regex` | [#177](https://github.com/ai-pipestream/turboembed/issues/177) |
| `fastembed` | [#168](https://github.com/ai-pipestream/turboembed/issues/168), [#169](https://github.com/ai-pipestream/turboembed/issues/169) |
| `fend-core` | [#219](https://github.com/ai-pipestream/turboembed/issues/219) |
| `ffsvm` | [#214](https://github.com/ai-pipestream/turboembed/issues/214) |
| `finitediff` | [#228](https://github.com/ai-pipestream/turboembed/issues/228) |
| `fixed_decimal` | [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| `flate2` | [#155](https://github.com/ai-pipestream/turboembed/issues/155) |
| `float-cmp` | [#153](https://github.com/ai-pipestream/turboembed/issues/153) |
| `focaccia` | [#157](https://github.com/ai-pipestream/turboembed/issues/157) |
| `fst` | [#174](https://github.com/ai-pipestream/turboembed/issues/174), [#175](https://github.com/ai-pipestream/turboembed/issues/175), [#189](https://github.com/ai-pipestream/turboembed/issues/189), [#198](https://github.com/ai-pipestream/turboembed/issues/198), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#201](https://github.com/ai-pipestream/turboembed/issues/201), [#216](https://github.com/ai-pipestream/turboembed/issues/216), [#221](https://github.com/ai-pipestream/turboembed/issues/221), [#224](https://github.com/ai-pipestream/turboembed/issues/224) |
| `geo` | [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `geographiclib-rs` | [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `geohash` | [#222](https://github.com/ai-pipestream/turboembed/issues/222), [#224](https://github.com/ai-pipestream/turboembed/issues/224) |
| `gline-rs` | [#207](https://github.com/ai-pipestream/turboembed/issues/207) |
| `h3o` | [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `hex` | [#188](https://github.com/ai-pipestream/turboembed/issues/188) |
| `hf-hub` | [#154](https://github.com/ai-pipestream/turboembed/issues/154) |
| `hmac` | [#226](https://github.com/ai-pipestream/turboembed/issues/226) |
| `hunspell-rs` | [#197](https://github.com/ai-pipestream/turboembed/issues/197) |
| `hunspell-sys` | [#197](https://github.com/ai-pipestream/turboembed/issues/197) |
| `hyphenation` | [#159](https://github.com/ai-pipestream/turboembed/issues/159) |
| `iban_validate` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `icu_casemap` | [#157](https://github.com/ai-pipestream/turboembed/issues/157), [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `icu_decimal` | [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| `icu_normalizer` | [#156](https://github.com/ai-pipestream/turboembed/issues/156), [#179](https://github.com/ai-pipestream/turboembed/issues/179), [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `icu_properties` | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#159](https://github.com/ai-pipestream/turboembed/issues/159), [#160](https://github.com/ai-pipestream/turboembed/issues/160), [#179](https://github.com/ai-pipestream/turboembed/issues/179), [#182](https://github.com/ai-pipestream/turboembed/issues/182), [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `icu_segmenter` | [#160](https://github.com/ai-pipestream/turboembed/issues/160), [#163](https://github.com/ai-pipestream/turboembed/issues/163) |
| `infer` | [#188](https://github.com/ai-pipestream/turboembed/issues/188) |
| `insta` | [#153](https://github.com/ai-pipestream/turboembed/issues/153) |
| `interim` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `iso_currency` | [#218](https://github.com/ai-pipestream/turboembed/issues/218), [#223](https://github.com/ai-pipestream/turboembed/issues/223) |
| `j4rs` | [#152](https://github.com/ai-pipestream/turboembed/issues/152) |
| `java-properties` | [#155](https://github.com/ai-pipestream/turboembed/issues/155) |
| `jieba-rs` | [#192](https://github.com/ai-pipestream/turboembed/issues/192) |
| `jiff` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `jni` | [#152](https://github.com/ai-pipestream/turboembed/issues/152) |
| `jpreprocess` | [#191](https://github.com/ai-pipestream/turboembed/issues/191) |
| `kenlm-rs` | [#216](https://github.com/ai-pipestream/turboembed/issues/216) |
| `kiddo` | [#221](https://github.com/ai-pipestream/turboembed/issues/221), [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `lbfgs` | [#228](https://github.com/ai-pipestream/turboembed/issues/228) |
| `levenshtein_automata` | [#189](https://github.com/ai-pipestream/turboembed/issues/189), [#224](https://github.com/ai-pipestream/turboembed/issues/224) |
| `liblbfgs` | [#228](https://github.com/ai-pipestream/turboembed/issues/228) |
| `liblinear` | [#231](https://github.com/ai-pipestream/turboembed/issues/231) |
| `libm` | [#155](https://github.com/ai-pipestream/turboembed/issues/155), [#216](https://github.com/ai-pipestream/turboembed/issues/216) |
| `libsvm-rs` | [#214](https://github.com/ai-pipestream/turboembed/issues/214) |
| `lindera` | [#191](https://github.com/ai-pipestream/turboembed/issues/191), [#192](https://github.com/ai-pipestream/turboembed/issues/192) |
| `lindera-cc-cedict` | [#192](https://github.com/ai-pipestream/turboembed/issues/192) |
| `lindera-crf` | [#206](https://github.com/ai-pipestream/turboembed/issues/206) |
| `linfa` | [#170](https://github.com/ai-pipestream/turboembed/issues/170) |
| `linfa-bayes` | [#230](https://github.com/ai-pipestream/turboembed/issues/230) |
| `linfa-logistic` | [#213](https://github.com/ai-pipestream/turboembed/issues/213), [#231](https://github.com/ai-pipestream/turboembed/issues/231) |
| `linfa-svm` | [#214](https://github.com/ai-pipestream/turboembed/issues/214), [#231](https://github.com/ai-pipestream/turboembed/issues/231) |
| `lingua` | [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#239](https://github.com/ai-pipestream/turboembed/issues/239) |
| `linkify` | [#181](https://github.com/ai-pipestream/turboembed/issues/181) |
| `lt-chunk` | [#155](https://github.com/ai-pipestream/turboembed/issues/155), [#173](https://github.com/ai-pipestream/turboembed/issues/173) |
| `luhn` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `markdown` | [#165](https://github.com/ai-pipestream/turboembed/issues/165) |
| `measurements` | [#219](https://github.com/ai-pipestream/turboembed/issues/219) |
| `memchr` | [#163](https://github.com/ai-pipestream/turboembed/issues/163), [#181](https://github.com/ai-pipestream/turboembed/issues/181), [#198](https://github.com/ai-pipestream/turboembed/issues/198), [#237](https://github.com/ai-pipestream/turboembed/issues/237) |
| `n_gram` | [#216](https://github.com/ai-pipestream/turboembed/issues/216) |
| `ndarray` | [#169](https://github.com/ai-pipestream/turboembed/issues/169), [#205](https://github.com/ai-pipestream/turboembed/issues/205) |
| `nlprule` | [#174](https://github.com/ai-pipestream/turboembed/issues/174), [#175](https://github.com/ai-pipestream/turboembed/issues/175), [#176](https://github.com/ai-pipestream/turboembed/issues/176), [#209](https://github.com/ai-pipestream/turboembed/issues/209) |
| `nnsplit` | [#161](https://github.com/ai-pipestream/turboembed/issues/161) |
| `num-format` | [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| `ordered-float` | [#167](https://github.com/ai-pipestream/turboembed/issues/167), [#172](https://github.com/ai-pipestream/turboembed/issues/172) |
| `ort` | [#169](https://github.com/ai-pipestream/turboembed/issues/169) |
| `parse_datetime` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `pathfinding` | [#167](https://github.com/ai-pipestream/turboembed/issues/167) |
| `perceptron` | [#229](https://github.com/ai-pipestream/turboembed/issues/229) |
| `phonenumber` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `pretty_assertions` | [#153](https://github.com/ai-pipestream/turboembed/issues/153) |
| `proptest` | [#153](https://github.com/ai-pipestream/turboembed/issues/153), [#156](https://github.com/ai-pipestream/turboembed/issues/156) |
| `prost` | [#193](https://github.com/ai-pipestream/turboembed/issues/193) |
| `psl` | [#181](https://github.com/ai-pipestream/turboembed/issues/181) |
| `ptb-reader` | [#211](https://github.com/ai-pipestream/turboembed/issues/211), [#238](https://github.com/ai-pipestream/turboembed/issues/238) |
| `publicsuffix` | [#181](https://github.com/ai-pipestream/turboembed/issues/181) |
| `pulldown-cmark` | [#165](https://github.com/ai-pipestream/turboembed/issues/165) |
| `punkt` | [#161](https://github.com/ai-pipestream/turboembed/issues/161) |
| `quick-xml` | [#161](https://github.com/ai-pipestream/turboembed/issues/161), [#173](https://github.com/ai-pipestream/turboembed/issues/173), [#174](https://github.com/ai-pipestream/turboembed/issues/174), [#195](https://github.com/ai-pipestream/turboembed/issues/195), [#199](https://github.com/ai-pipestream/turboembed/issues/199), [#201](https://github.com/ai-pipestream/turboembed/issues/201), [#202](https://github.com/ai-pipestream/turboembed/issues/202), [#216](https://github.com/ai-pipestream/turboembed/issues/216), [#233](https://github.com/ai-pipestream/turboembed/issues/233), [#238](https://github.com/ai-pipestream/turboembed/issues/238), [#239](https://github.com/ai-pipestream/turboembed/issues/239) |
| `rayon` | [#227](https://github.com/ai-pipestream/turboembed/issues/227) |
| `redact-core` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `regex` | [#162](https://github.com/ai-pipestream/turboembed/issues/162), [#177](https://github.com/ai-pipestream/turboembed/issues/177) |
| `reverse_geocoder` | [#221](https://github.com/ai-pipestream/turboembed/issues/221) |
| `roxmltree` | [#161](https://github.com/ai-pipestream/turboembed/issues/161), [#173](https://github.com/ai-pipestream/turboembed/issues/173), [#195](https://github.com/ai-pipestream/turboembed/issues/195), [#233](https://github.com/ai-pipestream/turboembed/issues/233), [#238](https://github.com/ai-pipestream/turboembed/issues/238) |
| `rstar` | [#221](https://github.com/ai-pipestream/turboembed/issues/221), [#222](https://github.com/ai-pipestream/turboembed/issues/222) |
| `rust-bert` | [#207](https://github.com/ai-pipestream/turboembed/issues/207), [#208](https://github.com/ai-pipestream/turboembed/issues/208) |
| `rust-stemmers` | [#171](https://github.com/ai-pipestream/turboembed/issues/171), [#184](https://github.com/ai-pipestream/turboembed/issues/184), [#196](https://github.com/ai-pipestream/turboembed/issues/196) |
| `rust_decimal` | [#217](https://github.com/ai-pipestream/turboembed/issues/217), [#218](https://github.com/ai-pipestream/turboembed/issues/218) |
| `rust_iso3166` | [#223](https://github.com/ai-pipestream/turboembed/issues/223) |
| `rusty-money` | [#218](https://github.com/ai-pipestream/turboembed/issues/218), [#223](https://github.com/ai-pipestream/turboembed/issues/223) |
| `secrecy` | [#226](https://github.com/ai-pipestream/turboembed/issues/226) |
| `sentencepiece` | [#193](https://github.com/ai-pipestream/turboembed/issues/193) |
| `sentencepiece-model` | [#193](https://github.com/ai-pipestream/turboembed/issues/193) |
| `sentencepiece-rs` | [#193](https://github.com/ai-pipestream/turboembed/issues/193) |
| `sentencex` | [#161](https://github.com/ai-pipestream/turboembed/issues/161) |
| `serde_json` | [#152](https://github.com/ai-pipestream/turboembed/issues/152) |
| `sha2` | [#152](https://github.com/ai-pipestream/turboembed/issues/152), [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#155](https://github.com/ai-pipestream/turboembed/issues/155), [#225](https://github.com/ai-pipestream/turboembed/issues/225), [#226](https://github.com/ai-pipestream/turboembed/issues/226) |
| `sha3` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `simdutf8` | [#187](https://github.com/ai-pipestream/turboembed/issues/187) |
| `similar` | [#153](https://github.com/ai-pipestream/turboembed/issues/153) |
| `simsimd` | [#169](https://github.com/ai-pipestream/turboembed/issues/169), [#170](https://github.com/ai-pipestream/turboembed/issues/170) |
| `smartcore` | [#214](https://github.com/ai-pipestream/turboembed/issues/214), [#231](https://github.com/ai-pipestream/turboembed/issues/231) |
| `spellbook` | [#197](https://github.com/ai-pipestream/turboembed/issues/197) |
| `srx` | [#161](https://github.com/ai-pipestream/turboembed/issues/161) |
| `statrs` | [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#230](https://github.com/ai-pipestream/turboembed/issues/230) |
| `stop-words` | [#200](https://github.com/ai-pipestream/turboembed/issues/200) |
| `stopwords` | [#200](https://github.com/ai-pipestream/turboembed/issues/200) |
| `strsim` | [#189](https://github.com/ai-pipestream/turboembed/issues/189), [#224](https://github.com/ai-pipestream/turboembed/issues/224) |
| `symspell` | [#189](https://github.com/ai-pipestream/turboembed/issues/189) |
| `symspell_rs` | [#189](https://github.com/ai-pipestream/turboembed/issues/189) |
| `tantivy` | [#154](https://github.com/ai-pipestream/turboembed/issues/154), [#171](https://github.com/ai-pipestream/turboembed/issues/171), [#184](https://github.com/ai-pipestream/turboembed/issues/184), [#224](https://github.com/ai-pipestream/turboembed/issues/224) |
| `tantivy-stemmers` | [#196](https://github.com/ai-pipestream/turboembed/issues/196) |
| `text-splitter` | [#167](https://github.com/ai-pipestream/turboembed/issues/167) |
| `text2num` | [#217](https://github.com/ai-pipestream/turboembed/issues/217) |
| `thesaurus` | [#203](https://github.com/ai-pipestream/turboembed/issues/203) |
| `tiktoken-rs` | [#194](https://github.com/ai-pipestream/turboembed/issues/194) |
| `tiny-keccak` | [#225](https://github.com/ai-pipestream/turboembed/issues/225) |
| `tokenizations` | [#166](https://github.com/ai-pipestream/turboembed/issues/166) |
| `tokenizers` | [#162](https://github.com/ai-pipestream/turboembed/issues/162), [#166](https://github.com/ai-pipestream/turboembed/issues/166), [#193](https://github.com/ai-pipestream/turboembed/issues/193), [#194](https://github.com/ai-pipestream/turboembed/issues/194), [#195](https://github.com/ai-pipestream/turboembed/issues/195), [#207](https://github.com/ai-pipestream/turboembed/issues/207) |
| `tongrams` | [#216](https://github.com/ai-pipestream/turboembed/issues/216) |
| `tree-sitter-md` | [#165](https://github.com/ai-pipestream/turboembed/issues/165) |
| `triple_accel` | [#189](https://github.com/ai-pipestream/turboembed/issues/189) |
| `twitter-text` | [#182](https://github.com/ai-pipestream/turboembed/issues/182) |
| `two_timer` | [#220](https://github.com/ai-pipestream/turboembed/issues/220) |
| `ucd-trie` | [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `udgraph` | [#235](https://github.com/ai-pipestream/turboembed/issues/235), [#236](https://github.com/ai-pipestream/turboembed/issues/236) |
| `udpipe-rs` | [#235](https://github.com/ai-pipestream/turboembed/issues/235) |
| `unicase` | [#157](https://github.com/ai-pipestream/turboembed/issues/157) |
| `unicode-case-mapping` | [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `unicode-general-category` | [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `unicode-normalization` | [#156](https://github.com/ai-pipestream/turboembed/issues/156), [#178](https://github.com/ai-pipestream/turboembed/issues/178), [#179](https://github.com/ai-pipestream/turboembed/issues/179), [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `unicode-normalization-alignments` | [#156](https://github.com/ai-pipestream/turboembed/issues/156) |
| `unicode-properties` | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#182](https://github.com/ai-pipestream/turboembed/issues/182), [#183](https://github.com/ai-pipestream/turboembed/issues/183), [#187](https://github.com/ai-pipestream/turboembed/issues/187) |
| `unicode-script` | [#187](https://github.com/ai-pipestream/turboembed/issues/187) |
| `unicode-security` | [#158](https://github.com/ai-pipestream/turboembed/issues/158), [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `unicode-segmentation` | [#160](https://github.com/ai-pipestream/turboembed/issues/160), [#162](https://github.com/ai-pipestream/turboembed/issues/162), [#182](https://github.com/ai-pipestream/turboembed/issues/182), [#183](https://github.com/ai-pipestream/turboembed/issues/183) |
| `unicode-skeleton` | [#180](https://github.com/ai-pipestream/turboembed/issues/180) |
| `unicode_categories` | [#190](https://github.com/ai-pipestream/turboembed/issues/190) |
| `unidecode` | [#179](https://github.com/ai-pipestream/turboembed/issues/179) |
| `uom` | [#219](https://github.com/ai-pipestream/turboembed/issues/219) |
| `url` | [#181](https://github.com/ai-pipestream/turboembed/issues/181) |
| `vader-parity` | [#215](https://github.com/ai-pipestream/turboembed/issues/215) |
| `vader-sentimental` | [#215](https://github.com/ai-pipestream/turboembed/issues/215) |
| `vader_sentiment` | [#215](https://github.com/ai-pipestream/turboembed/issues/215) |
| `vaporetto` | [#192](https://github.com/ai-pipestream/turboembed/issues/192) |
| `verbora-wordnet` | [#203](https://github.com/ai-pipestream/turboembed/issues/203) |
| `vibrato` | [#191](https://github.com/ai-pipestream/turboembed/issues/191) |
| `waken_snowball` | [#184](https://github.com/ai-pipestream/turboembed/issues/184), [#196](https://github.com/ai-pipestream/turboembed/issues/196) |
| `whatlang` | [#164](https://github.com/ai-pipestream/turboembed/issues/164), [#239](https://github.com/ai-pipestream/turboembed/issues/239) |
| `whichlang` | [#164](https://github.com/ai-pipestream/turboembed/issues/164) |
| `wn` | [#201](https://github.com/ai-pipestream/turboembed/issues/201) |
| `wordnet` | [#201](https://github.com/ai-pipestream/turboembed/issues/201) |
| `wordnet-db` | [#201](https://github.com/ai-pipestream/turboembed/issues/201), [#203](https://github.com/ai-pipestream/turboembed/issues/203) |
| `wordnet-lemmatizer` | [#204](https://github.com/ai-pipestream/turboembed/issues/204) |
| `wordnet-morphy` | [#204](https://github.com/ai-pipestream/turboembed/issues/204) |
| `wordnet-types` | [#201](https://github.com/ai-pipestream/turboembed/issues/201), [#203](https://github.com/ai-pipestream/turboembed/issues/203) |
| `yada` | [#191](https://github.com/ai-pipestream/turboembed/issues/191), [#192](https://github.com/ai-pipestream/turboembed/issues/192) |
| `zeroize` | [#226](https://github.com/ai-pipestream/turboembed/issues/226) |
| `zip` | [#155](https://github.com/ai-pipestream/turboembed/issues/155), [#212](https://github.com/ai-pipestream/turboembed/issues/212), [#227](https://github.com/ai-pipestream/turboembed/issues/227) |
| `zspell` | [#197](https://github.com/ai-pipestream/turboembed/issues/197) |
