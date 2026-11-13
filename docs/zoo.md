# The model zoo

The zoo lists the models, classifier heads, lexicons, label centroid
sets and datasets the analysis add-ons use, and fetches them according
to the licence policy on #326. Annotators ask it for an artifact by id
and get back local files whose size, SHA-256 and SHA-512 have been
checked. `turbo-zoo` (`analysis/zoo/`) is the library and the command;
docs/licences.md is the table of datasets and checkpoints the index is
built from.

The zoo builds on the bundle (docs/bundle.md): like a bundle shipped as
a recipe, an index entry carries each file's size and digests and the
address it is fetched from, and nothing is handed out until every check
passes. The zoo adds a licence class per artifact, the licences of the
data the artifact was trained on, and an acceptance record per user and
licence, checked before any fetch.

## Licence classes

Every licence the zoo knows belongs to one class. An artifact's class is
the most restrictive of its own licence's class and the classes of the
licences of every dataset it was trained on. A model published under
Apache-2.0 but trained on a CC BY-SA treebank is `CLASS_SHARE_ALIKE`.

| Class | Licences | Bundled, checked in, shipped | Fetched |
|---|---|---|---|
| `CLASS_PERMISSIVE` | Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, CC0-1.0, LicenseRef-CC-BY-2.0-FR, CC-BY-2.5, CC-BY-3.0, CC-BY-3.0-US, CC-BY-4.0, ODC-By-1.0, PDDL-1.0, LicenseRef-PublicDomain | yes, with the attribution the licence asks for | yes |
| `CLASS_LGPL` | LGPL-2.1-only, LGPL-2.1-or-later, LGPL-3.0-only, LGPL-3.0-or-later | no, and not attributed to this project | after the user accepts the licence |
| `CLASS_SHARE_ALIKE` | CC-BY-SA-3.0, CC-BY-SA-4.0, ODbL-1.0 | no | after the user accepts the licence (decided 2026-10-14) |
| `CLASS_GPL` | GPL-2.0-only, GPL-2.0-or-later, GPL-3.0-only, GPL-3.0-or-later | no | yes; by default after the user accepts the licence |
| `CLASS_LOCAL_ONLY` | LicenseRef-RestrictedPublic, CC-BY-NC-4.0, CC-BY-NC-SA-3.0, CC-BY-NC-SA-4.0 | no, and no model trained on it is either | requested only at addresses under its original source (`origin`), after the user accepts the host's terms (restricted data) or the licence (non-commercial data), under every policy; by this project's policy neither the data nor any model trained on it is shipped, and such a model stays on the user's machine |
| `CLASS_REFUSED` | CC-BY-ND-4.0, CC-BY-NC-ND-4.0, LicenseRef-ResearchOnly, LicenseRef-PlatformTerms, LicenseRef-NoLicence, LicenseRef-Proprietary, LicenseRef-Unconfirmed | no | never; not listed |

The order, least to most restrictive, is permissive, LGPL, share-alike,
GPL, local-only, refused. LGPL sits below CC BY-SA because its copyleft
is weaker; GPL sits above it as the strongest copyleft the zoo fetches.
Under the default policy those three classes behave the same, so the
order between them only decides which class an artifact reports.
Local-only sits above them because nothing in it ever leaves the user's
machine, which also covers what the share-alike term of CC BY-NC-SA
asks, and a refused licence anywhere still wins over everything.

### Granite requires an explicit download decision

Kristian clarified #1451: users may download
`ibm-granite/granite-embedding-30m-english` at
`9b5b096411652ec1189c68fcfb90d0a82c5b45af` after accepting the notice.
TurboEmbed must not include the weights or generated bundles in releases
or download them automatically. The distribution rules above still apply.

The project inventory assigns `CLASS_SHARE_ALIKE` due to CC BY-SA training
inputs. IBM licenses the weights under Apache-2.0. The recipe notice
also identifies web-scraped and IBM-internal text with no auditable grant.
The exception permits this named local download after user acceptance;
it grants no rights to training text and approves no other model.

`turbo-bundle fetch` and `make` require acceptance of the notice, even for
cached inputs. `turbo-fetch.json` saves the notice and acceptance. For a
model whose licence itself needs acceptance or whose recipe carries
`notice_terms` (Granite's model card README, MS MARCO's terms page), the
tool reads and writes the zoo's acceptance file
(`$TURBO_ZOO_HOME/acceptances.json`, for `$USER`): `notice_terms` is
recorded under `LicenseRef-RestrictedPublic` with that address as
`license_text`, so a requirement the user accepted before, with the tool,
`turbo-zoo accept` or the catalog service on the page, needs no
`--accept-terms` flag, and `--accept-terms` records the acceptance there. A
recipe notice with no single `notice_terms` address stays a per-run
`--accept-terms` decision. API clients
must set `accept_terms=true` only after an explicit user decision through
configuration or a prompt. The manifest identifies the weight licence;
the notice adds the project download restriction. The existing handling
of `deepset/roberta-base-squad2` is unchanged.

### Restricted but public data

`CLASS_LOCAL_ONLY` holds two kinds of data: restricted but public data,
described here, and non-commercial data, described in the next section.
Restricted but public data is data anyone can download from its
publisher but that carries terms of its own: IMDB (no stated licence),
Sentiment140 and the SemEval Twitter sets (Twitter's terms). The rule is
Kristian's, from his 2026-10-12 comment on #326:

- The zoo requests it only at addresses on the original host, after the
  user accepts that host's terms, so the user takes on the terms. The
  project never carries, mirrors or checks in the data. The index
  enforces this: an artifact of the class must name its `origin`, the
  publisher's own location, and every file's `url` must lie under it
  (the `.origin` row of the index table gives the rule; fetch rule 4
  says how redirects are treated).
- It is used for local training and evaluation only. Evaluation numbers
  may be recorded; the files are not.
- A model trained on it is in the class too, by the class rule: the
  most restrictive licence among its own and its training data's. Such
  a model stays with the user who trained it. It is never bundled,
  checked in, shipped, or listed as a shipped model
  (`Index::shippable`, `LicenceClass::may_redistribute`).
  `turbo_zoo::class_of` gives the class of a model trained locally from
  the licences of its data.

The licence is `LicenseRef-RestrictedPublic`. It names a kind of terms,
not one text, so its `license_terms` in the index is the host's terms
page, and the user accepts that page, not the licence as a whole:
accepting IMDB's terms never covers Sentiment140's.

Research-only data stays `CLASS_REFUSED`. An entry moves such data into
`CLASS_LOCAL_ONLY` only by naming `LicenseRef-RestrictedPublic` for it,
which is a decision recorded with the entry in docs/licences.md. Refused
data mixed into a local-only artifact keeps it refused.

### Non-commercial data

Data under CC BY-NC or CC BY-NC-SA, such as English GUM in Universal
Dependencies (CC BY-NC-SA 4.0), is `CLASS_LOCAL_ONLY` as well. The rule
is Kristian's, from 2026-10-14, for data the project cannot distribute.
It is this project's policy, stricter than the
licences, which allow sharing the data and adapted material for
non-commercial purposes:

- Our code reads such data when the user supplies the files. A format
  reader needs no acceptance.
- The zoo requests it only at addresses under its original source
  (`origin`, checked as for restricted data), after the user accepts the
  licence, under every policy.
- The project never ships, bundles, mirrors or checks in the data, or
  any model trained on it. A model a user trains on it is in the class
  by the class rule (`turbo_zoo::class_of`); by accepting, the user
  agrees to treat it as non-commercial and keep it on their machine. A
  recipe's model card makes no zoo entry for it
  (`turbo_train_recipes::card::publishable_class`).

What the user accepts is the licence's legal code (its `legal_url`, at
creativecommons.org), once per licence, as for CC BY-SA. That differs
from restricted data on purpose: a CC licence is one published text,
the same at every host, while `LicenseRef-RestrictedPublic` names a kind
of terms whose text is different at each host. The entry's
`license_terms` still records where the publisher states the licence
(for a UD treebank, its README at the release). `turbo-zoo accept` and
`turbo-zoo show` print the address of the text and two statements kept
apart: what the licence itself allows, its rights for non-commercial
purposes only (`LocalOnlyReason::terms`), and this project's policy,
which accepting agrees to (`LocalOnlyReason::policy`): the zoo requests
the data only at addresses under its original source, the project never
ships the data or any model trained on it, and the user treats any model
they train on it as non-commercial and keeps it on their machine.

Non-commercial terms with no-derivatives (CC BY-NC-ND 4.0) stay
`CLASS_REFUSED`, as CC BY-ND 4.0 does. That is this project's default
(2026-10-14), not what the licences require: CC BY-ND 4.0 and CC BY-NC-ND
4.0 allow producing adapted material (the latter for non-commercial
purposes) and forbid only sharing it, so they would allow a model
trained on such data and kept on the user's machine. Earlier versions
differ: CC BY-ND 3.0 grants no right to make adaptations (its section
3).

Licences are named by their SPDX identifiers, case and all. Terms SPDX
has no identifier for use `LicenseRef-`: `LicenseRef-Unconfirmed` is for
terms whose sources disagree (the Leipzig corpora), refused until they
are confirmed. A licence not in the table is an error, not a default.

GPL acceptance is a policy choice: #326 says "download only" and leaves
open whether the user must accept first. The default
(`GplAcceptance::Required`) asks, as for LGPL and CC BY-SA;
`GplAcceptance::NotRequired` fetches GPL artifacts without asking.

## Index

The index is JSON in one canonical form, as the bundle manifest is:
field names in snake_case, enum values as `KIND_*`, `COMPONENT_*`,
`FORMAT_*` and `CLASS_*` constants, licences as SPDX strings, sizes as
JSON numbers, lists rather than maps. A field the table marks required
is present even when it is empty; a field marked "no" may be left out.
Nothing else is accepted: no unknown field or constant, no number
written as a string, no field filled in from a default. The canonical
form is what serde_json's pretty printer writes (two-space indent,
fields in the table's order) with a final newline; the built-in index
is in that form and a test keeps it there.

```json
{
  "zoo_version": 1,
  "artifacts": [
    {
      "id": "symspell-frequency-dictionary-en-82-765",
      "kind": "KIND_LEXICON",
      "language": "en",
      "format": "FORMAT_TEXT",
      "version": "c239062ae02961df18ab7da1671d01b4388204e0",
      "license": "MIT",
      "license_terms": "https://github.com/wolfgarbe/SymSpell/blob/c239062ae02961df18ab7da1671d01b4388204e0/LICENSE",
      "trained_on": [
        {
          "name": "Google Books Ngram data",
          "license": "CC-BY-3.0",
          "license_terms": "https://storage.googleapis.com/books/ngrams/books/datasetsv2.html"
        }
      ],
      "class": "CLASS_PERMISSIVE",
      "attribution": "SymSpell frequency lists by Wolf Garbe at c239062ae02961df18ab7da1671d01b4388204e0 (MIT, https://github.com/wolfgarbe/SymSpell/blob/c239062ae02961df18ab7da1671d01b4388204e0/LICENSE), compiled as the intersection of Google Books Ngram data (CC-BY-3.0, https://storage.googleapis.com/books/ngrams/books/datasetsv2.html) with SCOWL (http://wordlist.aspell.net/).",
      "files": [
        {
          "path": "frequency_dictionary_en_82_765.txt",
          "url": "https://raw.githubusercontent.com/wolfgarbe/SymSpell/c239062ae02961df18ab7da1671d01b4388204e0/SymSpell/frequency_dictionary_en_82_765.txt",
          "size": 1332881,
          "sha256": "c604e1121e398ae7c7fbf777f11e0a0f2fa66eda932cb9fba1321466cf3acd7b",
          "sha512": "c89ad6785ba97e6179049e7033dc2b9c1f7d000982b24c12bfb66841c7b5344a96b7b2d6bbfa8c679bf081ffdb1da9a41fa50c2a9bf7d7aef6a02181d48c04e7"
        }
      ]
    }
  ]
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `zoo_version` | uint32 | yes | 1. Anything else is rejected. |
| `artifacts[].id` | string | yes | Unique. Lowercase ASCII letters, digits, `.`, `_`, `-`, starting with a letter or digit, at most 128 bytes. It names the artifact's cache directory. For a published file it is the file name without its extension. |
| `.type` | enum | no | Which schema the entry follows beside the fields every entry has ("Model types", below): `TYPE_ANALYSIS_ARTIFACT` or `TYPE_ENCODER_BUNDLE`. An entry without the field is an analysis artifact, and the canonical form leaves that default out. |
| `.kind` | enum | yes | `KIND_MODEL`, `KIND_HEAD` (a classifier head over an encoder, #321), `KIND_LEXICON` (#320), `KIND_CENTROIDS` (a label centroid set), `KIND_DATASET`. |
| `.component` | enum | no | The component a model serves: `COMPONENT_LEMMATIZER`, `COMPONENT_TOKENIZER`, `COMPONENT_SENTENCE_DETECTOR`, `COMPONENT_POS`, `COMPONENT_NAME_FINDER`, `COMPONENT_CHUNKER`, `COMPONENT_PARSER`, `COMPONENT_LANGUAGE_DETECTOR`, `COMPONENT_DOCUMENT_CATEGORIZER`, `COMPONENT_ENTITY_EXTRACTOR` (GLiNER), `COMPONENT_SCHEMA_EXTRACTOR` (GLiNER2), `COMPONENT_RELATION_EXTRACTOR` (GLiREL), `COMPONENT_EXTRACTIVE_QA`, `COMPONENT_ZERO_SHOT_CATEGORIZER` (an NLI cross-encoder), `COMPONENT_DEPENDENCY_PARSER`, `COMPONENT_COREFERENCE_RESOLVER` (mentions grouped into chains by an antecedent-ranking head), `COMPONENT_DENSE_ENCODER` (one pooled vector per text, the BGE v1.5 encoders), `COMPONENT_SPARSE_ENCODER` (SPLADE term weights), `COMPONENT_LATE_INTERACTION_ENCODER` (ColBERT token vectors), `COMPONENT_PII_DETECTOR` (personally identifiable information spans, `turbo:pii`) and `COMPONENT_ASPECT_SENTIMENT` (aspect spans with a polarity each). Every listed model names one, except a tokenizer fetched beside a model by id. |
| `.language` | string | no | A language tag: two or three lowercase letters, then optional `-` subtags of one to eight letters or digits. Absent for a multilingual artifact. |
| `.format` | enum | yes | `FORMAT_SAFETENSORS`, `FORMAT_CONLLU`, `FORMAT_TEXT`, `FORMAT_TSV`, `FORMAT_CSV`, `FORMAT_PARQUET`, `FORMAT_ZIP`, `FORMAT_TAR_BZ2`, `FORMAT_PYTORCH` (a `torch.save` checkpoint in its zip form, read without running its pickle), `FORMAT_SENTENCEPIECE` (an `spm.model` with the model's `config.json`), `FORMAT_HEAD` (a binary classifier head over an encoder, `TEVH`, `TIVH` or `THEAD001`), `FORMAT_BUNDLE` (a sealed bundle directory, docs/bundle.md, whatever format each of its files has), or a packaged `.bin` archive. |
| `.version` | string | yes | The publisher's version. Not empty. |
| `.license` | SPDX string | yes | The artifact's own licence. |
| `.license_terms` | https URL | yes | Where the artifact's publisher states that licence. |
| `.trained_on[]` | message | yes | What the artifact was trained or derived from; empty for an artifact made from no data (a dataset, a hand-written lexicon). |
| `.trained_on[].name` | string | yes | The dataset and release as its publisher names them. |
| `.trained_on[].license` | SPDX string | yes | The dataset's licence. |
| `.trained_on[].license_terms` | https URL | yes | Where the dataset's publisher states it, at the release used. |
| `.class` | enum | yes | The class `license` and every `trained_on[].license` give. Written out so a reader that does not know a licence still sees the class; a value that differs from the computed one rejects the index. |
| `.attribution` | string | when any licence asks for it | The copyright and attribution notice for the artifact and its training data. Required whenever `license` or any `trained_on[].license` requires attribution (`Licence::requires_attribution`, true for every licence except `CC0-1.0`, `PDDL-1.0`, `LicenseRef-PublicDomain`, `LicenseRef-RestrictedPublic`, and the `LicenseRef-` refusal terms); optional otherwise, and rejected if empty, whitespace-only, or containing control characters. Printed by `turbo-zoo show <id>`. For models trained by `turbo-train-recipes`, it equals the model card's `attribution` field (`<id>.card.json`). |
| `.origin` | https URL | for `CLASS_LOCAL_ONLY`, and only for it | The original publisher's own location: for a dataset the data's own place on its host (a treebank's repository path on `raw.githubusercontent.com`), for a model the model publisher's. Its path has at least one non-empty segment, so it is never a whole host (`https://raw.githubusercontent.com/` is rejected); a file at the root of a host is named as the origin by its own address. Every `files[].url` must lie under it: the same host, compared without regard to case, and a path that is the origin's or continues it after a `/`, so `.../UD_English-GUM` holds `.../UD_English-GUM/r2.16/...` but not `.../UD_English-GUM-copy/...`; a trailing `/` on the origin makes no difference. No user information, query or fragment in the origin. In its path and a file's: no `.` or `..` segment (plain or percent-encoded); no `\` or `;`, plain or percent-encoded (some servers read `\` as `/`, and strip a `;` with what follows it, so `..;` reads as `..`); no percent-encoded `/`, `%` or NUL (`%2f`, `%25`, `%00`, in either case); every `%` followed by two hex digits, so malformed escapes such as `%%32%65` and IIS's `%u002e` are rejected; and percent-encoded bytes that decode to valid UTF-8 (RFC 3629), so no overlong form (`%c0%ae`), surrogate, five- or six-byte form, byte from `%f5` up, or lone or missing continuation byte. A query or fragment on a file's address is not checked. The rule covers the addresses the index names; fetch rule 4 says how redirects are treated. |
| `.bundle` | message | for `TYPE_ENCODER_BUNDLE`, and only for it | What the bundle's manifest records, so a reader picks a bundle without fetching it. |
| `.bundle.model` | string | yes | The manifest's `model.id`. Not empty, no leading or trailing whitespace, no control character. |
| `.bundle.revision` | string | yes | The manifest's `model.revision`, under the same rule. |
| `.bundle.task` | enum | yes | The manifest's `task`: `TASK_EMBED`, `TASK_RERANK` or `TASK_EXTRACTIVE_QA`. |
| `.files[]` | message | yes | At least one. |
| `.files[].path` | path | yes | Where the file goes in the artifact's directory: relative, `/` separated, no empty, `.` or `..` component, no backslash, no control character. Unique in the artifact. |
| `.files[].url` | https URL | yes | Where it is fetched from. The address must serve the same bytes for as long as the entry exists: a release archive (`archive.apache.org/dist`, not the mirror network, which drops old releases), a repository path at a commit, or an asset of a release of this repository (the models the recipes train, `analysis/train-recipes`). A release asset can be deleted or replaced by the repository's maintainers, so the models releases are made once and their assets are never replaced; a replaced file fails the digests below. |
| `.files[].also[]` | https URL | no | Further locations of the same bytes, tried in order after `url` (fetch rule 4), each held to the rule `url` is: a repository path at a commit, never a branch. Only on a `CLASS_PERMISSIVE` artifact: a file fetched after acceptance, or only under its `origin`, has one address, and a further location on it rejects the index. An address that repeats `url` or another entry rejects it too. For a file with a `download` entry a location serves either the compressed bytes or the file itself. Left out when there is none. |
| `.files[].size` | uint64 | yes | Bytes. |
| `.files[].sha256` | 64 lowercase hex | yes | Checked as the bundle tool checks files. |
| `.files[].sha512` | 128 lowercase hex | yes | Checked against the published 512-bit digest. |
| `.files[].download` | message | for a file served compressed | The file as the `url` serves it. `size`, `sha256` and `sha512` above stay the unpacked file's, which is what the cache holds. |
| `.files[].download.compression` | `gzip` or `zstd` | yes | `gzip` is one RFC 1952 member, whose trailer CRC-32 and length must match; `zstd` is one or more RFC 8878 frames. |
| `.files[].download.download_bytes` | uint64 | yes | The compressed bytes' size. |
| `.files[].download.download_sha256` | 64 lowercase hex | yes | The compressed bytes' SHA-256. |

Every field of type https URL (`.license_terms`,
`.trained_on[].license_terms`, `.origin`, `.files[].url` and
`.files[].also[]` here, and
`accepted[].license_text` in the acceptance file) is an `https://`
address that names a host, in printable ASCII with no space and no
backslash. `https:///x`, `https://user@/x` and `https://:443/x` name no
host and are rejected.

An artifact whose class is `CLASS_REFUSED` is set aside when the index
is read: it is not listed, and asking for it is an error naming the
refused licences. The built-in index holds none; a test keeps it so.
Refused artifacts are listed in docs/licences.md instead, with the
reason.

## Model types

Every entry has the fields above: an id, files pinned by size, SHA-256
and SHA-512, the licences and the class they give. `type` names the
schema an entry follows beside them. A reader that does not know a type
still has every shared field, and this build rejects a type it does not
know.

| Type | Entries | Beside the shared fields |
|---|---|---|
| `TYPE_ANALYSIS_ARTIFACT` | The models, heads, lexicons, centroid sets and datasets the analysis add-ons read. The type of an entry with no `type`. | Nothing. `bundle` is rejected. |
| `TYPE_ENCODER_BUNDLE` | A sealed encoder bundle (docs/bundle.md): an embedding model, a reranker or an extractive QA model as the core loads it. | `bundle`, required. `kind` is `KIND_MODEL`, `format` is `FORMAT_BUNDLE`, and `files` lists `manifest.json` and every file the manifest names, each at its path in the bundle directory. |

An encoder bundle is pinned twice. The index pins every file, the
manifest among them, and the manifest pins every other file by size and
SHA-256. Once the files are fetched and verified (fetch rule 6) the
manifest must describe the entry: its `model.id`, `model.revision` and
`task` are the entry's `bundle`, its `model.license` is the entry's
`license`, every file it names is listed in the entry with that size and
SHA-256, and the entry lists no file but the manifest that it does not
name. The entry's class comes from its licences by the table above, as
for any artifact, so a bundle of a model trained on restricted data is
`CLASS_LOCAL_ONLY` and fetched only under its `origin`. The artifact's
directory in the cache is the bundle directory.

### Importing a sealed bundle

The zoo reads only its own index. A model that exists in another layout
gets an entry through an importer (`turbo_zoo::import`), which reads the
model, verifies it, and writes the entry; nothing is taken on trust from
the layout it came in.

`turbo-zoo import-bundle <bundle-dir>` (`import::bundle`) writes the
`TYPE_ENCODER_BUNDLE` entry of a sealed bundle directory, one the bundle
tool made or sealed. It reads `manifest.json` and every file the
manifest names, holds each to the size and SHA-256 the manifest pins,
and records its SHA-512. The model id, revision, task and licence are
the manifest's; `version` is the revision. The flags give what a bundle
does not say of itself:

| Flag | Entry field |
|---|---|
| `--id <id>` | `id`. Required. |
| `--base <https url>` | Where the directory is published: each file's `url` is `<base>/<path>`. Required. |
| `--license-terms <https url>` | `license_terms`. Required. |
| `--trained-on <name> <licence> <https url>` | One `trained_on` entry; repeat it for each dataset. The entry's class is the most restrictive of the manifest's licence and these. |
| `--attribution <text>` | `attribution`, required whenever a licence asks for it. |
| `--origin <https url>` | `origin`, for a class of `CLASS_LOCAL_ONLY`; `--base` must lie under it. |
| `--component <COMPONENT_*>` | `component`. Without it an embedding bundle is `COMPONENT_DENSE_ENCODER`, an extractive QA bundle `COMPONENT_EXTRACTIVE_QA`, and a reranker names none. |
| `--language <tag>` | `language`. |
| `--into <zoo.json>` | Adds the entry to that index file, last or in the place of the entry of the same id, and writes the file in the canonical form through a temporary name. Every other entry of the file is kept, one the zoo refuses and does not list among them, and a file that does not load as an index is left as it is. A file that is absent is made holding the one entry. Without the flag the entry is printed as an index of its own. |

A file that is absent, a symlink, under a directory of the bundle that
is a symlink, or not the bytes its manifest pins refuses the import (no
byte is read from outside the bundle directory), and so does an entry the index would not load: a
class that needs an `origin` and has none, a missing attribution, a
refused licence. `turbo-zoo index` prints the index the command reads
(the built-in one, or `--index <zoo.json>`) in the canonical form, with
any entry the zoo refuses, to
start an index file that also lists the built-in entries.

### Importing from Hugging Face

`turbo-zoo import-hf <owner/name> <commit>` (`import::hugging_face`)
writes the entry of a Hugging Face model repository at one commit. The
commit is 40 lowercase hex digits; a branch or a tag may move and is
refused before anything is asked. The importer reads the repository's
own listing of that commit
(`/api/models/<owner>/<name>/tree/<commit>?recursive=true`), fetches
each file at `https://huggingface.co/<owner>/<name>/resolve/<commit>/<path>`,
which becomes its `url`, and refuses a file that is not the size the
listing gives or, for a file in large file storage, not the SHA-256 the
repository records for it. The entry pins the size, SHA-256 and SHA-512
of the bytes read. A listing of 1000 entries or more is refused, since
it may have been cut, and so is a path that an address would not carry
as it is.

| Flag | Entry field |
|---|---|
| `--id`, `--license-terms`, `--trained-on`, `--attribution`, `--origin`, `--component`, `--language`, `--into` | As for `import-bundle`. |
| `--license <SPDX id>` | `license`. Required: the importer does not read a model card for it. |
| `--kind <KIND_*>`, `--format <FORMAT_*>` | `kind` and `format` of an analysis artifact. Required without `--bundle`. |
| `--version <text>` | `version`; the commit when left out. |
| `--file <path>` | One file of the repository to list; repeat it, in the entry's order. Without it every file of the commit is listed, in the listing's order. |
| `--bundle` | The repository holds a sealed bundle directory: the entry is a `TYPE_ENCODER_BUNDLE` one, with `kind`, `format`, `version` and `bundle` from `manifest.json`, checked against the entry as fetch rule 6 does. `--license` must be the manifest's. |

A model whose class is `CLASS_LOCAL_ONLY` takes
`--origin https://huggingface.co/<owner>/<name>`, which every file's
address lies under.

### Importing an OCI artifact

`turbo-zoo import-oci <registry>/<name>@sha256:<digest>` (`import::oci`)
writes the entry of an OCI artifact in a registry, such as one `oras
push` made from a model's files. The reference names the manifest by
its digest; a tag may move and is refused before anything is asked. The
manifest is held to that digest and each layer to the size and SHA-256
digest the manifest gives it. A layer is one file: its `path` is the
layer's `org.opencontainers.image.title` annotation and its `url` is the
blob's own address, `https://<registry>/v2/<name>/blobs/sha256:<hex>`,
which names its bytes. The flags are `import-hf`'s without `--file`;
`version` is `sha256:<digest>` when left out, and `--bundle` reads a
sealed bundle directory from the layers.

Docker Hub's registry is `registry-1.docker.io`. A reference that names
`docker.io` or `index.docker.io`, as image references spell it, is read
as that host, and the entry's addresses name it.

An image index, a layer that names no file (a container image's file
system layer, so a model packed into an image's file system is not
imported), a layer that is a directory packed as one archive (what
`oras push` makes of a directory; push each file as a layer of its
own), and a digest that is not SHA-256 are refused.

`OciRegistry` (the `http` feature) is the registry client and a
`Transport`: it reads manifests and blobs by digest without
credentials, answers a registry's `WWW-Authenticate: Bearer` challenge
once per repository with an anonymous token from the service the
challenge names, and follows a redirect to the registry's storage
without the token, over HTTPS only. Each redirect is followed one hop at a time, and a manifest
request that any hop sends on to another host is reported as that
redirect and is not followed.
Most registries ask for such a
token for public repositories too, so a file at a blob address is
fetched through it: `turbo-zoo fetch` asks the address plainly and then
through `OciRegistry`. `OciRegistry::with` sends a registry's requests
to another address: an `https://` one, such as a pull-through mirror,
or `http://` on this machine alone (`127.0.0.1` or `localhost`), and
refuses any other.
`analysis/zoo/tests/import_oci.rs` runs the importer against an
in-process registry that issues the challenge, the token and the
redirect. The manifest and the challenge Docker Hub itself
serves for the published model are kept as recorded
(`analysis/zoo/testdata/recorded/`, the manifest held to its digest),
and unit tests read them.

### What an import keeps

`import-hf` and `import-oci` read every file to pin it, and keep each
one in the cache as the entry's file (`<cache>/<id>/<path>`, the
directory `fetch` uses), so the `fetch` that follows downloads nothing
more. Run again, the same import takes from the cache every file whose
SHA-256 its source names beforehand: every layer of an OCI artifact, and
each file a Hugging Face repository keeps in large file storage. A kept
file is held to that size and SHA-256 before it is used, and one that
holds other bytes is downloaded again and replaced. A Hugging Face file
outside large file storage has no digest in the listing, so it is read
from the repository each time. `--no-keep` imports without keeping
anything. In the library the importers take the cache directory as
`keep` (`RepoImport::keep`, `OciImport::keep`), and keep nothing
without it.

The two network tests keep their downloads in the directory
`TURBO_TEST_ZOO_CACHE` names when it is set, and leave it in place, so
a later run on the same machine does not download the files again.

A zoo server lists the imported bundles when it is started with that
file (`--catalog-index`, docs/grpc-catalog.md), and serves their files
from its store or from a directory that holds the bundle at
`<root>/<id>/` (`Directory`), each verified against the entry.

## Acceptance file

Which user accepted which licence, in the zoo's home directory as
`acceptances.json`. Strict JSON in canonical form, like the index:

```json
{
  "acceptances_version": 1,
  "accepted": [
    {
      "user": "alice",
      "license": "CC-BY-SA-4.0",
      "license_text": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
      "accepted_at": 1792000000
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `acceptances_version` | uint32 | 1, or 2 when an acceptance holds `artifact` or `license_text_sha256`. The file is written at the lowest version that holds it, so a reader of version 1 reads every file made without a page. Both versions read. |
| `accepted[].user` | string | Who accepted: 1 to 256 bytes, no control characters, no leading or trailing whitespace. |
| `accepted[].license` | SPDX string | A licence of `CLASS_LGPL`, `CLASS_SHARE_ALIKE`, `CLASS_GPL` or `CLASS_LOCAL_ONLY`. A permissive licence needs no acceptance and a refused one cannot be accepted; either rejects the file. |
| `accepted[].license_text` | https URL | The text the user was shown. For a licence with one text, the zoo's address for it; for `LicenseRef-RestrictedPublic`, the host's terms page as the index gives it (`license_terms`), so each host's terms are accepted on their own. |
| `accepted[].accepted_at` | uint64 | Seconds since the Unix epoch, UTC. |
| `accepted[].artifact` | artifact id | Version 2, optional: the artifact whose page showed the terms (the demo's Models tab, through `RecordLicenceAcceptance`). For audit only; the acceptance covers every artifact under the same terms. |
| `accepted[].license_text_sha256` | SHA-256 | Version 2, optional: the SHA-256 of `license_text` as the page displayed it. A digest of other text rejects the file. |

Every field of version 1 is required; a version 1 file with a version 2
field is rejected. One user accepts one licence and text at most
once. An absent file means nothing has been accepted. The zoo never
writes the file on its own: only the user's explicit action adds an
entry. `turbo-zoo accept <licence>` (or `turbo-zoo accept
LicenseRef-RestrictedPublic <terms>`, with the address `turbo-zoo show`
prints) prints the address of the text, and for local-only terms,
restricted or non-commercial, what they allow, and records nothing; the
same command with `--yes`, run after reading it, records the
acceptance. A permissive or refused licence is not offered: the command
says why and records nothing. `turbo-zoo revoke` removes one, and names
a permissive or refused licence the same way.

## Fetch rules

For `fetch(id, user)`:

1. An id the index does not have: `NotFound`. A refused artifact:
   `Refused`, naming each refused licence and why.
2. Every distinct licence among the artifact's own and its training
   data's whose class needs acceptance under the policy must be
   accepted by `user`, with its text: the licence's own address, or for
   `LicenseRef-RestrictedPublic` the `license_terms` next to it.
   Otherwise `NotAccepted`, naming the class, the user and each missing
   licence with its text's address. Nothing is read, fetched or
   written. The `CLASS_LOCAL_ONLY` check of step 2 does not depend on
   the GPL policy. The acceptance file is read only when the artifact
   needs some acceptance (`Zoo::fetch_with`), so a malformed or stale
   file never blocks a permissive artifact; when it is read, an
   unusable entry is reported with the file's path and the entry's
   position (`accepted[3]`).
3. Each file goes to `<cache>/<id>/<path>`. The directories below the
   cache are checked one at a time, from the top, before anything is
   made below them: a symlink or a non-directory on the way, or a
   symlink at the file's own path, is an error, so the zoo never
   creates, reads or writes anything elsewhere on disk.
4. A file already in the cache is read and checked by size, SHA-256 and
   SHA-512 on every call. One that is absent or fails is fetched from
   its `url`. A file that lists further locations (`also`) is asked for
   at `url` first and then at each of them in order: a location that
   fails, or serves bytes that are not the file by the checks of step
   5, passes the request to the next, and the first whose bytes are the
   file is the one kept. When none serves it the error names what each
   location said, and nothing is written. The HTTP transport follows the redirects the server
   issues, up to 10, over HTTPS only: a redirect to an `http://` address
   is an error. Where a redirect leads is not checked against the index
   or the `origin` (a GitHub release redirects to another host), so the
   `origin` rule covers the addresses the index names, not every host a
   download passes through. Step 5 checks the bytes, so they are the
   ones the index records wherever a redirect leads.
5. Fetched bytes are checked by size, then SHA-256, then SHA-512, before
   they are written. The first failure is `Integrity`, naming the
   artifact, the path, the check and both values; nothing is written.
   Bytes that pass are written through a temporary name beside the
   file that is this write's own (`<path>.partial.<pid>.<n>`), created
   anew, never opened through a link, and renamed onto the file; it is
   removed if the write fails. Fetches of one artifact in several
   threads or processes therefore never touch each other's bytes, and
   each rename puts a whole, verified file in place. The acceptance file
   is saved the same way. The transport is told the expected size; the HTTP transport reads at
   most one byte more, and asks for the bytes unencoded
   (`Accept-Encoding: identity`). For a file with a `download` entry,
   the transport is told the larger of `download_bytes` and `size`; the
   fetched bytes are checked by `download_bytes` and `download_sha256`
   (`Check::DownloadSize`, `Check::DownloadSha256`), unpacked to exactly
   `size` bytes (`Check::Unpack` when they do not), and the unpacked
   bytes are then checked by size, SHA-256 and SHA-512 as any other
   file before they are written. Fetched bytes that fail the wire check
   but verify as the file itself are kept: a server encoded the file
   anyway (`Content-Encoding: gzip`) and the transport decoded it. The
   zoo's own HTTP transport is built without that decoding.

6. For a `TYPE_ENCODER_BUNDLE` entry the cached `manifest.json`, verified
   in steps 4 and 5 as the entry's file, is read and checked against the
   entry ("Model types", above). A manifest that describes another
   model, or names other files than the entry lists, is
   `ZooError::Bundle` and the artifact is not handed out.

Step 2 runs on every call, cached files or not, so a revoked acceptance
takes effect at once. There is no hash cache, as in the bundle loader.

`Zoo::ensure` (and `Zoo::ensure_with`) wraps `Zoo::fetch_with` for
redistributable models and heads (#1726): it enforces `Acceptances` for
`CLASS_LGPL`, `CLASS_SHARE_ALIKE`, and `CLASS_GPL` and refuses
`CLASS_LOCAL_ONLY` with `ZooError::LocalOnly` before touching the
transport. `Index::find(language, component)` orders matching entries
from least to most restrictive licence class (`CLASS_PERMISSIVE` first,
in index order within a class) so callers default to a permissive model
or head when several serve the same language and component. Beside
`Blobs` and `Http`, `Directory` serves verified bundle files from a
local directory root (`<root>/<id>/<path>` or `<root>/<path>`).

## Self-hosted mirrors

An artifact file's `url` is any `https://` address that serves the
recorded bytes for as long as the entry exists; the fetch checks in step
4 above make the transport untrusted. A self-hosted store needs no zoo
changes when it serves plain HTTPS GETs: an S3 bucket with public-read
objects (or presigned URLs) works as is, with object keys named so they
never change bytes (`<artifact>@<commit>/<path>`, or the file's SHA-256
in the key; never a mutable `latest`).

Two knobs in `turbo_zoo` cover the cases plain HTTPS does not:

- A mirror served with a publicly trusted certificate (a wildcard
  certificate for the organization's domain, for example) needs no zoo
  knob at all: the default transport's public roots already trust it.
  Certificates and private keys live on the host only and are never
  committed; the tree test refuses a committed private key block.
- `Http::with_root_ca_bundle_pem` builds an HTTPS transport trusting
  only the PEM-encoded CA bundle it is given, for a mirror served with a
  private CA. Every `-----BEGIN CERTIFICATE-----` block in the bundle
  must parse. The public roots are not in the set; a deployment that
  fetches both mirrored and public artifacts keeps `Http::new()` for the
  public ones and the custom transport for the mirror.
- `Mirror::new(base, inner)` rewrites every fetch to
  `<base>/<original host>/<original path>`, so one mirror of several
  origins keeps them apart, and a file's further locations (`also`) are
  rewritten the same way. The base cannot include a query or fragment.
  Digest checks run unchanged on what the mirror serves.

An OCI registry also works: blobs are served by digest over HTTPS
(`/v2/<name>/blobs/sha256:<digest>`), so the address is self-pinning and
cross-checks the index's `sha256`. Bundles go up as OCI artifacts (ORAS)
when containers and models should share one pipeline; otherwise the S3
bucket is the simpler host.

`Zoo::cache_status(id)` reads and checks the cached files as step 4
does, and reports `Cached`, `Missing` (the first absent file) or
`Corrupt` (the first file that fails, and why), with no fetch, no write
and no directory made. An encoder bundle whose files all verify and
whose manifest does not describe its entry is `Corrupt` at
`manifest.json`, with the reason a fetch refuses it for, and a zoo
server does not list such a bundle as installed. It checks no licence and hands out no path:
`fetch` does that. `Home::from_env` finds the directory the `turbo-zoo`
command uses (`$TURBO_ZOO_HOME`, else `$XDG_DATA_HOME/turbo-zoo`, else
`~/.local/share/turbo-zoo`), with the cache at `Home::artifacts` and the
acceptance file at `Home::acceptances`; `UserId::from_env` reads `$USER`,
else `$USERNAME`.

### Other sources: a content-addressed store and a list of sources

The zoo asks its transport for each file with `Transport::get_file`,
which gets the artifact and the index's record of the file. By default
it fetches the file's address, as before; a transport that finds files
another way overrides it (#1613).

- `Blobs::new(root)` is a content-addressed store on disk: each file,
  unpacked, at `<root>/blobs/sha256/<hex>` of its SHA-256. The same
  layout serves a zoo server's store, a library's local copy and an S3
  bucket, so any one can seed another, and a key always names the bytes
  the index pins. `Blobs::put(bytes)` keeps bytes under their own
  SHA-256, written whole through a temporary name. As a transport it
  serves a file by its SHA-256 alone, never by address, and refuses a
  symlink or anything but a plain file.
- `Sources::new().then(name, source)...` tries its sources in order for
  each file and keeps the first answer that verifies by size, SHA-256
  and SHA-512 (after the wire check, for a file served compressed); a
  source that lacks the file, fails or serves other bytes passes the
  request on. When none has it, the error names what each said. A
  typical list is the local store, then a zoo server, then `Http`
  for the index's own address.
- `Sources::then_keep(name, store)` adds a store (anything that is a
  `Transport` and a `Keep`: `Blobs` or `S3Bucket`) that also keeps every
  file a later source serves, once it verifies, so the next read is
  local. A file it cannot keep is still handed over. The keeping store
  neither serves nor keeps an artifact whose class may not be
  redistributed (`CLASS_LOCAL_ONLY`, refused): such a file comes only
  from the sources after it, and no copy of it is made, on a server's
  store or on a user's own machine.
- `S3Bucket` (the `http` feature) is an S3 bucket or prefix, on AWS or
  an S3-compatible store such as RustFS, in the same
  `<prefix>/blobs/sha256/<hex>` layout: a file is read with one GET and
  kept with one PUT, each signed with AWS Signature Version 4 (computed
  with `turbo-sha256`, no AWS SDK), or unsigned for a public bucket.
  `S3Bucket::new("s3://bucket/prefix")` is configured the way the AWS
  tools are: `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL` for a store
  other than AWS (addressed path-style), `AWS_REGION` or
  `AWS_DEFAULT_REGION` (else `us-east-1`), and `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`.
  `S3Bucket::signed_in(address, &sign_in)` opens a holder's bucket: from
  that environment for `SignIn::Environment`, or with the holder's own
  `SignIn::S3Keys`, which read nothing from the environment. Redirects
  are not followed, so a signed request is not replayed elsewhere.
  `analysis/zoo/tests/s3.rs` runs it against an in-process stand-in that checks each
  signature; `analysis/zoo/src/fetch/sigv4.rs` checks HMAC-SHA-256 against RFC 4231
  and the signatures against the S3 documentation's GET and PUT Object
  examples.

- `GoogleDrive` (the `http` feature) is a Google Drive folder, read
  through the Drive API v3 over the same HTTPS stack (no SDK), shared
  drives included. `GoogleDrive::new("gdrive://<folder-id>[/name/...]")`
  names the folder by its Drive id (`root` is My Drive), and the names
  after the id walk down from it. A file is looked for at
  `blobs/sha256/<hex>`, the layout the other stores share, and when it
  is not there at `<artifact-id>/<path>`, the zoo cache's own layout,
  so a store or a cache copied to Drive serves as is. Each folder on the
  way is found with one `files.list` query and remembered; the file is
  the first of its name whose size is the index's (read page by page),
  so a file of another size is never downloaded, and is read with one
  `files.get?alt=media`, at most its size and one byte more. Sign-in comes from `GOOGLE_OAUTH_ACCESS_TOKEN`, sent as a bearer
  token (one from `gcloud auth print-access-token`, or any OAuth flow
  the caller runs), or else `GOOGLE_API_KEY`, enough for a folder
  shared with anyone who has the link (sent as `X-Goog-Api-Key`); one
  of them is required. Neither ever travels in a URL, and a redirect is
  followed over HTTPS only and unsigned. `GoogleDrive::with` takes
  another endpoint; a plain `http://` one sends the token in clear text
  and is only for a stand-in on loopback in tests. It
  reads and does not keep: a library puts its own `file://` store first
  to keep what Drive serves. `analysis/zoo/tests/drive.rs` runs it
  against an in-process stand-in that answers the Drive API calls.

A zoo server keeps its files in such a store with
`--catalog-store file:///dir` or `--catalog-store s3://bucket/prefix`,
or in a data directory several servers share with `--catalog-data DIR`
(docs/grpc-catalog.md, The data directory). That directory's `public/`
side is a store of its own holding only files that need no acceptance,
so a library with the directory mounted reads it directly, with no
server running. Mount it read-only, or name a local store before it,
`TURBO_ZOO_SOURCE=file:///local/store,file:///mount/public,origin`,
since the first `file://` store named keeps what later sources serve.
An accept-first file is never there and is read through a server.
Started with `--zoo`, a server fetches an artifact on
its first read, so a library pointed at it with `grpc(s)://` needs no
separate install step. docs/zoo-container.md packs that server, the
page and a RustFS store into one `docker compose`. `turbo-zoo-remote`
(top-level `zoo-remote/`) adds the library's remote sources: a zoo
server over `grpc://` or `grpcs://`, an S3 bucket over `s3://`, a
Google Drive folder over `gdrive://` (the `gdrive` feature), and the
`TURBO_ZOO_SOURCE` list that names them.

### Holders: sources a user registers

An application that lets each user bring a store keeps the user's
sources as holders (#1744) rather than in one process-wide
`TURBO_ZOO_SOURCE`. A holder (`turbo_zoo::Holder`) is a name the user
picks (1 to 64 characters, no comma or control character, unique per
user), a source address, and how it signs in (`SignIn`):

- `environment`: from the process environment, as `TURBO_ZOO_SOURCE`
  does, or not at all for a source that needs none;
- `google_oauth`: a Google grant, kept as the OAuth client id (and
  secret, for a client type that has one) and the grant's refresh token;
- `api_key`: a key, for a Drive folder shared by link;
- `s3_keys`: an S3 access key pair (`access_key_id`,
  `secret_access_key`, and `session_token` for temporary keys) that an
  `s3://` holder signs its requests with in place of the process
  `AWS_*` environment, with the bucket's `region` (else `us-east-1`)
  and, for an S3-compatible store, its `endpoint` (`http://host:port`
  or `https://host`, addressed path-style).

`Holders` keeps every user's holders, in the order they are tried, in
one strict JSON file (`Home::holders()`, `holders.json` beside the
acceptance file), one canonical form, unknown fields refused:

```json
{
  "holders_version": 1,
  "users": [
    {
      "user": "alice",
      "holders": [
        { "name": "laptop", "address": "file:///srv/zoo" },
        {
          "name": "team drive",
          "address": "gdrive://<folder-id>",
          "sign_in": { "kind": "google_oauth", "client_id": "...", "refresh_token": "..." }
        }
      ]
    }
  ]
}
```

The file holds sign-in secrets, so it is written through a temporary
name readable and writable by its owner alone on Unix. `register` adds a
holder last or replaces the one of the same name where it was; `remove`
and `move_to` change the list.

Signing a user in to Drive is `turbo_zoo::GoogleOAuth` (the `http`
feature): OAuth 2.0 authorization code with PKCE (RFC 7636, S256), scope
`drive.readonly`, offline so the grant carries a refresh token, over the
zoo's HTTPS stack. A web application sends the user to
`authorize_url(redirect, &pkce)` and calls `exchange(code, redirect,
&pkce)` from its redirect, after checking the `state` it got back; a
command line or desktop calls `sign_in_loopback`, which listens once on
`127.0.0.1`, hands the consent page's address to a callback that opens
a browser, and checks the `state` itself. Only the refresh token is
kept. `DriveAuth::Refresh(Renewing)` gets an access token from it on
first use, again a minute before each runs out, and once more when Drive
answers 401. `analysis/zoo/tests/oauth.rs` runs all of it against an
in-process server that answers both the token endpoint and the Drive
API.

Holders open into sources through `turbo_zoo_remote::Registry`, the
kinds of source by scheme. `Registry::builtin()` has `file`, `grpc` and
`grpcs`, `s3` and `gdrive` (a scheme whose feature the build leaves out
is still known, and refused when opened); an application adds a kind of
its own by implementing `Kind` (the schemes it claims, and how it opens
a holder into a transport or a keeping store) and calling `register`,
which also lets it take a built-in scheme over.
`Registry::sources(holders.list(&user), &user, Some(origin))` is the
user's list, in order, then the index's own addresses; the first store in it keeps what the
sources after it serve, as the first `file://` store of
`TURBO_ZOO_SOURCE` does. `TURBO_ZOO_SOURCE` itself is parsed through
the same registry (`Registry::spec`), each entry a holder that signs in
from the environment. A `file` or `grpc` holder signs in from the
environment only; an `s3` holder from the environment or with its own
`s3_keys`; a `gdrive` holder from the environment, with an API key or
with a Google grant.

Whatever the source, the licence check runs first: no source is asked
for an artifact whose terms the user has not accepted.

### The Hugging Face copies of the published models

The permissive models this project trains and publishes are release
assets of this repository (`url`) and are also kept on Hugging Face, one
model repository per artifact named by its id under `pipestream-ai`,
holding each indexed file at its `path`. Each such file lists
`https://huggingface.co/pipestream-ai/<id>/resolve/<commit>/<path>` in
`also`, at the 40-digit commit that holds those bytes, so the artifact
installs when either host is unreachable. `turbo-zoo show <id>` prints
every location. `analysis/zoo/tests/builtin/hugging_face.rs` keeps the
list, and its ignored test installs each model from Hugging Face alone
(`cargo test -p turbo-zoo -- --ignored hugging_face`).

## Reading the index from external clients

The index is structured so external tools can query it without scraping HTML
directories:

- The JSON uses only objects, arrays, strings and non-negative
  integers: no floats, no booleans, no nulls, no maps keyed by data.
  Any JSON reader, or a small hand-written one, reads it.
- `component` names component constants with a `COMPONENT_` prefix,
  and `language` is the ISO code the index keys its map by, so
  `(language, component)` resolves the matching models directly. `Index::find`
  is that lookup here; `Index::find_within` keeps the models of a class or less
  restrictive ones, the least restrictive first.
- `sha512` is the digest in the form the published `.sha512` files hold, so a
  client checks a file against the index without fetching a separate checksum
  file.
- `class` is written out, so a reader that does not map licences itself
  can still refuse `CLASS_REFUSED`, keep `CLASS_LOCAL_ONLY` to the
  user's machine, and ask before the other non-permissive classes.

## Decisions taken with the design

- The training data's licences are recorded beside the artifact's own,
  not folded into it, so the reason for a class is visible and a change
  in policy for one licence re-classifies every artifact trained under
  it.
- Acceptance is per licence, not per artifact: accepting CC-BY-SA-4.0
  once covers every artifact whose only restrictive licence it is. An
  artifact with two restrictive licences needs both. The exception is
  `LicenseRef-RestrictedPublic`, accepted per host's terms page.
- Restricted data is a class of its own rather than share-alike or
  refused because it differs from both: it may be fetched (unlike
  refused data), but nothing made from it may leave the user's machine
  (unlike share-alike data, whose derived models may be shared under
  the same licence).
- Non-commercial data shares that class rather than having one of its
  own because this project applies the same rules to it: requested only
  at addresses under its origin, after acceptance, and nothing made from
  it leaves the user's machine. `LocalOnlyReason` says which kind of
  terms put a licence there, which decides what the user accepts and is
  told.
- Both digests are required. SHA-256 is what the bundle tool and the
  repository's model READMEs use; SHA-512 is what upstream Apache
  distributions publish.
- The HTTP transport is a cargo feature (`http`, on by default). Without
  it the library still parses, checks licences and verifies files a
  caller's own transport fetched; `Offline` serves the cache alone.

## Left out

- Signatures. Hashes catch corruption; signing belongs to distribution.
- Encoder bundles in the built-in index. `TYPE_ENCODER_BUNDLE` is the
  schema of one ("Model types", above), and an index given to the zoo
  may list them; the built-in index lists none. The bundle tool's
  recipes and catalogue (docs/bundle.md) make a bundle from its upstream
  files.
- Further locations for anything but a permissive artifact. A file of
  any other class has one address (`.files[].also[]`); a whole-index
  mirror rewrites every address to one host with `Mirror`
  ("Self-hosted mirrors", above).
