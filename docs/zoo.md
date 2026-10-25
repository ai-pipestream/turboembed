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
is Kristian's, from 2026-10-14, and follows what OpenNLP does with data
it cannot distribute. It is this project's policy, stricter than the
licences, which allow sharing the data and adapted material for
non-commercial purposes:

- Our code reads such data when the user supplies the files. A format
  reader needs no acceptance, as OpenNLP's need none.
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
      "id": "opennlp-en-ud-ewt-sentence-1.3-2.5.4",
      "kind": "KIND_MODEL",
      "component": "COMPONENT_SENTENCE_DETECTOR",
      "language": "en",
      "format": "FORMAT_OPENNLP_MODEL",
      "version": "1.3-2.5.4",
      "license": "Apache-2.0",
      "license_terms": "https://archive.apache.org/dist/opennlp/models/ud-models-1.3/README",
      "trained_on": [
        {
          "name": "UD_English-EWT r2.16",
          "license": "CC-BY-SA-4.0",
          "license_terms": "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md"
        }
      ],
      "class": "CLASS_SHARE_ALIKE",
      "attribution": "Apache OpenNLP 1.3 UD model (Apache-2.0, The Apache Software Foundation), trained on UD_English-EWT r2.16 (CC-BY-SA-4.0, https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md).",
      "files": [
        {
          "path": "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
          "url": "https://archive.apache.org/dist/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
          "size": 20061,
          "sha256": "4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8",
          "sha512": "fed5d53ea87bf66afeb08c084f89ca9b95a96de0920dca40487b2ab9972ee6b7ef211e477db480b7253431de079fd66dfd14329cd7d19ad1f565155eb0cf5191"
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
| `.kind` | enum | yes | `KIND_MODEL`, `KIND_HEAD` (a classifier head over an encoder, #321), `KIND_LEXICON` (#320), `KIND_CENTROIDS` (a label centroid set), `KIND_DATASET`. |
| `.component` | enum | no | The component a model serves. `COMPONENT_LEMMATIZER`, `COMPONENT_TOKENIZER`, `COMPONENT_SENTENCE_DETECTOR`, `COMPONENT_POS`, `COMPONENT_POS_GENERIC`, `COMPONENT_NAME_FINDER`, `COMPONENT_CHUNKER` and `COMPONENT_PARSER` are OpenNLP's `ModelType` constants; `COMPONENT_LANGUAGE_DETECTOR`, `COMPONENT_DOCUMENT_CATEGORIZER`, `COMPONENT_ENTITY_EXTRACTOR` (GLiNER), `COMPONENT_SCHEMA_EXTRACTOR` (GLiNER2), `COMPONENT_RELATION_EXTRACTOR` (GLiREL), `COMPONENT_EXTRACTIVE_QA` and `COMPONENT_ZERO_SHOT_CATEGORIZER` (an NLI cross-encoder) have no `ModelType`. A tokenizer fetched beside a model by id has none. |
| `.language` | string | no | A language tag: two or three lowercase letters, then optional `-` subtags of one to eight letters or digits. Absent for a multilingual artifact. |
| `.format` | enum | yes | `FORMAT_OPENNLP_MODEL` (an OpenNLP `.bin` archive), `FORMAT_SAFETENSORS`, `FORMAT_CONLLU`, `FORMAT_TEXT`, `FORMAT_TSV`, `FORMAT_CSV`, `FORMAT_PARQUET`, `FORMAT_ZIP`, `FORMAT_TAR_BZ2`, `FORMAT_PYTORCH` (a `torch.save` checkpoint in its zip form, read without running its pickle), `FORMAT_SENTENCEPIECE` (an `spm.model` with the model's `config.json`). |
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
| `.files[]` | message | yes | At least one. |
| `.files[].path` | path | yes | Where the file goes in the artifact's directory: relative, `/` separated, no empty, `.` or `..` component, no backslash, no control character. Unique in the artifact. |
| `.files[].url` | https URL | yes | Where it is fetched from. The address must serve the same bytes for as long as the entry exists: a release archive (`archive.apache.org/dist`, not the mirror network, which drops old releases), a repository path at a commit, or an asset of a release of this repository (the models the recipes train, `analysis/train-recipes`). A release asset can be deleted or replaced by the repository's maintainers, so the models releases are made once and their assets are never replaced; a replaced file fails the digests below. |
| `.files[].size` | uint64 | yes | Bytes. |
| `.files[].sha256` | 64 lowercase hex | yes | Checked as the bundle tool checks files. |
| `.files[].sha512` | 128 lowercase hex | yes | Checked as OpenNLP's `DownloadUtil` checks files. |
| `.files[].download` | message | for a file served compressed | The file as the `url` serves it. `size`, `sha256` and `sha512` above stay the unpacked file's, which is what the cache holds. |
| `.files[].download.compression` | `gzip` or `zstd` | yes | `gzip` is one RFC 1952 member, whose trailer CRC-32 and length must match; `zstd` is one or more RFC 8878 frames. |
| `.files[].download.download_bytes` | uint64 | yes | The compressed bytes' size. |
| `.files[].download.download_sha256` | 64 lowercase hex | yes | The compressed bytes' SHA-256. |

Every field of type https URL (`.license_terms`,
`.trained_on[].license_terms`, `.origin` and `.files[].url` here, and
`accepted[].license_text` in the acceptance file) is an `https://`
address that names a host, in printable ASCII with no space and no
backslash. `https:///x`, `https://user@/x` and `https://:443/x` name no
host and are rejected.

An artifact whose class is `CLASS_REFUSED` is set aside when the index
is read: it is not listed, and asking for it is an error naming the
refused licences. The built-in index holds none; a test keeps it so.
Refused artifacts are listed in docs/licences.md instead, with the
reason.

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
| `acceptances_version` | uint32 | 1. |
| `accepted[].user` | string | Who accepted: 1 to 256 bytes, no control characters, no leading or trailing whitespace. |
| `accepted[].license` | SPDX string | A licence of `CLASS_LGPL`, `CLASS_SHARE_ALIKE`, `CLASS_GPL` or `CLASS_LOCAL_ONLY`. A permissive licence needs no acceptance and a refused one cannot be accepted; either rejects the file. |
| `accepted[].license_text` | https URL | The text the user was shown. For a licence with one text, the zoo's address for it; for `LicenseRef-RestrictedPublic`, the host's terms page as the index gives it (`license_terms`), so each host's terms are accepted on their own. |
| `accepted[].accepted_at` | uint64 | Seconds since the Unix epoch, UTC. |

Every field is required. One user accepts one licence and text at most
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
   its `url`. The HTTP transport follows the redirects the server
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

Step 2 runs on every call, cached files or not, so a revoked acceptance
takes effect at once. There is no hash cache, as in the bundle loader.

`Zoo::cache_status(id)` reads and checks the cached files as step 4
does, and reports `Cached`, `Missing` (the first absent file) or
`Corrupt` (the first file that fails, and why), with no fetch, no write
and no directory made. It checks no licence and hands out no path:
`fetch` does that. `Home::from_env` finds the directory the `turbo-zoo`
command uses (`$TURBO_ZOO_HOME`, else `$XDG_DATA_HOME/turbo-zoo`, else
`~/.local/share/turbo-zoo`), with the cache at `Home::artifacts` and the
acceptance file at `Home::acceptances`; `UserId::from_env` reads `$USER`,
else `$USERNAME`.

## Reading the index from OpenNLP

OpenNLP's `DownloadUtil` (3.x) finds models by scraping the HTML index
of the Apache mirror for `.bin` links, maps each to a language by a
substring of its name and to a `ModelType` by another, and checks a
`.sha512` file fetched beside it. It knows nothing of licences: it
fetches the UD models trained on non-commercial treebanks as readily as
the others, and from the mirror network, which carries only current
releases.

The index is meant to replace that page for OpenNLP 3.1 too:

- The JSON uses only objects, arrays, strings and non-negative
  integers: no floats, no booleans, no nulls, no maps keyed by data.
  Any JSON reader, or a small hand-written one, reads it.
- `component` names `ModelType` constants with a `COMPONENT_` prefix,
  and `language` is the code `DownloadUtil` keys its map by, so
  `(language, component)` gives the same lookup `downloadModel(language,
  modelType, type)` does. `Index::find` is that lookup here;
  `Index::find_within` keeps the models of a class or less restrictive
  ones, the least restrictive first.
- `sha512` is the digest `DownloadUtil.verifyChecksum` compares, in the
  form the published `.sha512` files hold, so OpenNLP checks a file
  against the index without fetching a checksum file.
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
  repository's model READMEs use; SHA-512 is what OpenNLP and the Apache
  distribution publish.
- The HTTP transport is a cargo feature (`http`, on by default). Without
  it the library still parses, checks licences and verifies files a
  caller's own transport fetched; `Offline` serves the cache alone.

## Left out

- Signatures. Hashes catch corruption; signing belongs to distribution.
- Encoder bundles. They have their own recipes and catalogue
  (docs/bundle.md); their `model.license` maps to a class by the same
  table when the bundle tool learns to check it.
- Mirrors and fallbacks. One address per file.
