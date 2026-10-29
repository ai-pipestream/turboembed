# The model catalog over gRPC

`turbo-kserve` serves `turbo.training.v1.TurboCatalogService` on its port
when started with `--catalog-dir`: the turbo-zoo index (docs/zoo.md) as
a catalog, the artifacts its cache holds, the licence acceptances a user
records, and installs that stream their progress.
`server/proto/turbo_training.proto` is the crate's own file; generate a
client from it in any language with gRPC.

The catalog directory is a zoo home, laid out as the `turbo-zoo` command
keeps it: the cache under `artifacts/<id>/` and the acceptance file
`acceptances.json`. A server and the command pointed at one home see
the same cache and the same acceptances. `--catalog-index FILE` serves
that index instead of the built-in one.

`--catalog-store URL` puts a content-addressed store in front of the
index's own addresses: `file:///dir` (a directory) or
`s3://bucket/prefix` (an S3 bucket on AWS, or on RustFS or
another S3-compatible store named by `AWS_ENDPOINT_URL`), both in the
`blobs/sha256/<hex>` layout docs/zoo.md describes. An install reads each
file from the store first and fetches it from its address only when the
store lacks it, and the store then keeps it once it verifies. So
servers that share one bucket fetch each file from its origin once, and
a store seeded ahead (`aws s3 sync`) serves installs with no outside
fetch. The bucket's credentials are the `AWS_*` variables the AWS tools
read. A file the store cannot keep is still installed. A
`CLASS_LOCAL_ONLY` artifact never goes through the store: it is fetched
from its publisher on each install and never kept there, since the class
allows no copy beyond the cache it is installed into.

## The service

| RPC | Admin | Answer |
|---|---|---|
| `ListModelCatalog` | no | Every artifact the index lists, in its order, with its licences, files and the terms it needs accepted. |
| `ListInstalledModels` | no | The artifacts whose files are all in the cache. |
| `RecordLicenceAcceptance` | yes, or zoo mode | Records that a user accepted a licence's terms. |
| `InstallModel` | yes, or zoo mode | Fetches and verifies an artifact into the cache, streaming its progress. |
| `DescribeModel` | no | One artifact's entry, as the listing gives it. |
| `ResolveModel` | no | The artifact and file the index pins at a SHA-256. |
| `ReadModelFile` | no | An installed file's bytes, for a user who accepted its terms; in zoo mode a file not yet installed is fetched first. |
| `RevokeLicenceAcceptance` | yes, or zoo mode | Withdraws a user's acceptance of a licence's terms. |
| `ListLicenceAcceptances` | no | The terms a user accepted, and whether the admin calls are served. |

The admin calls are served only when the server is started with
`--admin` or `--zoo`; without either they refuse with `UNIMPLEMENTED`
and touch no file. Without `--catalog-dir` every call refuses with `UNIMPLEMENTED`.
Fetching needs the server's `catalog-http` feature, which is on by
default; a server built with `--no-default-features` has no TLS stack
and exits at start when given `--catalog-dir`.

## Zoo mode

`--zoo` (or `TURBO_KSERVE_ZOO`) runs the server as a model zoo for its
users. It needs `--catalog-dir`, and the server may start with no
`--model`, serving only its catalog. In zoo mode:

- Any caller records an acceptance and starts an install, without
  `--admin`. The user a request names is the caller's own word, which
  suits a personal or LAN zoo; a shared zoo turns on mutual TLS
  (`--tls-client-ca`) so only holders of a client certificate reach it.
  `--admin` keeps its meaning for every other service.
- A `ReadModelFile` of an artifact the user is permitted, whose file is
  not installed yet, installs the whole artifact first, with the same
  licence check, free disk check and verified fetch `InstallModel`
  makes, through `--catalog-store` when one is set. The first reader
  waits for the fetch; readers of the same artifact that arrive while it
  runs wait for it too and fetch nothing. A read the licence check, the
  path or the pin refuses fetches nothing.

Without `--zoo` a read of a file not installed is refused, and an
install is an admin's step.

The turbo-zoo image runs the server this way, with the demo page in
front of it and, through `docker-compose.zoo.yml`, RustFS as its store
(docs/zoo-container.md).

## Prefetch

`--catalog-prefetch permissive` (or `TURBO_KSERVE_CATALOG_PREFETCH`)
installs every permissive artifact the index lists, in the background
from the start, one after another, through `--catalog-store` when one
is set. Each install takes the same free disk check and verified fetch
as `InstallModel` and holds the artifact's read-through gate, so a read
that arrives meanwhile waits for it. A failure is logged and the next
artifact is tried. Artifacts that need an acceptance are never
prefetched. It needs `--catalog-dir`.

## Listing

`ListModelCatalog` takes optional filters, `kind` (`KIND_LEXICON`, as
docs/zoo.md names kinds) and `language`, and an optional `user`. Each
entry carries the artifact's fields as the index records them, every
file with its unpacked size and digests and, for a file served
compressed, its compression and wire size and SHA-256, and `terms`:
each licence text a user accepts before the artifact is installed. With
a `user`, `missing_terms` lists the ones that user has not accepted.
Refused artifacts are never listed.

An artifact is installed when every one of its files is a plain file in
the cache at the size the index records. Listing reads no digests;
every install, and every fetch the zoo makes, checks them.

## Licence acceptance

`RecordLicenceAcceptance` takes a `user`, a `licence` by its SPDX
identifier as the catalog lists it, and, only for a licence without a
text of its own (`LicenseRef-RestrictedPublic`), the `terms` address the
entry's `terms` gives. It records the acceptance in the home's
acceptance file, as `turbo-zoo accept --yes` does, and answers with the
terms and when they were accepted. Accepting again keeps the first
time. A client calls it only after showing the user the text at that
address.

A page that shows the terms also sends `artifact_id`, the artifact
whose page it was, and `text_sha256`, the SHA-256 of the
`LicenceTerms.text` it displayed, as 64 lowercase hex digits. The
server checks the digest against the text it serves for those terms, so
the acceptance names that exact text, and records both in the
acceptance file (version 2, docs/zoo.md). The artifact is for audit
only: one acceptance still covers every artifact under the same terms.
Both fields are optional; without them the record is the same as the
command line's.

| Request | Refusal |
|---|---|
| A user name that is empty, over 256 bytes, has control characters, or leading or trailing whitespace | `INVALID_ARGUMENT`, field 1 |
| A licence the zoo does not know, a refused one, or a permissive one, which has nothing to accept | `INVALID_ARGUMENT`, field 2 |
| A terms address for a licence with its own text, or none for one without | `INVALID_ARGUMENT`, field 3 |
| An `artifact_id` the index does not list, or whose terms do not include these | `INVALID_ARGUMENT`, field 4 |
| A `text_sha256` that is not 64 lowercase hex digits, or not the SHA-256 of the served text | `INVALID_ARGUMENT`, field 5 |

`RevokeLicenceAcceptance` takes the same `user`, `licence` and `terms`
and removes that acceptance from the file, as `turbo-zoo revoke` does,
with the same refusals for fields 1 to 3. `was_accepted` says whether
there was one. Files already installed stay in the cache, but neither
`ReadModelFile` nor the zoo hands any of them to that user again until
the terms are accepted again.

`ListLicenceAcceptances` takes a `user` and answers with that user's
acceptances in the order made: the terms, when, and the artifact and
digest when a page recorded them. `admin` says whether the server
serves the admin calls (`--admin` or `--zoo`), so a page can show its controls or say it is
read-only. An empty `user` lists nothing. Acceptances made with
`turbo-zoo accept` or `turbo-bundle` are in the same file and are
listed alike.

## Installing

`InstallModel` takes a `user` and an `artifact_id`. Before its first
message it checks, in order:

1. The user name is valid and the id is not empty: `INVALID_ARGUMENT`,
   field 1 or 2, otherwise.
2. No other install is running: `UNAVAILABLE` (`TURBO_E_BUSY`)
   otherwise, which a client may retry. One install runs at a time.
3. The id names an artifact: `NOT_FOUND` (`TURBO_E_BUNDLE_NOT_FOUND`)
   otherwise.
4. The user accepted every term the artifact's class needs:
   `FAILED_PRECONDITION` (`TURBO_E_INVALID_STATE`) otherwise, naming
   the missing terms. A refused artifact is refused the same way.
5. The files still to fetch, every file not in the cache whole and
   verified, fit on the cache's file system with 64 MiB to spare: `OUT_OF_RANGE` (`TURBO_E_CAPACITY`) otherwise, naming the
   bytes needed and free.

It then streams `InstallProgress` messages and ends with one
`InstalledModel`:

| Stage | When | `bytes` |
|---|---|---|
| `INSTALL_STAGE_ADMITTED` | The checks passed. | What the files still to fetch take on disk. |
| `INSTALL_STAGE_DOWNLOADING` | A file is requested. | Its size on the wire. |
| `INSTALL_STAGE_DOWNLOADED` | Its bytes arrived, not yet checked. | What arrived. |
| `INSTALL_STAGE_VERIFIED` | Every file is in the cache and verified. | Their total. |

A file already in the cache is read and verified instead of fetched, so
installing an installed artifact streams `ADMITTED` with 0 bytes, then
`VERIFIED`, while a cached file whose bytes no longer verify is fetched
again and counted. Two files may be fetched from one address; each
reports its own `file_index` and `path`, also when one of them is kept
from the cache. Each fetched file is checked as docs/zoo.md says, by its
wire size and SHA-256 when served compressed, then unpacked and checked
by size, SHA-256 and SHA-512, before it is written; it is written under
its name only whole. A file that fails ends the stream with `DATA_LOSS`
(`TURBO_E_BUNDLE_INTEGRITY`) naming the check, and is never cached. A
transport or disk failure ends it with `INTERNAL` (`TURBO_E_RUNTIME`).
An install the client abandons carries on to its end, so the files it
fetched are cached for the next one.

The server fetches over HTTPS as the `turbo-zoo` command does: redirects
followed over HTTPS only, at most one byte more read than the index
expects, and the bytes asked for unencoded (`Accept-Encoding:
identity`). A file served compressed reaches the wire check as sent;
if a server encodes it anyway and the transport decodes that, the bytes
that arrive are the file itself, kept when they verify as it.

## Describing, resolving and reading

These are the calls a client that loads models from this server makes
(#1613). None is an admin call.

`DescribeModel` takes an `artifact_id` and an optional `user` and
answers with the artifact's `CatalogEntry`, the one `ListModelCatalog`
gives, `missing_terms` included when a user is named. An id the index
does not list is `NOT_FOUND` (`TURBO_E_BUNDLE_NOT_FOUND`); a refused one
is `FAILED_PRECONDITION`.

`ResolveModel` takes a file's `sha256` (64 lowercase hexadecimal
digits; anything else is `INVALID_ARGUMENT`, field 1) and answers with
the first artifact in index order that has a file pinned at it, and that
file's `path`. With an `artifact_id`, only that artifact's files are
searched. A hash the index pins nowhere is `NOT_FOUND`.

`ReadModelFile` takes a `user`, an `artifact_id`, a `path` and,
optionally, the `sha256` the caller's own index pins for that file. It
checks, in order, before its first message:

1. The user name is valid, the id and path are not empty, and the
   `sha256`, when given, is well formed: `INVALID_ARGUMENT`, field 1,
   2, 3 or 4, otherwise.
2. The user accepted every term the artifact's class needs:
   `FAILED_PRECONDITION` (`TURBO_E_INVALID_STATE`) otherwise, naming
   the missing terms, as `InstallModel` refuses. An unknown id is
   `NOT_FOUND`.
3. The artifact has a file at `path`: `NOT_FOUND` otherwise.
4. The server's index pins that file at the request's `sha256`, when
   one is given: `FAILED_PRECONDITION`, field 4, naming both, otherwise.
   A client and a server whose indexes disagree never exchange bytes.
5. The file is installed, a plain file in the cache at the size the
   index records: `FAILED_PRECONDITION` otherwise. In zoo mode the
   artifact is installed first instead (Zoo mode, above), and a fetch
   that fails answers as `InstallModel` would.

It then streams the file's `CatalogFile`, then its bytes in frames of
at most 64 KiB, read from the cache as they are sent, never held whole.
The bytes are hashed on the way; when they are not the size and SHA-256
the index records (the cache was changed since the install), the stream
ends with `DATA_LOSS` (`TURBO_E_BUNDLE_INTEGRITY`) instead of ending
cleanly. A client verifies every file by size, SHA-256 and SHA-512 all
the same, as it verifies a download from anywhere else.
