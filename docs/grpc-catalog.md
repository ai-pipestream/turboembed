# The model catalog over gRPC

`turbo-kserve` serves `turbo.training.TurboCatalogService` on its port
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

## The service

| RPC | Admin | Answer |
|---|---|---|
| `ListModelCatalog` | no | Every artifact the index lists, in its order, with its licences, files and the terms it needs accepted. |
| `ListInstalledModels` | no | The artifacts whose files are all in the cache. |
| `RecordLicenceAcceptance` | yes | Records that a user accepted a licence's terms. |
| `InstallModel` | yes | Fetches and verifies an artifact into the cache, streaming its progress. |

The admin calls are served only when the server is started with
`--admin`; without it they refuse with `UNIMPLEMENTED` and touch no
file. Without `--catalog-dir` every call refuses with `UNIMPLEMENTED`.
Fetching needs the server's `catalog-http` feature, which is on by
default; a server built with `--no-default-features` has no TLS stack
and exits at start when given `--catalog-dir`.

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

| Request | Refusal |
|---|---|
| A user name that is empty, over 256 bytes, has control characters, or leading or trailing whitespace | `INVALID_ARGUMENT`, field 1 |
| A licence the zoo does not know, a refused one, or a permissive one, which has nothing to accept | `INVALID_ARGUMENT`, field 2 |
| A terms address for a licence with its own text, or none for one without | `INVALID_ARGUMENT`, field 3 |

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
reports its own `file_index` and `path`. Each fetched file is checked as docs/zoo.md says, by its
wire size and SHA-256 when served compressed, then unpacked and checked
by size, SHA-256 and SHA-512, before it is written; it is written under
its name only whole. A file that fails ends the stream with `DATA_LOSS`
(`TURBO_E_BUNDLE_INTEGRITY`) naming the check, and is never cached. A
transport or disk failure ends it with `INTERNAL` (`TURBO_E_RUNTIME`).
An install the client abandons carries on to its end, so the files it
fetched are cached for the next one.

The server fetches over HTTPS as the `turbo-zoo` command does: redirects
followed over HTTPS only, at most one byte more read than the index
expects.
