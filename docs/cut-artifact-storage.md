# Evaluation cut artifacts

`turbo-eval run --dump-cuts <directory>` stores one completed artifact per set.
`turbo-eval cuts <directory-a> <directory-b>` compares the cut starts using the
chunker's comparison contract: start offset, level and parent. An identical
result does not certify complete chunk-range coverage or retrieval quality.
Both the source text and the actual view anchoring the chunks must match.

The pipeline receives a typed `DumpConfig` with an `EvaluationArtifactStore`
and `Precision`. Storage controls streaming reads, staged writes, publication,
listing and deletion through logical `ArtifactKey` names and `StorageError`.
The cut codec supplies names such as `scifact.cuts`; the backend has no format
extension policy. The first backend, `TemporaryFileStore`, stages in the chosen
local directory. Each application chooses that directory explicitly through
`TemporaryFileStore::new(directory)` or the CLI `--dump-cuts <directory>`.
Both staging and committed files stay there; there is no global temporary
directory fallback. Applications using distinct directories can reuse logical
keys without sharing bytes, listings or deletion. Completed files persist
until explicitly deleted.
Cache storage and document export have separate lifecycle requirements; no
S3, database, repository service, Git or Parquet implementation is provided.

Each writer exclusively creates its own staging file. Publication flushes and
syncs it, then uses a same-directory hard link that refuses existing names,
including dangling symlinks. There is no overwriting rename fallback. Dropping
or aborting a write discards its buffer and removes only its staging file.
Normal cleanup removes that link after publication. If cleanup fails, the
publication receipt carries that error while retaining the successful key.
A directory-sync failure after publication returns `StorageError::Published`;
callers must not interpret this as an unpublished write. Unix also syncs the
directory after publication/deletion and creates staging files with mode 0600.
Other platforms sync the file but do not provide directory-sync durability
through this backend. Filesystems without hard links return an explicit error.
Listing includes regular visible files only, excluding symlinks, directories
and staging files. Explicit reads use normal filesystem open semantics.
Crashes can leave unlisted staging files; automatic crash scavenging is not
implemented, and unrelated partial files are never removed.

Version 2 uses a binary header, framed document graphs, and a mandatory footer
with the document count and incremental SHA-256 of all preceding bytes through
the count. The digest must match, and no trailing bytes are allowed. Missing
footers, record-boundary truncation and invalid lengths are errors. The digest
provides accidental-corruption detection, not authentication or encryption.
The codec rejects version 1 and asks for regeneration with `--dump-cuts`.
This includes the older artifacts cited by #1484: they cannot be silently
upgraded into evidence of verified completion.

Readers and writers reject more than 100,000 documents, 64 MiB per serialized
graph, 64 KiB per document ID/header string, 1 MiB per header, 4,096 settings,
or 1 GiB total serialized bytes (including the 32-byte digest). Limits are
explicit refusals, never truncation. Reads stream and hash the serialized
artifact without loading it all first, but retain all decoded documents for
comparison; memory is not constant and decoded graphs can exceed wire size.

Graphs retain source text and non-vector document layers. Mode 0600 and an
unkeyed digest do not make that content encrypted or anonymous. The future
security and key-management design is tracked in #1490. No hardware speed,
model quality or cross-backend parity claim follows from codec tests.
