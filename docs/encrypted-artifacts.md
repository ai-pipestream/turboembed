# Encrypted evaluation artifacts

`turbo_eval::artifact::encrypted::EncryptedStore` wraps an
`EvaluationArtifactStore` with client-side AES-256-GCM encryption. The explicit
`SecurityContext` supplies an authenticated principal, encoding identifier,
key provider, authorization policy, retention policy and audit sink. This is
an opt-in library adapter, part of #1490. Existing CLI cut exports still use
`TemporaryFileStore` directly and remain plaintext. No key service, login
flow or production remote provider is introduced here.

Purpose, destination and encoding remain separate. This adapter supports only
the existing evaluation-artifact operations: begin, publish, open, list and
delete. It does not implement cache replacement, document append, format
conversion, database operations or a provider registry. The application picks
the byte encoding; its opaque `EncodingId` participates in authentication.

## Identity and policy

Call `new_artifact_key()` for a fresh opaque identifier. Names must be 32
lower-case hexadecimal characters; human document names are refused. The
application keeps the document-name mapping in its own protected metadata
store. Never derive an identifier from source text or reuse it after deletion.
The provider name combines opaque principal and artifact identifiers. This
reveals object grouping but no title. Listing filters by the fixed caller's
scope. Bytes copied to another principal, identifier or encoding fail
verification, even if both principals use the same key provider.

The application constructs the context after authentication and keeps the
backing store and key provider private. A request payload cannot change a
store's principal. Every operation requires authorization; publication checks
again after writing. There is no permissive default policy. An unavailable
audit sink refuses access before the provider is touched. Audit records carry
only opaque identifiers, operation and outcome. The trusted audit sink must
persist intents and reliably record outcomes, or reconcile incomplete intents.
It must not add document text, errors from external providers, credentials or
key material. A completion callback cannot undo an already published object.

Deletion also requires the authoritative `Retention` ledger to allow it. That
ledger lives outside mutable object metadata and must enforce minimum
retention, expiry and holds. The adapter does not invent a deadline or silently
expire data. The host supplies this contract and owns durable retention state;
the test policy exercises a hold and release, not a deployed compliance ledger.
Authorization or ledger failure leaves the object intact. Principals cannot
list or delete another principal's names through this adapter. This is an
application capability boundary, not an S3 IAM or operating-system sandbox.
Direct access to the backing store bypasses policy and must remain privileged.

## Encryption and completion

The narrow approved dependency exception uses `ring` only for maintained AEAD,
HKDF and operating-system randomness. No cipher, MAC or key derivation primitive
is implemented here. Each artifact gets a 256-bit random salt. HKDF-SHA-256
derives a distinct AES-256-GCM key from the supplied master key. Per-record
nonces use a monotonically increasing counter and never wrap. Data frames hold
at most 64 KiB of plaintext. Version, opaque key ID, salt, caller/artifact/
encoding binding, record kind, length and index are authenticated. A mandatory
terminal frame authenticates both record count and total plaintext length.
Truncation, reordered/duplicated/omitted frames, changed metadata, invalid tags
and trailing bytes are refused. Lengths are checked before allocating buffers.

`KeyProvider` returns opaque IDs and non-serializable key material. Key IDs
must permanently identify the same material. Rotation selects a new active ID;
old objects need their original key. Missing and revoked keys have distinct
typed errors and never select plaintext. Publication re-resolves its key to
check revocation. Revocation denies new opens and publication, but does not
recall an already returned reader or remove plaintext already in application
memory. No memory-locking or guaranteed key zeroization claim is made.

An open first streams the provider's ciphertext into a private snapshot under
the explicitly configured scratch root, verifying every frame and completion
while discarding plaintext. Only then does it return a decrypting reader over
that same snapshot descriptor. Replacing the provider object cannot change
that reader. Scratch contains only ciphertext. Unix snapshot directories are
mode 0700 and files mode 0600; other platforms require equivalent private ACLs
on the root. The root must exist. There is no global temporary-directory
fallback. The snapshot size is capped by the caller's ciphertext budget.

Writing copies plaintext into one bounded frame buffer and encrypts it in
place. It only sends ciphertext to the provider. Flush/I/O failures permanently
poison the writer. The same ciphertext budget bounds completed writes. An
underlying `TemporaryFileStore` preserves exclusive staging and atomic
no-replace hard-link publication. Competing writers cannot replace a winner;
abort and drop operate on their owned staging only. Underlying providers must
honor that same contract: the adapter cannot turn an overwriting PUT into a
conditional publication. Errors are sanitized at the adapter boundary;
published-but-not-durable and staging-cleanup warnings remain distinguishable.

Snapshot drop and writer drop perform best-effort cleanup. Crashes or cleanup
failures can leave private ciphertext staging; no automatic scavenger is
provided. Recovery must remove only proven abandoned staging, never arbitrary
files or published names. Keep the configured directories private from other
processes. A compromised process, administrator or key service is outside the
at-rest threat model.

## Remote and transport limits

The explicit `encrypted_rustfs` integration test uses a real, pre-existing
RustFS service and a fresh isolated bucket. Its test-only provider streams
ciphertext staging through the repository's existing zoo SigV4 signer and
ureq transport. It sends conditional PUT with `If-None-Match: *`, refuses
redirects, and streams GET. It is not a production provider: listing is
unsupported, ambiguous network outcomes need reconciliation, and it is not
wired into the zoo holder registry. No provider authentication implementation
is duplicated and no zoo source code is changed.

The observed test transport was local HTTP. This is not TLS, certificate or
remote-network proof. Production transport must enforce authenticated TLS;
client-side encryption does not hide object names, counts, grouping, sizes,
access patterns, key IDs or request metadata. Wrong-credential refusal proves
signature enforcement only. Scoped application checks are not separate-user
IAM, bucket-policy or cross-account proof. Provider retention locks, versioning,
backups and replication were not tested. DELETE removes the current object;
it does not certify deletion from backups, versions, snapshots or retained
keys. The host must implement those policies before making deletion guarantees.

The format authenticates individual object content and metadata. It does not
provide a trusted inventory, detect a provider omitting names from listings,
or prevent rollback to an older valid object with the same identifier. Fresh,
never-reused identifiers and a protected application inventory are required.
Store only the document content and metadata actually needed by the consumer;
this wrapper encrypts bytes but does not redact or anonymize them.

## Reproducing evidence

Run the focused tests with Rust 1.99:

```sh
cargo +1.99 test --locked -p turbo-eval --lib artifact::encrypted
cargo +1.99 test --locked -p turbo-eval --test encrypted_artifacts --test artifact_storage
```

The ignored RustFS test requires explicit `ARTIFACT_TEST_ENDPOINT`,
`ARTIFACT_TEST_BUCKET`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in its
process environment. Use a fresh bucket whose name starts with
`artifact-security-test-`; credentials must never be printed or put on the
command line. The test deletes only its exact run-scoped keys. The operator
owns bucket creation and removal and must preserve the existing service and
volume. Do not use a production bucket.

```sh
cargo +1.99 test --locked -p turbo-eval --test encrypted_rustfs -- --ignored
```

The full-adapter measurement is opt-in through `ARTIFACT_MEASURE_MIB`:

```sh
ARTIFACT_MEASURE_MIB=128 cargo +1.99 test --locked -p turbo-eval \
  --test encrypted_artifacts bounded_adapter_measurement -- --ignored
```

For peak RSS, invoke the compiled test executable under the platform's process
measurement tool so compilation memory is excluded. The reader performs one
provider pass that writes and authenticates a ciphertext snapshot, followed by
one snapshot read/decryption pass. Snapshot disk usage scales with artifact
size; plaintext working buffers do not. Provider buffers and application
consumption are separate: reading the result into a growing vector is not a
bounded-memory consumer. Measurements and their limits are recorded with the
pull request, not claimed from the ignored test alone.
