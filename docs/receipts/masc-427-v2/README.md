# MASC namefinder registered -2 attempt

Execution source: `8812b1db16ff47890be43be96d6835eea12306af`. Release binary SHA-256:
`be818fba6d2cc25b4f045b0ad7f55950c146a544d3fe5efdf2d0c6ba4976c664`.

This is an execution receipt, not a model release. Original #427 acceptance requires every type
to pass its held-out non-inferiority gate. Location, organization and date fail that unchanged
gate. No -2 model is added to the release inventory or zoo index, and #427 remains open.

The four configurations were registered before execution, in this order:

1. 300 iterations, cutoff 0, document clearing.
2. 300 iterations, cutoff 0, sentence clearing.
3. 100 iterations, cutoff 5, document clearing.
4. 100 iterations, cutoff 5, sentence clearing.

Each configuration uses 10 whole-document CV folds over the 351 training documents. The highest
merged F1 selects a configuration; ties select the earliest configuration. All four types use
24,958 training samples. Final training uses all 351 training documents, followed by one gate on
the unchanged 39 held-out documents. Held-out evaluation uses 6,384 samples and records one
skipped sample. Twitter's two documents remain excluded. This is the existing MASC held-out
partition, already used for the historical -1 attempt, not a new untouched test set. Candidate
and pinned OpenNLP reference use the selected adaptive-state clearing policy.

The unchanged gate computes span-level aggregate F1 and a paired 95% bootstrap interval by
drawing whole held-out samples with replacement. It uses 10,000 resamples, seed 154, and passes
only when the lower difference bound is at least -0.5 percentage points. These intervals
resample samples, not whole documents. Organization's positive point difference is insufficient
because its lower bound is below the margin.

Runtime is elapsed process time on a shared host, not an isolated throughput comparison. Types
run serially; each process uses one training thread, affinity to two logical CPUs, and nice 10.
Compilation uses release profile and two Cargo jobs. Start/end load averages and per-type
durations are retained. The execution summary removes private command paths and process IDs; the
original private wrapper log remains preserved.

No corpus text, reference model weights, or candidate model weights are in this receipt.
Compressed JSON retains the original selection, card and paired count record bytes. SHA-256
records cover compressed files and decompressed originals; the model hash in each original card
identifies the locally preserved, unpublished binary.

No held-out-driven retuning occurred. Training-only CV shows low recall for location and
organization across the registered grid. A later model change needs preregistration and an
evaluation plan that accounts for the exposed holdout; these results do not justify weakening
the gate or treating that holdout as unseen.

This run does not resolve the existing per-source rights audit question. ANC's documented CC BY
3.0 US statement and the Twitter exclusion are unchanged. Full training was not repeated for a
second bitwise determinism check; prior authored-fixture determinism tests do not establish such
a full-run result.

## Records and verification

The complete CV, selected-configuration, final-gate and runtime tables are in [the recipe
README](../../../analysis/train-recipes/README.md#namefinder-cross-validation-clearing-policy).
`execution.json` is a sanitized wrapper summary, with compiler, CPU class, source/binary hash,
start/end UTC, wall time and start/end load averages. `inputs.json` contains the archive and
pinned reference file sizes and hashes. `summary.json` contains the independently checked counts
and F1 values. `checksums.json` identifies each compressed record and its original bytes. The
`*-selection.json.gz`, `*-card.json.gz` and `*-gate.json.gz` files preserve all 160 fold records
and all 4 original final gates without corpus excerpts. No model weights or full execution logs
are committed.

From the repository root:

```sh
python3 docs/receipts/masc-427-v2/verify.py
```

This verifies record hashes, every fold's disjointness and coverage, merged counts, winning
configurations, final counts/F1, model identity across records, process status and the original
margin decisions. A successful verifier means the records are consistent; it does not change any
failed accuracy gate. It does not train, load models or repeat bootstrap resampling. The
original card records iterations and cutoff but has no adaptive-state clearing field; clearing
is recorded in selection.json. The execution source above connects that chosen policy to
training and to both models in the final gate.

The recorded execution used this form once per registered type, in person, location,
organization, date order, after a release build with 2 Cargo jobs:

```sh
nice -n 10 taskset -c 30,31 <build-target>/release/turbo-train-recipes \
  masc-namefind-select --corpus MASC-3.0.0 --references references \
  --out models-v2 --recipe turbo-en-masc-3.0.0-ner-<type>-2
```

These commands document the completed attempt; no training or gate was retried. Original model
files remain preserved locally and match the SHA-256/SHA-512 and sizes in their cards; they have
not been published.
