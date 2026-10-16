# Real document trace receipt (#500)

This harness requires the implementations in PRs #495, #496 and #497.
It is outside the Cargo workspace: CI does not download MASC or a model.
Use a clean checkout containing those patches. The reference run used
base `272b9298` with `98bee392`, `c958b17d` and `cc62410b`, assembled as
`0a5cf9ee`. Only an append conflict in `docs/analysis.md` needed resolution.
The later documentation-only change in #497 removes that conflict.

```sh
export TURBO_TRACE_CHECKOUT=/path/to/checkout-with-the-three-patches
export TURBO_MASC_ROOT=/path/to/MASC-3.0.0/data
export TURBO_MINILM_BUNDLE=/path/to/verified-minilm-bundle
bash docs/receipts/document-trace/run.sh /path/to/output
```

The runner generates a standalone Cargo project with the checked-in lock
file and dependencies from the selected checkout, and removes the temporary
build directory when it exits. It requires the real MiniLM bundle; a load
failure stops the run. It checks the MASC source hashes, 5,214 token
mappings back to original bytes, model identity, 384-dimensional normalized
vectors, selected layer/edge counts, filtered geo mentions and leaf chunk
budgets.

Each run writes `<output>/trace.json` (and `<output>/<name>-trace.json`),
which include timings, layer provenance and text excerpts from the caller's
corpus files, alongside `<output>/reference.json`, which holds the count-only
summary, model identity and execution metadata without timings or corpus
excerpts. Compare a rerun against the committed `reference.json` with:

```sh
diff -u \
  <(jq 'del(.checkout_commit)' docs/receipts/document-trace/reference.json) \
  <(jq 'del(.checkout_commit)' /path/to/output/reference.json)
```

`reference.json` contains the CPU results. It excludes timings because this
is a debug integration run, not a benchmark. Existing benchmark records
were left unchanged and checked against their original SHA-256 digests.

Observed results:

- Dublin: 14 single-letter candidates filtered; 44 locations; 1 abbreviation
  edge; 8 leaf chunks; 0 over-budget leaves and 1 over-budget parent.
- Pulmonary fibrosis: 17 single-letter candidates filtered; 5 locations;
  74 abbreviation edges; 17 structure blocks; 30 leaf chunks; 0 over-budget
  leaves and 2 over-budget parents.
- Normalized term indexing builds and runs with the linguistic layer.
  Per-document term counts are 516 and 866.

The original diagnostic counted parents as budget failures. Large parent
chunks represent sections, and do not establish a leaf-size defect.
General long-sentence splitting remains part of stage 4 (#148).

This trace does not evaluate NER quality, contextual geo disambiguation,
dependency/coreference/relation inference, sentiment, graph persistence,
gRPC, or GPU execution. It tests available components in combination.
Dublin's markup is in a separate MASC logical annotation file; importing
that file is tracked by #498. A graph export boundary is tracked by #499.

The model used for this receipt was built from all-MiniLM-L6-v2 snapshot
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with weight SHA-256
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
The project bundle reference builder and all 3 native CPU conformance
checks passed before the original trace. Bundle identity is in the JSON.
