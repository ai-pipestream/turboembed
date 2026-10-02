# Landing PR #118

This file is the shared checklist for landing the npu backend. Two people
work this PR from two sides: the author, with the Arrow Lake NPU and the
machine that sealed the MiniLM bundle, and the reviewer, with the B70 and
CUDA hosts where the rest of the merge bar runs. All coordination happens
in PR comments, nowhere else; a change of plan is a comment first. This
file is deleted in the last commit before the merge so main's tree does
not carry it.

## Rules every commit on this branch meets

- Author and committer are the repository owner; no trailers of any kind.
- No host names, user names, machine paths, plans or roadmaps in the tree,
  the records, the commit messages or the PR text. Machine classes
  (arl-npu, b70, rtx4080) are fine.
- No mocks and no fake devices: a test without the device skips and says
  so; `TURBO_TEST_REQUIRE_NPU=1` turns the skip into a failure.
- No ONNX execution and no Python in the library; Python runs only in the
  pinned reference container.
- A SUPPORTED cell rests on ROWS_MIXED records for that device class,
  model and precision, made from a clean pushed tree, whose commit, driver
  and reference pins agree with the docs. Everything else is EXPERIMENTAL
  and is presented as such.
- Merge bar on the exact tip: `cargo fmt --all -- --check`,
  `cargo +1.98 clippy --workspace --all-targets -- -D warnings` plain and
  with each feature (npu, cuda, cuda,cuda-cublas, levelzero,
  levelzero-onednn), tests plain and with each feature on the host that has
  the device, CI green. The reviewer runs the cuda and levelzero halves.

## State at this tip

Main (47d5013, which has the reference-image lookup fix of #126) is merged
in. The review items of 2026-10-02 are done. The bar passed on all hosts at
16ac473, before the last merge of main. The reference image pinned by the
MiniLM recipe is loaded on a team machine, and with #126 the pin resolves
there.

## Steps, in order

1. Author: merge main again if it has moved, then run
   `turbo-bundle verify` on the team machine holding the reference image
   against the sealed MiniLM bundle, and post the result on the PR.
2. Author: delete this file in the same commit as any last fix, mark the PR
   ready, and post the tip's commit.
3. Reviewer: run the cuda, levelzero and levelzero-onednn halves of the bar
   on that tip and post the result.
4. CI green on the tip, then the reviewer merges with a merge commit (never a
   squash: the records and the docs cite branch commits), deletes the
   branch, and fast-forwards the publishing branch.

## Follow-ups, each its own PR after the merge

- Pin the reference Dockerfile's base image by digest, so a rebuild gives a
  stable image id.
- NATIVE graph format on hardware: the cell stays EXPERIMENTAL until a
  record exists.
- Linux records for the same model, if a Linux NPU driver is available.
