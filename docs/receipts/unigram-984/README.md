# Unigram default-overhead correction

This receipt measures the implementation in `sources.json`. The actual main
baseline is `bf52319dfc49486e5109f2c76faccb8fcd6edc02`; the blocked PR baseline
is `059947a90031c5081605512087a63cc925ba6565`. The dictionary, corpora, gold
files and training-word lists are pinned by SHA-256 in `identities.json`.
No corpus text or new model data is included.

The comparison uses the complete PKU and MSR test files. Model construction,
file reading and accuracy scoring precede timing. Every process scores every
mode, checks reconstructed text, and records exact boundary counts, OOV counts
and a token-output fingerprint. The summary refuses changed outputs between
blocked and fixed modes, and checks main against explicit OpenNLP mode.

Each process performs one warmup and two measured passes, each of 50 complete
corpus repetitions. Method order reverses on alternate passes. Twelve rounds
reverse the order of fixed, blocked and standalone-main processes. Turbo reuses
its public scratch/output buffers; jieba-rs uses its normal allocated result.
This compares public APIs, including their allocator behavior.

For the primary default/main ratio, the fixed process also calls the unmodified
main library. Only its private Cargo package/library names and dependency paths
are changed to permit both versions in one binary. Its `src` and `data` are
symlinks into the pristine main checkout; the shared dependencies have identical
source between main and fixed. Thus the current main comparison runs under the
same process and affinity, rather than depending on stable load across processes.
Each paired value divides the mean of that round's two default samples by the
mean of its two main-source samples. The reported ratio is the median of twelve
pairs. Standalone-main and original blocked binaries remain separate controls;
their cross-process ratios are retained rather than treated as isolation proof.

The CPU is an AMD Ryzen 9 9950X3D, with affinity to logical CPU 16. Compiler,
Cargo, flags, source objects, driver/lock hashes and executable SHA-256 values
are recorded. The host was shared with other CPU work, GPU compilation and
CUDA correctness work. Our integration scheduler was paused during timing.
Raw one-minute load readings are in `runs.json`; these local measurements
do not establish isolated-host significance or a throughput guarantee.

After measurement, main advanced to
`93aa4ba2401809ea955b1608fa57db9cdf021246` through runner and ABI-test changes.
It was integrated normally at `fc1e0170cb991dced701c50f15804d9110dc290c`.
The relevant runtime source trees are unchanged. Rebuilding both the fixed and
standalone-main comparison binaries after synchronization produced exactly the
measured SHA-256 values, so no unchanged timing run was repeated. The source
manifest records this synchronization separately from the measured revisions.

## Recorded result

The fixed process has median main/default times of 4.134205/4.054950 ms on
PKU and 4.461030/4.373875 ms on MSR. Median paired default/main ratios are
0.991857 and 0.946148. The default shows no slowdown against main in this run.
The now faster explicit OpenNLP path remains below the grouped default:
default/OpenNLP paired ratios are 1.079423 and 1.043576. Both comparisons are
retained in `summary.json`; the compatibility-path cost is not concealed.
One-minute host load ranges from 36.77 to 81.72. No final-run sample was removed.

## Reproduce

Use the final PR checkout as `FIXED`; its runtime code is unchanged from the
fixed runtime commit in `sources.json`. Use the pinned main and blocked
revisions in distinct checkouts, and the pinned dictionary and extracted
SIGHAN release already obtained under its research terms. Each generated
comparison has its own build target. Node and Cargo are required; no product
manifest or workspace dependency is modified.

```sh
node "$FIXED/docs/receipts/unigram-984/prepare.mjs" \
  "$FIXED" "$MAIN" "$BLOCKED" "$NEW_COMPARISON_DIR"
node "$FIXED/docs/receipts/unigram-984/measure.mjs" \
  "$NEW_COMPARISON_DIR/fixed/target/release/unigram-comparison" \
  "$NEW_COMPARISON_DIR/main/target/release/unigram-comparison" \
  "$NEW_COMPARISON_DIR/blocked/target/release/unigram-comparison" \
  "$DICT" "$CORPUS" "$NEW_RESULTS_DIR" 1 12
node "$FIXED/docs/receipts/unigram-984/summarize.mjs" "$NEW_RESULTS_DIR"
TURBO_JIEBA_DICT="$DICT" TURBO_SIGHAN_DIR="$CORPUS" \
  cargo test --locked --release --manifest-path "$FIXED/Cargo.toml" \
  -p turbo-tokenize --test unigram_sighan -- --ignored --nocapture
```

The full accuracy assertions preserve every mode's established PKU/MSR counts.
The product's trained BMES checkpoint and configuration hashes are unchanged.
The earlier byte-scan and repeated-preparation prototypes are preserved at the
commits named in `sources.json`; their raw unsuccessful trials are retained
under `rejected/`, including the partial first trial. They are development
evidence and are not acceptance measurements for the final implementation.
