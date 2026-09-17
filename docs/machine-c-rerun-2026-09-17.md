# Machine C re-run — 2026-09-17 gate failures

Runbook for re-running the three Machine C gates that failed on
2026-09-17 (logs were under `/Volumes/pipework/work/m4-apple-logs/` on
the host), after the fixes on this branch. Run on the Apple Silicon
Metal host from a clean checkout of this branch. SDK release and
acceptance already passed on Machine C that day and are unchanged; the
optional commands to re-confirm them are at the end.

## What changed and why

1. **`apple_minilm_metal_cosine_vs_goldens`** failed at
   apple-mlx↔apple-golden min cosine 0.979436 < 0.99 (worst
   `sts-0056:a`). Root cause: the three Japanese-line entries in
   `testdata/e2e/goldens/apple/minilm.json` are a stale 2026-09-12
   capture that predates the `78a88d8` tokenizer fix; the live path now
   matches the nvidia reference on those texts (live↔nvidia was
   ≥ 0.999999, n=237, in the same failing run). The goldens are kept
   unmodified; the tests now gate provably stale entries (cross-golden
   cosine vs nvidia < 0.99, computable offline from the committed dumps)
   against the **nvidia** golden at ≥ 0.999 instead, list them in the
   receipt under `stale_apple_golden_items`, and cap the exemption at 5%
   of the set. See `testdata/e2e/goldens/apple/README.md`.
2. **`apple_solidify_bench_writes_machine_c_receipt`** panicked at the
   same apple-golden floor assertion (the rerank slice itself passed:
   Berlin max_abs ≈ 1.9e-6, allocs 0). Same fix as (1).
3. **`make bench-apple-overhead`** duplicated the Objective-C
   Tokenizers/MLX classes (swift-transformers and MLX were statically
   linked into the bench executable *and* embedded in the dlopen'd
   `libTurboEmbed.dylib`), inflated the first timed case to ABI p50
   ≈ 4.07× direct, and then died with `[metal::malloc] Resource limit
   (499000) exceeded` from two MLX runtimes accumulating resources
   across the 18-case grid. The ABI leg now runs in a separate
   `bench-abi-worker` process (dlopen-only, links no MLX and no
   swift-transformers), one worker per case that exits afterwards, and
   the direct path clears the MLX buffer cache between cases. Budgets
   are unchanged (ABI p50 ≤ 1.05×, throughput ≥ 0.95×, parity max abs
   ≤ 5e-4, RMSE ≤ 1e-4, steady-state arena allocs == 0). If the ratio
   still exceeds 1.05× after this isolation, that is a real overhead to
   investigate, not a budget to raise.

## Follow-up (2026-09-17, second attempt): overhead pilot IPC deadlock

The first re-run of `make bench-apple-overhead` on this branch passed the
cosine gate but then wedged for over 7 minutes on the first overhead
case: the `bench-abi-worker` child had finished warmup and was parked at
0% CPU reading stdin, while the parent never observed the
`{"ready":true}` line and blocked forever with no timeout.

Both ends of that handshake ran through layers whose buffering/queueing
behavior we do not control (`FileHandle.standardOutput.write` /
`FileHandle.read(upToCount:)` and C stdio `readLine`), the parent held
its copies of the worker's pipe ends open (so even a crashed worker
could never produce EOF), and nothing enforced a deadline. The wire
protocol now lives in
`swift/Sources/BenchOverheadCore/WorkerWire.swift`: raw POSIX
`read(2)`/`write(2)` on both sides (stdio stdout forced unbuffered and
flushed before every reply in the worker), the parent closes the child's
pipe ends after spawn, and every parent-side read has a `poll(2)`
deadline — ready within `--worker-ready-seconds` (default 600) and each
command within its own expected duration plus
`--worker-reply-grace-seconds` (default 120). A worker that misses a
deadline is killed and the make target fails with the phase and pid
instead of wedging the host.

The protocol is provable without Metal, models, or the dylib:

```bash
make bench-overhead-ipc-selftest      # any host with a Swift toolchain
```

runs `bench-abi-worker --io-selftest-parent`, which spawns the same
binary in `--io-selftest` mode (canned replies, never a bench result)
and exercises ready/collect/steady/time/concurrent/exit under tight
deadlines, plus the fail-loud path (a silent worker must trip the ready
timeout). `make bench-apple-overhead` also runs this self-test
automatically before loading any model or touching the GPU. Step 3
below is otherwise unchanged; process isolation and all budgets are
untouched.

## Prerequisites (once)

```bash
git fetch && git checkout <this branch> && git pull
make fetch-mlx ALIASES=minilm
make fetch-rerankers
```

## 1. Baseline (must stay green)

```bash
cargo test --locked --workspace          # parallel, macOS
```

## 2. Live Metal MiniLM cosine gate (failure 1)

```bash
cargo test -p turboembed --features mlx-live \
  --test apple_minilm_metal -- --include-ignored --nocapture
```

Expect `apple-mlx↔nvidia` min ≥ 0.97 (was 0.999999 on 2026-09-17),
`apple-mlx↔apple-golden` min ≥ 0.99 over the non-stale items, and
exactly the three `sts-005[678]:a` entries reported as exempted with
live↔nvidia ≥ 0.999. Refreshes
`testdata/receipts/turboembed/apple-minilm.json`; check its
`stale_apple_golden_items` before committing.

## 3. Overhead pilot (failure 3)

```bash
make bench-apple-overhead
cargo test -p turboembed --features mlx-live \
  --test apple_overhead_receipt -- --ignored --nocapture
```

The build now produces `swift/.build/release/bench-abi-worker` next to
the bench binary. There must be no `objc[…]: Class … is implemented in
both …` warnings for the worker legs, and the run must complete all 18
cases without the Metal resource-limit abort. Writes
`testdata/receipts/bench/machine-c-metal-overhead.json`; the receipt
validator gates it.

## 4. SOLIDIFY bench (failure 2)

```bash
make bench-turbo MACHINE=C               # = make bench-machine-c
```

Runs the rerank slice, then the combined receipt test. Expect the same
stale-item handling as step 2 and `pass=true` in
`testdata/receipts/bench/machine-c-metal.json`.

## 5. Unchanged — re-confirm only if desired

```bash
make apple-sdk-release
make apple-sdk-acceptance MODE=metal
```

Both passed on Machine C on 2026-09-17; nothing in this branch touches
the SDK packaging or acceptance path.
