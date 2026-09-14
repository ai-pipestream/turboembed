# Intel prepared execution pilot, 2026-09-14

This preliminary pilot compares a direct OpenVINO GPU reference with the
prepared C ABI. Its raw JSON is preserved in
[`testdata/receipts/bench/intel-prepared-performance-2026-09-14-raw-json.tar.gz`](../testdata/receipts/bench/intel-prepared-performance-2026-09-14-raw-json.tar.gz)
(SHA-256 `cdd2c35980b9538f06fdad3094633a057c7d64605524c0d4165c18a4bc0b3c76`; see
the adjacent `.sha256` file). The archive contains 18 unmodified case receipts.
For the exact model bundle provenance and environment, see the
[Intel prepared SDK validation](intel-prepared-sdk-2026-09-14.md). The native
implementation was at `596e5c3`; the benchmark source accompanies this receipt.

## Workload and measurements

The runner covered batches 1, 8, and 32; sequence lengths 32, 128, and 256;
and both full-length and deterministic mixed-length rows. It creates token IDs,
masks, and type IDs from the bundle tokenizer, then uses the same fixed inputs
for both paths. Each case has a 20-execution warmup and three repeats, with
native-first and ABI-first order alternating. A repeat stops at 10 seconds or
10,000 requests.

The direct reference compiles the pooled MiniLM graph for `GPU.0`, binds three
OpenCL input buffers and a `[batch, 384]` OpenCL output buffer, then measures
synchronous `InferRequest::infer`. The ABI path creates a GPU context, loads the
same bundle, creates a fixed slot, writes the same prepared inputs once, then
measures `slot_execute` plus result release. Timed samples exclude host result
readback; explicit reads occur only in the separate parity check. These are
synchronous end-to-end call/completion measurements for each path, not GPU
kernel-time measurements.

All receipts name `Intel(R) Graphics [0xe223] (dGPU)` and OpenVINO build
`2026.3.1-22476-759c5a6ab8c-releases/2026/3`.

## Result

All 18 cases passed direct-versus-ABI numerical parity: maximum absolute error
was at most `2.09e-7` against the `5e-4` gate and maximum RMSE was `2.71e-8`
against the `1e-4` gate.

All 54 repeats met the pilot p50 and throughput gates. ABI p50 ranged from
1.58% faster to 1.49% slower than direct OpenVINO, within the 5% overhead limit.
Throughput was at least 98.41% of the native reference, above the 95% limit.
The slowest p50 ratio occurred at batch 8, sequence 256, mixed-length input.

Each timed path produced 584 to 10,000 samples. Forty-eight native/ABI sample
pairs have at least 1,000 samples and record p99 as descriptive only. The six
repeats for batch 32, sequence 256 have 584 to 634 samples, so their receipts
mark p99 as insufficient; this pilot makes no tail-latency claim.

The raw records also retain direct and ABI load/compile/upload timings: 162.74
to 459.11 ms for the direct path and 478.39 to 496.67 ms for the ABI path. They
are evidence for this captured workload, but no cold-start budget was declared
or qualified here. Device profiling data is retained separately with profiling
disabled during timing. The batch-8, sequence-128, mixed case adds three
concurrent two-context observations: each context measured p50 between 3432.08
and 3439.06 microseconds and 290.05 to 290.89 requests per second. This is a
recorded isolation observation, not a scalability acceptance result.

## Limits and follow-up

The allocation probe runs ten post-timing executions while a global C++
`new`/`new[]` counter in the benchmark executable is enabled. It observed
18,920 to 19,470 calls and about 5.28 to 5.34 MiB for both paths, so this pilot
does not support a zero-allocation claim. The counter observes C++ allocations
that resolve through the benchmark executable, including OpenVINO C++ calls that
interpose there. It excludes `malloc`, aligned allocation, calls that bypass
that interposition, and cannot count all driver or runtime allocations. The ABI
slot statistics record API-visible input/output bytes, not a complete process
allocation or transfer accounting.

The SDK receipt separately covers installation and model provenance. This
pilot qualifies only the recorded synchronous workload, model and Intel device;
external queue imports, asynchronous submission and other providers require
separate measurements.
