# Chunker cut identity and DP margins

`turbo_chunker::segment_with_trace` runs the same integer DP as `segment`,
with one `DecisionMargin` for each selected segment. It records the best
suffix cost and the cheapest alternative with a different next cut. The
margin is `best_cost - second_best_cost`, in boundary-cost units: negative
for a strict preference, zero for a tie, and absent when only one next cut
fits. Equal totals choose the earliest unit end, then the earliest suffix
cuts. Source-ordered units therefore choose the earliest byte offsets.
The existing saturating i128 arithmetic also applies to margin subtraction.

The distinction between a different next cut and a different suffix path
matters. A tie several chunks later must not make an earlier, stable cut
appear ambiguous. The runner-up includes the optimal suffix for its own
next cut. Tracing adds one optional runner-up per DP state; ordinary
`segment` calls do not allocate that table.

`Chunker::chunk_with_trace` accepts the same sentence, subword, boundary,
structure and clause inputs as the annotator. It returns the chunk rows and
aligned decisions without changing the persisted `Chunks` schema.
`ChunkTrace::leaf_cuts()` supplies flat leaf spans and closing margins.
Parent rows and forced section boundaries have no margin. Clause fallback
rows also have no sentence-DP margin: their separate sub-sentence decisions
are outside this trace, so the comparator cannot classify them as ties.

## Comparing backend results

`turbo_chunker::eval::compare_chunk_cuts(a, margins_a, b, margins_b, epsilon)`
compares flat `Spans<Chunks>` from two runs of the same document. Call it
separately for each corpus document; equal view IDs do not prove equal text.
Inputs must use the same view and outer extent, with ordered, nonempty,
non-overlapping, continuous spans and one optional margin per chunk.
Margins must be nonpositive. Epsilon must be finite and nonnegative, in the
same integer cost units as the DP, not an unscaled floating-point tolerance.

A cut comprises a chunk's end and the following chunk's start, preserving
any gap. The final document end is fixed and excluded. The report contains
both chunk counts, matching cuts, union cut count, intersection/union
agreement rate (1 for no internal cuts), the first differing byte offset,
and both margins for unmatched cuts paired in source order between shared
cuts. Insertions and deletions have an absent side and cannot pass the
precision check. Different gaps count as differences even when starts agree.

`within_epsilon` means both cuts have available margins whose magnitudes
are at most epsilon. Integer magnitudes are compared with floor(epsilon)
without rounding the costs through f64. `None` never stands for a tie.
`all_within_epsilon` flags precision-sensitive disagreements; it does not
prove that a particular alternate partition costs within epsilon of the
best. Use `is_identical` for the strict cross-backend gate. The existing
`turbo_chunker::compare` API separately prices the other run's partition
when its `Pricing` inputs are available. Neither API executes a model or
substitutes for backend accuracy and throughput receipts.

Tests in `analysis/chunker/tests/cut_identity.rs` compare DP decisions to
exhaustive suffix enumeration, check deterministic ties, compare real
annotator and traced outputs, and cover gaps, missing margins, section and
clause boundaries, insertions, invalid inputs and large integer precision.
