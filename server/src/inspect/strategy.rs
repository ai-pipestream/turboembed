//! Strategy comparison matrix (`greedy_sentences`, `structure_only`, and
//! `semantic_dp` chunking runs) for `ProjectChunks` (issue `#1220`).

use turbo_analysis::{Annotator, Document, Ix, IxRange, Span, Spans};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_chunker::compare::{Pricing, Run, compare};
use turbo_chunker::{Budget, Chunker, ClauseCuts, Costs, StructureBoundaries};

use super::bindings::require_default_binding;
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::Offsetter;
use crate::proto::result::{BudgetSpec, StrategyComparisonEntryProto, StrategyMatrixProto};

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn greedy_sentence_chunks(sentences: &Sentences, subwords: &Subwords, budget: Budget) -> Chunks {
    let view = sentences.spans.view();
    let n = sentences.spans.len();
    let (sub_starts, sub_ends) = subwords.spans.columns();
    let mut spans = Spans::<Chunks>::new(view);
    let mut sent_ranges = Vec::new();
    let mut cuts = Vec::new();
    let mut costs = Vec::new();
    let mut model_tokens = Vec::new();
    let mut over_budget = Vec::new();

    if n > 0 {
        let mut cur_start = 0usize;
        let mut cur_tokens = budget.overhead;
        for i in 0..n {
            let sp = sentences.spans.get(Ix::new(i as u32));
            let s_sub = sub_ends.partition_point(|&e| e <= sp.start);
            let e_sub = sub_starts.partition_point(|&s| s < sp.end).max(s_sub);
            let tok_count = (e_sub - s_sub) as u32;
            if i > cur_start && cur_tokens.saturating_add(tok_count) > budget.max {
                let first = sentences.spans.get(Ix::new(cur_start as u32));
                let last = sentences.spans.get(Ix::new((i - 1) as u32));
                spans.push(Span::new(first.start, last.end));
                sent_ranges.push(IxRange::new(cur_start as u32, i as u32));
                cuts.push(if cur_start == 0 { Cut::DocumentStart } else { Cut::Sentence });
                costs.push(0);
                model_tokens.push(cur_tokens);
                over_budget.push(cur_tokens > budget.max);
                cur_start = i;
                cur_tokens = budget.overhead.saturating_add(tok_count);
            } else {
                cur_tokens = cur_tokens.saturating_add(tok_count);
            }
        }
        let first = sentences.spans.get(Ix::new(cur_start as u32));
        let last = sentences.spans.get(Ix::new((n - 1) as u32));
        spans.push(Span::new(first.start, last.end));
        sent_ranges.push(IxRange::new(cur_start as u32, n as u32));
        cuts.push(if cur_start == 0 { Cut::DocumentStart } else { Cut::Sentence });
        costs.push(0);
        model_tokens.push(cur_tokens);
        over_budget.push(cur_tokens > budget.max);
    }

    let len = spans.len();
    Chunks {
        spans,
        level: vec![0; len],
        parent: vec![None; len],
        sentences: sent_ranges,
        cut: cuts,
        cost: costs,
        model_tokens,
        over_budget,
        prompt_overhead: budget.overhead,
    }
}

pub(super) fn build_strategy_matrix(
    doc: &Document,
    text: &str,
    offsetter: &Offsetter,
    requested: &[BudgetSpec],
) -> Result<Option<StrategyMatrixProto>, Failure> {
    let (Some(sentences), Some(subwords), Some(bounds)) =
        (doc.get::<Sentences>(), doc.get::<Subwords>(), doc.get::<Boundaries>())
    else {
        return Ok(None);
    };
    require_default_binding::<Boundaries, Sentences>(doc)?;
    require_default_binding::<ClauseCuts, Sentences>(doc)?;
    require_default_binding::<Chunks, Sentences>(doc)?;
    let structure = doc.get::<Structure>();
    let clauses = doc.get::<ClauseCuts>();

    let mut runs: Vec<(BudgetSpec, Budget, Chunks)> = Vec::new();
    if requested.is_empty() {
        let base_spec = BudgetSpec {
            name: "semantic_dp".into(),
            max_tokens: 256,
            target_tokens: 192,
            size_weight: 1000,
            overhead: 2,
        };
        let b = Budget { max: 256, target: 192, size_weight: 1000, overhead: 2 };
        let greedy = greedy_sentence_chunks(sentences, subwords, b);
        runs.push((BudgetSpec { name: "greedy_sentences".into(), ..base_spec.clone() }, b, greedy));

        let struct_bounds = StructureBoundaries { costs: Costs::default() }
            .annotate((sentences, structure))
            .map_err(|e| bad(format!("structure_only boundaries: {e}")))?
            .0;
        let struct_chunks = Chunker { budget: b }
            .annotate((sentences, subwords, &struct_bounds, structure, None, None))
            .map_err(|e| bad(format!("structure_only chunker: {e}")))?
            .0;
        runs.push((BudgetSpec { name: "structure_only".into(), ..base_spec.clone() }, b, struct_chunks));

        let sem_chunks = Chunker { budget: b }
            .annotate((sentences, subwords, bounds, structure, clauses, None))
            .map_err(|e| bad(format!("semantic_dp chunker: {e}")))?
            .0;
        runs.push((base_spec, b, sem_chunks));
    } else {
        for (idx, spec) in requested.iter().cloned().enumerate() {
            if spec.max_tokens == 0 {
                return Err(bad(format!("strategy_budgets[{idx}].max_tokens must be > 0")));
            }
            let b = Budget {
                max: spec.max_tokens,
                target: if spec.target_tokens == 0 { spec.max_tokens } else { spec.target_tokens },
                size_weight: spec.size_weight,
                overhead: spec.overhead,
            };
            let chunks = Chunker { budget: b }
                .annotate((sentences, subwords, bounds, structure, clauses, None))
                .map_err(|e| bad(format!("strategy_budgets[{idx}] chunker: {e}")))?
                .0;
            runs.push((spec, b, chunks));
        }
    }

    let baseline_chunks = doc.get::<Chunks>().or_else(|| runs.last().map(|(_, _, c)| c));
    let mut entries = Vec::with_capacity(runs.len());
    for (spec, b, chunks) in &runs {
        let (matched, disagreements) = if let Some(base) = baseline_chunks {
            let run_a = Run {
                chunks: base,
                execution: None,
                pricing: Some(Pricing { sentences, boundaries: bounds, budget: *b }),
            };
            let run_b =
                Run { chunks, execution: None, pricing: Some(Pricing { sentences, boundaries: bounds, budget: *b }) };
            let rep = compare(run_a, run_b).map_err(|e| bad(format!("strategy compare: {e}")))?;
            (rep.matched as u32, rep.disagreements.len() as u32)
        } else {
            (0, 0)
        };
        let over_budget_count = chunks.over_budget.iter().filter(|&&ob| ob).count() as u32;
        let total_cost: i64 = chunks.cost.iter().copied().fold(0i64, i64::saturating_add);
        let starts_bytes: Vec<u32> =
            (0..chunks.spans.len()).map(|i| chunks.spans.get(Ix::new(i as u32)).start).collect();
        let cut_starts = offsetter.column_to_wire(text, &starts_bytes);
        entries.push(StrategyComparisonEntryProto {
            name: if spec.name.is_empty() { format!("max_{}", spec.max_tokens) } else { spec.name.clone() },
            budget: Some(spec.clone()),
            chunk_count: chunks.spans.len() as u32,
            over_budget_count,
            total_cost,
            matched_cuts_with_baseline: matched,
            disagreements_with_baseline: disagreements,
            cut_starts,
        });
    }

    Ok(Some(StrategyMatrixProto { strategies: entries }))
}
