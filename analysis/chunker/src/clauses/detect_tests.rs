use turbo_analysis::{Ix, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Dependencies, Sentences, Tokens};

use super::STEPS;
use crate::{ClauseCosts, clause_cuts};

/// Steps the cut search takes on one sentence of `n` one-byte tokens where
/// each token's head is the one before it, every edge labelled `rel_name`.
fn chain_steps(n: usize, rel_name: &str) -> u64 {
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, (n * 2) as u32));
    let mut t_spans = Spans::new(ViewId::SOURCE);
    for i in 0..n {
        t_spans.push(Span::new((i * 2) as u32, (i * 2 + 1) as u32));
    }
    let tokens = Tokens { spans: t_spans, sentence: None };
    let deps = Dependencies::<Tokens> {
        head: (0..n).map(|i| if i == 0 { None } else { Some(Ix::new((i - 1) as u32)) }).collect(),
        relation: vec![Tag(0); n],
        relations: TagSet::new("dep", &[rel_name]),
    };
    let sentences = Sentences { spans: s_spans };
    STEPS.with(|s| s.set(0));
    let cuts = clause_cuts(&sentences, &tokens, Some(&deps), None, None, None, None, ClauseCosts::default());
    assert_eq!(cuts.sentences[0].tokens.len(), n);
    STEPS.with(std::cell::Cell::get)
}

/// A chain of `conj` edges makes every token a clause head whose subtree
/// runs to the end of the sentence. Walking each subtree on its own was
/// quadratic; the bounds are now precomputed in one walk, so eight times
/// the tokens is close to eight times the steps (a quadratic walk would be
/// 64) and a few steps a token. A chain of plain `dep` edges has no clause
/// relation and takes none of these steps.
#[test]
fn a_long_dependency_chain_takes_a_few_steps_a_token() {
    for rel_name in ["dep", "conj"] {
        let [n1, n8] = [400usize, 3_200].map(|n| chain_steps(n, rel_name));
        assert!(n8 * 10 <= n1 * 82, "{rel_name}: 8N/N steps {n8} against {n1}");
        assert!(n8 <= 4 * 3_200, "{rel_name}: {n8} steps for 3200 tokens");
    }
}
