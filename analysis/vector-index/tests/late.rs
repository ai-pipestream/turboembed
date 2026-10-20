//! MaxSim against a naive double loop, reranking against a full sort, and
//! the refusals of a token index.

mod common;

use common::{model, space, values};
use turbo_vector_index::{IndexError, LateHit, Normalization, Space, TokenIndex, TokenIndexBuilder, Which};

/// The crate's dot product written out: 16 lanes, element `i` in lane
/// `i % 16`, fused multiply-adds, then the lanes added pairwise.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        acc[i % 16] = x.mul_add(y, acc[i % 16]);
    }
    for half in [8, 4, 2, 1] {
        for j in 0..half {
            acc[j] += acc[j + half];
        }
    }
    acc[0]
}

/// For each query token the maximum over the candidate's tokens, the first
/// one reaching it, summed in query order.
fn naive(query: &[f32], cand: &[f32], dim: usize) -> (f32, Vec<u32>) {
    let mut total = 0.0f32;
    let mut won = Vec::new();
    for q in query.chunks(dim) {
        let mut best = (0u32, f32::NEG_INFINITY);
        for (j, c) in cand.chunks(dim).enumerate() {
            let s = dot(q, c);
            if s > best.1 {
                best = (j as u32, s);
            }
        }
        total += best.1;
        won.push(best.0);
    }
    (total, won)
}

fn candidates(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| values(seed + i as u64, (1 + (i * 37) % 181) * dim)).collect()
}

fn index_of(sp: &Space, cands: &[Vec<f32>]) -> TokenIndex<usize> {
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    for (i, c) in cands.iter().enumerate() {
        assert_eq!(b.push(c, i), Ok(i as u32));
    }
    b.build()
}

#[test]
fn a_score_is_the_naive_double_loop_bit_for_bit() {
    for dim in [7usize, 16, 96, 128] {
        let sp = space(dim as u32);
        let cands = candidates(40, dim, 3 * dim as u64);
        let index = index_of(&sp, &cands);
        for nq in [1usize, 3, 4, 5, 32] {
            let q = values(10_000 + nq as u64, nq * dim);
            for (c, cand) in cands.iter().enumerate() {
                let (want, won) = naive(&q, cand, dim);
                let got = index.score(&sp, &q, c as u32).unwrap();
                assert_eq!(got.to_bits(), want.to_bits(), "dim {dim}, {nq} query tokens, candidate {c}");
                let (score, matches) = index.explain(&sp, &q, c as u32).unwrap();
                assert_eq!(score.to_bits(), want.to_bits());
                assert_eq!(matches.iter().map(|m| m.token).collect::<Vec<_>>(), won);
                for (m, qt) in matches.iter().zip(q.chunks(dim)) {
                    let row = index.token(c as u32, m.token).unwrap();
                    assert_eq!(m.score.to_bits(), dot(qt, row).to_bits());
                    assert_eq!((m.source.piece, m.source.span), (m.token, None));
                }
            }
        }
    }
}

#[test]
fn rerank_and_top_k_are_the_head_of_a_full_sort() {
    let dim = 64;
    let sp = space(dim as u32);
    let mut cands = candidates(120, dim, 5);
    // Copies of candidates 3 and 10, at 120 and 121, tie with them.
    cands.push(cands[3].clone());
    cands.push(cands[10].clone());
    let index = index_of(&sp, &cands);
    let q = values(77, 8 * dim);
    let mut all: Vec<LateHit> =
        (0..cands.len() as u32).map(|c| LateHit { candidate: c, score: index.score(&sp, &q, c).unwrap() }).collect();
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    for k in [0usize, 1, 10, 121, 122, 500] {
        let want: Vec<LateHit> = all.iter().copied().take(k).collect();
        assert_eq!(index.top_k(&sp, &q, k).unwrap(), want, "k {k}");
    }
    let shortlist = [121u32, 3, 50, 10, 120, 7];
    let mut want: Vec<LateHit> =
        shortlist.iter().map(|&c| LateHit { candidate: c, score: index.score(&sp, &q, c).unwrap() }).collect();
    want.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.candidate.cmp(&b.candidate)));
    assert_eq!(index.rerank(&sp, &q, &shortlist, 6).unwrap(), want);
    assert_eq!(index.rerank(&sp, &q, &shortlist, 2).unwrap(), want[..2]);
    let tied = index.rerank(&sp, &q, &[120, 3], 2).unwrap();
    assert_eq!((tied[0].candidate, tied[1].candidate), (3, 120));
    assert_eq!(tied[0].score.to_bits(), tied[1].score.to_bits());
}

#[test]
fn queries_and_candidates_of_another_space_or_unscorable_tokens_get_typed_errors() {
    let sp = space(4);
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    assert_eq!(b.push(&[], 0), Err(IndexError::NoTokens(Which::Candidate(0))));
    assert_eq!(b.push(&[1.0; 6], 0), Err(IndexError::Dimension { index: 4, other: 6 }));
    let zero_second = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(b.push(&zero_second, 0), Err(IndexError::Zero(Which::Token(0, 1))));
    let nan_third = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, f32::NAN, 0.0];
    assert_eq!(b.push(&nan_third, 0), Err(IndexError::NonFinite(Which::Token(0, 2), 2)));
    assert!(b.is_empty());
    assert_eq!(b.push(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], 7), Ok(0));
    let index = b.build();
    assert_eq!((index.len(), index.total_tokens(), *index.meta(0)), (1, 2, 7));
    assert_eq!(index.token(0, 1), Some(&[0.0, 2.0, 0.0, 0.0][..]));
    let q = [0.0, 1.0, 0.0, 0.0];
    assert_eq!(index.score(&sp, &q, 0), Ok(2.0));
    let other = Space::new(model("colbert-small"), 4, Normalization::None);
    assert!(matches!(index.score(&other, &q, 0), Err(IndexError::Model { .. })));
    let mut projected = sp.clone();
    projected.model.sha256 = "projection-2".into();
    assert!(matches!(index.rerank(&projected, &q, &[0], 1), Err(IndexError::Model { .. })));
    let wide = Space::new(sp.model.clone(), 8, Normalization::None);
    assert_eq!(index.top_k(&wide, &[1.0; 8], 1), Err(IndexError::Dimension { index: 4, other: 8 }));
    let unit = Space::new(sp.model.clone(), 4, Normalization::Unit);
    assert!(matches!(index.score(&unit, &q, 0), Err(IndexError::Normalization { .. })));
    assert_eq!(index.score(&sp, &[], 0), Err(IndexError::NoTokens(Which::Query(0))));
    assert_eq!(index.score(&sp, &[1.0; 5], 0), Err(IndexError::Dimension { index: 4, other: 5 }));
    let zero_token = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(index.explain(&sp, &zero_token, 0), Err(IndexError::Zero(Which::Query(1))));
    assert_eq!(TokenIndexBuilder::<()>::new(space(0)).err(), Some(IndexError::ZeroDimension));
}

#[test]
fn a_unit_space_refuses_tokens_that_are_not_unit_length() {
    let sp = Space::new(model("colbert-small"), 2, Normalization::Unit);
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    assert!(matches!(b.push(&[1.0, 0.0, 3.0, 4.0], ()), Err(IndexError::NotUnit(Which::Token(0, 1), _))));
    b.push(&[1.0, 0.0, 0.6, 0.8], ()).unwrap();
    let index = b.build();
    assert!(matches!(index.score(&sp, &[2.0, 0.0], 0), Err(IndexError::NotUnit(Which::Query(0), _))));
    assert_eq!(index.score(&sp, &[0.0, 1.0, 1.0, 0.0], 0), Ok(0.8f32 + 1.0));
}

#[test]
fn concurrent_reranks_return_the_single_thread_result() {
    let dim = 96;
    let sp = space(dim as u32);
    let index = index_of(&sp, &candidates(200, dim, 11));
    let queries: Vec<Vec<f32>> = (0..16).map(|i| values(500 + i, 32 * dim)).collect();
    let want: Vec<_> = queries.iter().map(|q| index.top_k(&sp, q, 10).unwrap()).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| s.spawn(|| queries.iter().map(|q| index.top_k(&sp, q, 10).unwrap()).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), want);
        }
    });
}

#[test]
fn a_query_token_is_won_by_the_first_of_equal_candidate_tokens() {
    let sp = space(2);
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    // Tokens 1 and 3 equal, and 0 and 2 equal.
    b.push(&[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], ()).unwrap();
    let index = b.build();
    let (score, m) = index.explain(&sp, &[2.0, 0.0, 0.0, 3.0], 0).unwrap();
    assert_eq!(score, 5.0);
    assert_eq!(m.iter().map(|m| (m.token, m.score)).collect::<Vec<_>>(), [(1, 2.0), (0, 3.0)]);
}

#[test]
fn a_token_past_its_candidate_is_none() {
    let sp = space(1);
    let mut b = TokenIndexBuilder::new(sp).unwrap();
    b.push(&[1.0], ()).unwrap();
    b.push(&[2.0], ()).unwrap();
    b.push(&[3.0, 4.0], ()).unwrap();
    let index = b.build();
    assert_eq!(index.token(0, 0), Some(&[1.0][..]));
    assert_eq!(index.token(0, 1), None);
    assert_eq!(index.token(1, 1), None);
    assert_eq!(index.token(2, 1), Some(&[4.0][..]));
    assert_eq!(index.token(2, 2), None);
    assert_eq!(index.token(2, u32::MAX), None);
}
