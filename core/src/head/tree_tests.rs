//! Tree decoding against brute force: every head assignment of a small
//! sentence enumerated, the trees kept, the best one found by summing.

use super::tree::{eisner, mst};
use crate::status::{INVALID_ARGUMENT, INVALID_SHAPE};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    }
}

/// Whether `heads` (word d's head at d - 1) is a tree with one word on
/// the root.
fn is_tree(heads: &[u32]) -> bool {
    let n = heads.len();
    if heads.iter().filter(|&&h| h == 0).count() != 1 {
        return false;
    }
    (1..=n).all(|d| {
        let mut v = d;
        for _ in 0..=n {
            if v == 0 {
                return true;
            }
            v = heads[v - 1] as usize;
        }
        false
    })
}

/// Whether no two arcs of the tree cross.
fn is_projective(heads: &[u32]) -> bool {
    let arcs: Vec<(usize, usize)> = heads
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            let (a, b) = (i + 1, h as usize);
            (a.min(b), a.max(b))
        })
        .collect();
    arcs.iter().all(|&(a, b)| arcs.iter().all(|&(c, d)| !(a < c && c < b && b < d)))
}

fn score(scores: &[f32], heads: &[u32]) -> f64 {
    let n = heads.len();
    heads.iter().enumerate().map(|(i, &h)| scores[i * (n + 1) + h as usize] as f64).sum()
}

/// The best score over every tree, or every projective tree.
fn brute(scores: &[f32], n: usize, projective: bool) -> f64 {
    let mut heads = vec![0u32; n];
    let mut best = f64::NEG_INFINITY;
    loop {
        if heads.iter().enumerate().all(|(i, &h)| h as usize != i + 1)
            && is_tree(&heads)
            && (!projective || is_projective(&heads))
        {
            best = best.max(score(scores, &heads));
        }
        let mut i = 0;
        loop {
            if i == n {
                return best;
            }
            heads[i] += 1;
            if heads[i] as usize <= n {
                break;
            }
            heads[i] = 0;
            i += 1;
        }
    }
}

#[test]
fn mst_finds_the_best_tree_and_eisner_the_best_projective_one() {
    let mut rng = Rng(0x5eed);
    for n in 1..=6 {
        for _ in 0..40 {
            let scores: Vec<f32> = (0..n * (n + 1)).map(|_| rng.next()).collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            mst(&scores, n, &mut a).unwrap();
            eisner(&scores, n, &mut b).unwrap();
            assert!(is_tree(&a), "{a:?}");
            assert!(is_tree(&b) && is_projective(&b), "{b:?}");
            assert!((score(&scores, &a) - brute(&scores, n, false)).abs() < 1e-9, "n {n}: {a:?}");
            assert!((score(&scores, &b) - brute(&scores, n, true)).abs() < 1e-9, "n {n}: {b:?}");
        }
    }
}

#[test]
fn a_root_heavy_sentence_still_gets_one_root() {
    // Every word prefers the root; the best one-root tree hangs the others
    // from the word the root suits best.
    let n = 4;
    let mut scores = vec![0f32; n * (n + 1)];
    for d in 0..n {
        scores[d * (n + 1)] = 10.0 + d as f32;
    }
    let mut heads = Vec::new();
    mst(&scores, n, &mut heads).unwrap();
    assert!(is_tree(&heads));
    assert_eq!(heads[n - 1], 0);
}

#[test]
fn ties_go_to_the_lower_index_every_time() {
    let n = 5;
    let flat = vec![0f32; n * (n + 1)];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    mst(&flat, n, &mut a).unwrap();
    assert_eq!(a, [0, 1, 1, 1, 1]);
    eisner(&flat, n, &mut b).unwrap();
    assert!(is_tree(&b) && is_projective(&b));
    let mut again = Vec::new();
    eisner(&flat, n, &mut again).unwrap();
    assert_eq!(again, b);
}

#[test]
fn empty_and_malformed_scores() {
    let mut heads = vec![7];
    mst(&[], 0, &mut heads).unwrap();
    assert!(heads.is_empty());
    eisner(&[], 0, &mut heads).unwrap();
    assert!(heads.is_empty());
    mst(&[1.0, 2.0], 1, &mut heads).unwrap();
    assert_eq!(heads, [0], "one word can only hang from the root");
    assert_eq!(mst(&[0.0; 5], 2, &mut heads).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(eisner(&[0.0; 7], 2, &mut heads).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(mst(&[0.0, f32::NAN], 1, &mut heads).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(eisner(&[f32::INFINITY, 0.0], 1, &mut heads).unwrap_err().code, INVALID_ARGUMENT);
}
