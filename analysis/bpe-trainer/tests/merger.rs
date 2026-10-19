//! The merge loop against a trainer written plainly: every pair counted
//! again from the words before each merge, the best found by a scan, and
//! the merge made left to right in every word. On random corpora both
//! make the same merges with the same counts, under orders by id and by
//! text, with pairs the order does not allow and with pairs set aside.

use std::collections::{BTreeMap, BTreeSet};

use turbo_bpe_trainer::merger::{Merger, Order};
use turbo_bpe_trainer::pieces::{Pair, Pieces, SymbolId};

/// Ties to the smaller pair of ids, any pair allowed.
struct ById;

impl Order for ById {
    type Key = ();
    fn key(&self, _: &Pieces, _: Pair) {}
}

/// Ties to the shorter and then smaller joined text; a pair whose text
/// holds "ca" is not allowed.
struct ByText;

impl Order for ByText {
    type Key = (usize, String);
    fn key(&self, p: &Pieces, pair: Pair) -> (usize, String) {
        let s = format!("{}{}", p.text(pair.left), p.text(pair.right));
        (s.chars().count(), s)
    }
    fn allows(&self, p: &Pieces, pair: Pair) -> bool {
        !format!("{}{}", p.text(pair.left), p.text(pair.right)).contains("ca")
    }
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// Random words of a few letters, with counts.
fn corpus(rng: &mut Rng) -> Vec<(String, u64)> {
    let letters = 1 + rng.below(4);
    (0..1 + rng.below(12))
        .map(|_| {
            let w: String = (0..1 + rng.below(14)).map(|_| char::from(b'a' + rng.below(letters) as u8)).collect();
            (w, 1 + rng.below(5))
        })
        .collect()
}

fn start<O: Order>(order: O, corpus: &[(String, u64)]) -> Merger<O> {
    let mut pieces = Pieces::default();
    let words = corpus
        .iter()
        .map(|(w, n)| (w.chars().map(|c| pieces.intern(&c.to_string())).collect::<Vec<SymbolId>>(), *n))
        .collect();
    Merger::new(order, pieces, words)
}

/// The plain trainer's state: the words as piece texts.
struct Plain {
    words: Vec<(Vec<String>, u64)>,
}

impl Plain {
    fn counts(&self) -> BTreeMap<(String, String), u64> {
        let mut c = BTreeMap::new();
        for (w, n) in &self.words {
            for p in w.windows(2) {
                *c.entry((p[0].clone(), p[1].clone())).or_default() += n;
            }
        }
        c
    }

    /// Merges `l r` left to right in every word, and adds to `seen` each
    /// pair on either side of each place it is merged, as it stands just
    /// after that place is merged: a pair can be there only until the
    /// next place in the word is merged.
    fn merge(&mut self, l: &str, r: &str, seen: &mut BTreeSet<(String, String)>) {
        for (w, _) in &mut self.words {
            let mut i = 0;
            while i + 1 < w.len() {
                if w[i] == l && w[i + 1] == r {
                    let right = w.remove(i + 1);
                    w[i].push_str(&right);
                    if i > 0 {
                        seen.insert((w[i - 1].clone(), w[i].clone()));
                    }
                    if i + 1 < w.len() {
                        seen.insert((w[i].clone(), w[i + 1].clone()));
                    }
                }
                i += 1;
            }
        }
    }
}

/// Runs both to the end; `set_aside` says which candidates to set aside
/// rather than merge, given the texts made so far.
fn compare<O: Order>(
    order: O,
    plain_key: impl Fn(&Merger<O>, Pair) -> Option<O::Key>,
    corpus: &[(String, u64)],
    set_aside: bool,
) -> usize {
    let mut m = start(order, corpus);
    let mut plain = Plain { words: corpus.iter().map(|(w, n)| (w.chars().map(String::from).collect(), *n)).collect() };
    let mut made: BTreeSet<String> = BTreeSet::new();
    let mut aside: BTreeSet<(String, String)> = BTreeSet::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    let mut merged: BTreeSet<(String, String)> = BTreeSet::new();
    let mut steps = 0;
    loop {
        let counts = plain.counts();
        seen.extend(counts.keys().cloned());
        // The plain best: highest count, then smallest key, then smallest
        // pair of ids, among pairs allowed and not set aside.
        let id = |s: &str| m.pieces().get(s).expect("every piece is interned");
        let mut best: Option<(u64, O::Key, Pair)> = None;
        for ((l, r), &n) in &counts {
            if aside.contains(&(l.clone(), r.clone())) {
                continue;
            }
            let pair = Pair { left: id(l), right: id(r) };
            let Some(key) = plain_key(&m, pair) else { continue };
            let better = match &best {
                None => true,
                Some((bn, bk, bp)) => n > *bn || (n == *bn && (key < *bk || (key == *bk && pair < *bp))),
            };
            if better {
                best = Some((n, key, pair));
            }
        }
        let got = m.best();
        match (&best, got) {
            (None, None) => break,
            (Some((n, _, pair)), Some(c)) => {
                assert_eq!((c.pair, c.count), (*pair, *n), "step {steps} of {corpus:?}");
            }
            _ => panic!("step {steps} of {corpus:?}: {best:?} {got:?}", best = best.map(|b| (b.0, b.2))),
        }
        let pair = got.unwrap().pair;
        let (l, r) = (m.pieces().text(pair.left).to_owned(), m.pieces().text(pair.right).to_owned());
        let text = format!("{l}{r}");
        if set_aside && made.contains(&text) {
            m.discard(pair);
            aside.insert((l, r));
            continue;
        }
        made.insert(text);
        m.merge(pair);
        plain.merge(&l, &r, &mut seen);
        merged.insert((l, r));
        steps += 1;
        // The words agree after every merge.
        let words: Vec<Vec<&str>> = m.words().map(|(w, _)| w.iter().map(|&s| m.pieces().text(s)).collect()).collect();
        let plain_words: Vec<Vec<&str>> =
            plain.words.iter().map(|(w, _)| w.iter().map(String::as_str).collect()).collect();
        assert_eq!(words, plain_words, "{corpus:?}");
    }
    // The pairs left: those seen, even for a moment, gone from the words,
    // allowed, neither merged nor set aside.
    let counts = plain.counts();
    let mut spent: Vec<(O::Key, Pair)> = seen
        .iter()
        .filter(|p| !counts.contains_key(*p) && !merged.contains(*p) && !aside.contains(*p))
        .filter_map(|(l, r)| {
            let pair = Pair { left: m.pieces().get(l).unwrap(), right: m.pieces().get(r).unwrap() };
            plain_key(&m, pair).map(|k| (k, pair))
        })
        .collect();
    spent.sort();
    assert_eq!(m.spent(), spent.into_iter().map(|(_, p)| p).collect::<Vec<_>>(), "{corpus:?}");
    steps
}

#[test]
fn the_merge_loop_makes_the_merges_a_plain_recount_makes() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut steps = 0;
    for _ in 0..400 {
        let c = corpus(&mut rng);
        steps += compare(ById, |_, _| Some(()), &c, false);
        steps += compare(ById, |_, _| Some(()), &c, true);
        let key = |m: &Merger<ByText>, p: Pair| {
            let t = ByText;
            t.allows(m.pieces(), p).then(|| t.key(m.pieces(), p))
        };
        steps += compare(ByText, key, &c, false);
        steps += compare(ByText, key, &c, true);
    }
    assert!(steps > 10_000, "{steps}");
}
