//! The OpenNLP trainer and tokenizer against the reference goldens
//! (analysis/goldens/data/bpe): `BPETokenizerTrainer` trained on each
//! input of the `bpe` set with 40 and with 400 merges, and the
//! `BPETokenizer` it made run over the same text.
//!
//! Of pairs with the same count, `OpenNlpBpe` merges in deterministic
//! `MadeFirst` order (initial symbols in Unicode code point order, merged
//! symbols in the order they were created), whereas the JVM took the first
//! in the iteration order of a `HashMap`. So the checks are:
//!
//! - every merge the reference made is a pair of the highest count at its
//!   step, counted here: the counting is the same;
//! - with the made-first order (`OpenNlpBpe::train` and `TieOrder::MadeFirst`),
//!   the trainer makes the reference merges up to the first step where
//!   tied pairs occur, and at that tie chooses one of the top-counted pairs;
//! - the tokenizer, given the reference merges, gives the reference tokens.

mod common;

use common::*;

use turbo_bpe_trainer::merger::{Merger, Order};
use turbo_bpe_trainer::opennlp::{END_OF_WORD, symbols, words};
use turbo_bpe_trainer::pieces::{Pair, Pieces, SymbolId};
use turbo_bpe_trainer::{OpenNlpBpe, TieOrder};
use turbo_goldens::escape::unescape_utf16;

/// A merge as the JVM holds it: two strings of UTF-16 units.
type Utf16Pair = (Vec<u16>, Vec<u16>);

/// One golden of the `bpe` annotator.
struct BpeGolden {
    run: String,
    input: String,
    text: String,
    /// The merges, each side as UTF-16 units: a side can be half of a
    /// surrogate pair.
    merges: Vec<Utf16Pair>,
    /// The tokens' UTF-8 byte spans.
    tokens: Vec<(u32, u32)>,
}

impl BpeGolden {
    /// The merges as strings, when every side is whole characters.
    fn string_merges(&self) -> Option<Vec<(String, String)>> {
        self.merges.iter().map(|(l, r)| Some((String::from_utf16(l).ok()?, String::from_utf16(r).ok()?))).collect()
    }
}

/// Every golden, read by this test's own reader: the merges layer can
/// hold a lone surrogate, which `turbo_goldens`' reader refuses in a
/// `str` field.
fn goldens() -> Vec<BpeGolden> {
    let dir = repo().join("analysis/goldens/data/bpe");
    let mut out = Vec::new();
    for run in ["merges-40", "merges-400"] {
        for (input, text) in bpe_inputs() {
            let path = dir.join(run).join("bpe").join(format!("{input}.golden"));
            let file = std::fs::read_to_string(&path).unwrap();
            let mut lines = file.lines();
            let header = lines.by_ref().find(|l| l.starts_with("input\t")).unwrap();
            assert!(header.starts_with(&format!("input\tbpe/{input}\t{}\t", text.len())), "{header}");
            let mut merges = Vec::new();
            let mut tokens = Vec::new();
            let mut layer = "";
            for line in lines {
                let f: Vec<&str> = line.split('\t').collect();
                match f[0] {
                    "layer" => layer = if f[1] == "turbo:bpe-merges" { "merges" } else { "tokens" },
                    "end" | "config" | "annotator" => layer = "",
                    _ if layer == "merges" => {
                        merges.push((unescape_utf16(f[0]).unwrap(), unescape_utf16(f[1]).unwrap()));
                    }
                    _ if layer == "tokens" && f.len() == 2 => {
                        tokens.push((f[0].parse().unwrap(), f[1].parse().unwrap()));
                    }
                    _ => {}
                }
            }
            out.push(BpeGolden { run: run.into(), input, text, merges, tokens });
        }
    }
    assert_eq!(out.len(), 12);
    out
}

/// Ties to the smaller ids, as the trainer orders them; the replay below
/// does not depend on it.
struct ById;

impl Order for ById {
    type Key = Pair;
    fn key(&self, _: &Pieces, pair: Pair) -> Pair {
        pair
    }
}

/// The text's words as the trainer starts them, in a merger.
fn merger(text: &str) -> Merger<ById> {
    let mut pieces = Pieces::default();
    let words: Vec<(Vec<SymbolId>, u64)> =
        words([text]).into_iter().map(|(w, n)| (symbols(w).iter().map(|s| pieces.intern(s)).collect(), n)).collect();
    Merger::new(ById, pieces, words)
}

/// The goldens whose symbols are whole characters: all but emoji's.
fn whole_character_goldens() -> Vec<(BpeGolden, Vec<(String, String)>)> {
    goldens()
        .into_iter()
        .filter_map(|g| {
            let m = g.string_merges();
            match m {
                Some(m) => Some((g, m)),
                None => {
                    assert_eq!(g.input, "emoji", "{}: a lone surrogate", g.input);
                    None
                }
            }
        })
        .collect()
}

/// Each of the JVM's merges, made in turn here, is a pair of the highest
/// count when it is made, and the JVM stops where no pair is left or at
/// its number of merges.
#[test]
fn every_jvm_merge_is_a_pair_of_the_highest_count() {
    let mut steps = 0;
    let mut tied = 0;
    for (g, merges) in whole_character_goldens() {
        let mut m = merger(&g.text);
        for (i, (l, r)) in merges.iter().enumerate() {
            let p = m.pieces();
            let pair = Pair { left: p.get(l).unwrap(), right: p.get(r).unwrap() };
            let count = m.count(pair);
            let best = m.best().unwrap().count;
            assert_eq!(
                count, best,
                "{} {}: merge {i} ({l} {r}) occurs {count} times, the most is {best}",
                g.run, g.input
            );
            if m.counts().filter(|&(_, c)| c == best).count() > 1 {
                tied += 1;
            }
            m.merge(pair);
            steps += 1;
        }
        let limit: usize = g.run.trim_start_matches("merges-").parse().unwrap();
        assert!(merges.len() == limit || m.best().is_none(), "{} {}: stopped early", g.run, g.input);
    }
    // 10 goldens; most steps of these small texts are ties.
    assert!(steps > 600 && tied > 500, "{steps} steps, {tied} tied");
}

/// With [`TieOrder::MadeFirst`] (the default), the trainer makes the
/// reference merges up to the first step at which pairs of equal count
/// tie; there, its own merge is one of them.
#[test]
fn made_first_makes_the_jvms_merges_until_the_first_tie() {
    let mut same_before_tie = 0;
    for (g, jvm) in whole_character_goldens() {
        let limit: usize = g.run.trim_start_matches("merges-").parse().unwrap();
        let ours = OpenNlpBpe::train_with([g.text.as_str()], limit, TieOrder::MadeFirst).unwrap();
        assert_eq!(ours, OpenNlpBpe::train([g.text.as_str()], limit).unwrap());
        let ours: Vec<(&str, &str)> = ours.merges().iter().map(|m| (m.left.as_str(), m.right.as_str())).collect();
        let mut m = merger(&g.text);
        for (i, (l, r)) in jvm.iter().enumerate() {
            let best = m.best().unwrap().count;
            let tied: Vec<Pair> = m.counts().filter(|&(_, c)| c == best).map(|(p, _)| p).collect();
            let p = m.pieces();
            let text = |q: Pair| (p.text(q.left).to_owned(), p.text(q.right).to_owned());
            if tied.len() > 1 {
                let options: Vec<(String, String)> = tied.iter().map(|&q| text(q)).collect();
                let mine = (ours[i].0.to_owned(), ours[i].1.to_owned());
                // After a tie the two can take other paths, and even make
                // a different number of merges before no pair is left.
                assert!(options.contains(&mine), "{} {}: merge {i} {mine:?} is not one of {options:?}", g.run, g.input);
                break;
            }
            assert_eq!(ours[i], (l.as_str(), r.as_str()), "{} {}: merge {i}", g.run, g.input);
            same_before_tie += 1;
            let pair = Pair { left: p.get(l).unwrap(), right: p.get(r).unwrap() };
            m.merge(pair);
        }
    }
    // Small texts tie early: only prose's first 7 merges and ties' first
    // have no tie, in both runs.
    assert_eq!(same_before_tie, 16);
}

/// Given the JVM's merges, the tokenizer gives the JVM's tokens, `</w>`
/// written in a word included.
#[test]
fn the_tokenizer_gives_the_jvms_tokens_for_the_jvms_merges() {
    for (g, jvm) in whole_character_goldens() {
        let t = OpenNlpBpe::from_merges(jvm).tokenizer();
        assert_eq!(t.tokenize_pos(&g.text), g.tokens, "{} {}", g.run, g.input);
    }
}

/// A word that holds `</w>` as text: the JVM's merges join each such
/// word into one token, and taking the marker off that token leaves the
/// word with its own `</w>`; so does this tokenizer.
#[test]
fn end_of_word_written_in_a_word_is_kept_as_the_jvm_keeps_it() {
    let g = goldens().into_iter().find(|g| g.run == "merges-400" && g.input == "end-marker").unwrap();
    assert!(g.text.contains(END_OF_WORD));
    let mut words = Vec::new();
    let mut at = 0;
    for w in g.text.trim_end().split(' ') {
        words.push((at, at + w.len() as u32));
        at += w.len() as u32 + 1;
    }
    assert_eq!(g.tokens, words);
    let t = OpenNlpBpe::from_merges(g.string_merges().unwrap()).tokenizer();
    assert_eq!(t.tokenize_pos(&g.text), g.tokens);
    assert_eq!(t.tokenize("x</w> a</w>b"), ["x</w>", "a</w>b"]);
}

/// The JVM's symbols are UTF-16 units: it splits 😀 into its surrogates
/// and merges them back as one of its first merges, where the trainer
/// has 😀 as one symbol from the start.
#[test]
fn the_jvm_splits_a_character_outside_the_bmp_and_the_trainer_does_not() {
    let g = goldens().into_iter().find(|g| g.run == "merges-40" && g.input == "emoji").unwrap();
    let emoji: Vec<u16> = "😀".encode_utf16().collect();
    let first = &g.merges[0];
    assert_eq!(first.0, emoji[..1]);
    assert_eq!(first.1, [&emoji[1..], &END_OF_WORD.encode_utf16().collect::<Vec<_>>()[..]].concat());
    let ours = OpenNlpBpe::train([g.text.as_str()], 40).unwrap();
    assert!(ours.merges().iter().all(|m| !m.left.is_empty() && !m.right.is_empty()));
    assert_eq!(symbols("smile😀"), ["s", "m", "i", "l", "e", "😀</w>"]);
    // Both run out of pairs before 40 merges; the JVM spends its extra
    // merges joining surrogates.
    assert!(ours.merges().len() < g.merges.len() && g.merges.len() < 40, "{} {}", ours.merges().len(), g.merges.len());
    let lone = |s: &[u16]| String::from_utf16(s).is_err();
    let joining = g.merges.iter().filter(|(l, r)| lone(l) || lone(r)).count();
    assert!(joining > 0);
}

/// With [`TieOrder::MadeFirst`], training twice gives the same merges,
/// and so does the corpus in another order: ties are broken by the
/// symbols, not by the order the words were seen in.
#[test]
fn made_first_is_deterministic_and_ignores_the_order_of_the_texts() {
    let mut texts: Vec<String> = bpe_inputs().into_iter().map(|(_, t)| t).collect();
    texts.extend(tokenizer_texts());
    let train = |t: Vec<&str>| OpenNlpBpe::train_with(t, 1500, TieOrder::MadeFirst).unwrap();
    let a = train(texts.iter().map(String::as_str).collect());
    let b = train(texts.iter().map(String::as_str).collect());
    let c = train(texts.iter().rev().map(String::as_str).collect());
    assert_eq!(a.merges().len(), 1500);
    assert_eq!(a, b);
    assert_eq!(a, c);
}

/// Initial symbols are ordered by standard Rust UTF-8 / Unicode code point
/// order (`str::cmp`), so a BMP symbol above U+DFFF (`\u{FB00}`, `'ﬀ'`)
/// sorts before a supplementary symbol (`\u{1F600}`, `'😀'`).
#[test]
fn initial_symbols_order_by_unicode_code_point() {
    let m = OpenNlpBpe::train(["\u{1F600}a \u{FB00}a"], 1).unwrap();
    assert_eq!(m.merges().len(), 1);
    assert_eq!((m.merges()[0].left.as_str(), m.merges()[0].right.as_str()), ("\u{FB00}", "a</w>"));
}
