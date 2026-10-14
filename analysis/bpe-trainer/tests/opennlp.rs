//! The OpenNLP trainer and tokenizer against the JVM's goldens
//! (analysis/goldens/data/bpe): `BPETokenizerTrainer` trained on each
//! input of the `bpe` set with 40 and with 400 merges, and the
//! `BPETokenizer` it made run over the same text; and against the JVM's
//! trainer on random corpora (testdata/opennlp-random.jsonl).
//!
//! The JVM takes, of pairs with the same count, the first in the
//! iteration order of a `HashMap`. So the checks are:
//!
//! - every merge the JVM made is a pair of the highest count at its step,
//!   counted here: the counting is the same;
//! - with the JVM's tie order, the trainer makes every one of the JVM's
//!   merges, on the goldens and on the random corpora;
//! - with the made-first order, it makes the JVM's merges up to the first
//!   step where the JVM chose among tied pairs;
//! - a model of the JVM's loop with explicit `HashMap` bins (insertion,
//!   `merge` at the head of a bin, resizing, iteration), written here
//!   apart from the trainer, makes every one of the JVM's merges, the
//!   emoji golden's surrogate merges among them;
//! - the tokenizer, given the JVM's merges, gives the JVM's tokens.

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

/// The trainer, with the JVM's tie order, makes every merge the JVM made
/// on every golden whose symbols are whole characters, and says so.
#[test]
fn the_trainer_makes_every_jvm_merge() {
    let mut merges = 0;
    for (g, jvm) in whole_character_goldens() {
        let limit: usize = g.run.trim_start_matches("merges-").parse().unwrap();
        let ours = OpenNlpBpe::train([g.text.as_str()], limit).unwrap();
        let ours: Vec<(String, String)> = ours.merges().iter().map(|m| (m.left.clone(), m.right.clone())).collect();
        assert_eq!(ours, jvm, "{} {}", g.run, g.input);
        assert!(OpenNlpBpe::train([g.text.as_str()], limit).unwrap().is_jvm_exact());
        merges += ours.len();
    }
    assert_eq!(merges, 686);
}

/// The JVM's `BPETokenizerTrainer` on 160 random corpora
/// (`testdata/opennlp-random.jsonl`, written by `OpenNlpRandom.java`):
/// small alphabets, so most steps tie, and corpora of up to 3000 words,
/// so the maps grow past 64 bins. The trainer makes the same 9907
/// merges.
#[test]
fn the_trainer_makes_the_jvms_merges_on_random_corpora() {
    let file = std::fs::read_to_string(repo().join("analysis/bpe-trainer/testdata/opennlp-random.jsonl")).unwrap();
    let (mut runs, mut merges) = (0, 0);
    let mut inexact = Vec::new();
    for (i, line) in file.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let text = v["text"].as_str().unwrap();
        let limit = v["num_merges"].as_u64().unwrap() as usize;
        let jvm: Vec<(String, String)> = v["merges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m[0].as_str().unwrap().to_owned(), m[1].as_str().unwrap().to_owned()))
            .collect();
        let ours = OpenNlpBpe::train([text], limit).unwrap();
        if !ours.is_jvm_exact() {
            inexact.push(i);
        }
        let ours: Vec<(String, String)> = ours.merges().iter().map(|m| (m.left.clone(), m.right.clone())).collect();
        assert_eq!(ours, jvm, "run {i}");
        runs += 1;
        merges += jvm.len();
    }
    assert_eq!(runs, 160);
    assert_eq!(merges, 9907);
    // These reach a tree bin (eight keys or more in a bin of a table of 64
    // or more) holding keys with equal hash codes, which the JVM orders by
    // identity hash code; the trainer says it cannot rule out a difference
    // there, and here there is none. Run 87 also reaches a tree bin, with
    // distinct hash codes, whose order is reproduced.
    assert_eq!(inexact, [19, 35, 39, 135, 155]);
}

/// With the other tie order the trainer makes the JVM's merges up to the
/// first step at which the JVM chose among pairs of equal count; there,
/// its own merge is one of them.
#[test]
fn made_first_makes_the_jvms_merges_until_the_first_tie() {
    let mut same_before_tie = 0;
    for (g, jvm) in whole_character_goldens() {
        let limit: usize = g.run.trim_start_matches("merges-").parse().unwrap();
        let ours = OpenNlpBpe::train_with([g.text.as_str()], limit, TieOrder::MadeFirst).unwrap();
        assert!(!ours.is_jvm_exact());
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
    assert!(!ours.is_jvm_exact(), "a character outside the BMP is reported");
    assert_eq!(symbols("smile😀"), ["s", "m", "i", "l", "e", "😀</w>"]);
    // Both run out of pairs before 40 merges; the JVM spends its extra
    // merges joining surrogates.
    assert!(ours.merges().len() < g.merges.len() && g.merges.len() < 40, "{} {}", ours.merges().len(), g.merges.len());
    let lone = |s: &[u16]| String::from_utf16(s).is_err();
    let joining = g.merges.iter().filter(|(l, r)| lone(l) || lone(r)).count();
    assert!(joining > 0);
}

// ---- A model of the JVM's loop, with Java's HashMap -------------------------------------

use turbo_jvm::hash_map::{JavaHash, JavaHashMap, Utf16String};

/// `BPETokenizerTrainer.learnMerges` over `text` as one corpus element,
/// with Java's HashMap.
fn jvm_merges(text: &str, num_merges: usize) -> Vec<Utf16Pair> {
    let utf16 = |s: &str| Utf16String(s.encode_utf16().collect());
    let end = utf16(END_OF_WORD);
    let mut word_freqs: JavaHashMap<Utf16String, u64> = JavaHashMap::new();
    let mut spans = Vec::new();
    turbo_tokenize::whitespace_spans(
        text,
        turbo_tokenize::TokenizerOptions { whitespace: turbo_tokenize::WhitespaceMode::Unicode, keep_newlines: false },
        &mut spans,
    );
    for s in &spans {
        let w = utf16(&text[s.start as usize..s.end as usize]);
        word_freqs.merge(w, 1, |acc, v| *acc += v);
    }
    let mut vocab: JavaHashMap<Vec<Utf16String>, u64> = JavaHashMap::new();
    for (w, &n) in word_freqs.iter() {
        let mut symbols: Vec<Utf16String> = w.0.iter().map(|&c| Utf16String(vec![c])).collect();
        symbols.last_mut().unwrap().0.extend_from_slice(&end.0);
        vocab.put(symbols, n);
    }
    let mut merges = Vec::new();
    for _ in 0..num_merges {
        let mut pair_counts: JavaHashMap<(Utf16String, Utf16String), u64> = JavaHashMap::new();
        for (symbols, &n) in vocab.iter() {
            for w in symbols.windows(2) {
                let p = (w[0].clone(), w[1].clone());
                pair_counts.merge(p, n, |acc, v| *acc += v);
            }
        }
        if pair_counts.is_empty() {
            break;
        }
        let mut best: Option<(&(Utf16String, Utf16String), u64)> = None;
        for (p, &n) in pair_counts.iter() {
            if n > best.map_or(0, |b| b.1) {
                best = Some((p, n));
            }
        }
        let best = best.unwrap().0.clone();
        let mut new_vocab: JavaHashMap<Vec<Utf16String>, u64> = JavaHashMap::new();
        for (symbols, &n) in vocab.iter() {
            let mut merged: Vec<Utf16String> = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == best.0 && symbols[i + 1] == best.1 {
                    merged.push(Utf16String([&best.0.0[..], &best.1.0[..]].concat()));
                    i += 2;
                } else {
                    merged.push(symbols[i].clone());
                    i += 1;
                }
            }
            new_vocab.merge(merged, n, |acc, v| *acc += v);
        }
        vocab.clear();
        vocab.put_all(new_vocab.into_ordered().into_iter());
        merges.push((best.0.0, best.1.0));
    }
    merges
}

/// The model of the JVM's loop makes the JVM's merges on every golden,
/// emoji's included: its tie order is the HashMap's.
#[test]
fn a_model_of_the_jvms_hash_map_makes_every_jvm_merge() {
    let mut merges = 0;
    for g in goldens() {
        let limit: usize = g.run.trim_start_matches("merges-").parse().unwrap();
        let model = jvm_merges(&g.text, limit);
        for (i, (a, b)) in model.iter().zip(&g.merges).enumerate() {
            assert_eq!(a, b, "{} {}: merge {i}", g.run, g.input);
        }
        assert_eq!(model.len(), g.merges.len(), "{} {}", g.run, g.input);
        merges += model.len();
    }
    assert!(merges > 700, "{merges}");
}

/// The hash codes are Java's: values a JDK printed.
#[test]
fn hash_codes_are_javas() {
    let u = |s: &str| Utf16String(s.encode_utf16().collect());
    assert_eq!(u("ab").java_hash(), 3105);
    assert_eq!((u("ab"), u("c")).java_hash(), 96354);
    assert_eq!(vec![u("a"), u("b")].java_hash(), 31 * (31 + 97) + 98);
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
