//! Builds a phrase matcher of a million phrases and matches it over real
//! text: the build time, the memory it holds and the matching throughput.
//!
//! ```text
//! cargo run --release -p turbo-matcher --example phrase_bench -- \
//!     TRAIN.conllu EVAL.conllu [EVAL.conllu ...]
//! ```
//!
//! The text is read from CoNLL-U files (the `# text =` line of each
//! sentence, and its tokens' forms found in it in order), such as the
//! Universal Dependencies English Web Treebank. The phrases are the
//! distinct one- to four-token n-grams of the first file, lowercased, in
//! the order they occur, then, when there are fewer than a million, random
//! two- to four-word phrases over the first file's words, from a fixed
//! seed. The matcher matches over `Attr::Lower`. The other files are
//! matched, each as one input, and the best of five runs is reported.

use std::collections::HashSet;
use std::time::Instant;

use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_matcher::{Attr, Field, Flag, Input, Matcher, PhraseMatcher, Tok};

const PHRASES: usize = 1_000_000;

struct Text {
    text: String,
    tokens: Tokens,
}

/// The sentences of a CoNLL-U file as one text, a newline after each, and
/// the tokens' spans in it. A multi-word token's range line is skipped,
/// and so are its words (they are not in the text as such).
fn read_conllu(path: &str) -> Text {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut base = 0;
    let mut cursor = 0;
    let mut skip_to = 0u32;
    for line in raw.lines() {
        if let Some(t) = line.strip_prefix("# text = ") {
            base = text.len();
            cursor = 0;
            skip_to = 0;
            text.push_str(t);
            text.push('\n');
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(id), Some(form)) = (cols.next(), cols.next()) else { continue };
        if let Some((_, last)) = id.split_once('-') {
            // A multi-word token: its surface form, then skip its words.
            skip_to = last.parse().unwrap_or(0);
            push_form(&text, base, &mut cursor, form, &mut spans);
            continue;
        }
        let Ok(n) = id.parse::<u32>() else { continue };
        if n <= skip_to {
            continue;
        }
        push_form(&text, base, &mut cursor, form, &mut spans);
    }
    Text { text, tokens: Tokens { spans, sentence: None } }
}

fn push_form(text: &str, base: usize, cursor: &mut usize, form: &str, spans: &mut Spans<Tokens>) {
    let sentence = &text[base..];
    if let Some(at) = sentence[*cursor..].find(form) {
        let start = base + *cursor + at;
        spans.push(Span::new(start as u32, (start + form.len()) as u32));
        *cursor += at + form.len();
    }
}

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).map(str::to_owned))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}

/// A small fixed-seed generator (xorshift64*), so the phrase list is the
/// same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn phrases(train: &Text) -> (Vec<Vec<String>>, usize) {
    let words: Vec<String> = (0..train.tokens.spans.len())
        .map(|i| {
            let s = train.tokens.spans.get(turbo_analysis::Ix::new(i as u32));
            train.text[s.range()].to_lowercase()
        })
        .collect();
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    let mut out = Vec::new();
    'grams: for n in 1..=4 {
        for w in words.windows(n) {
            if seen.insert(w.to_vec()) {
                out.push(w.to_vec());
                if out.len() == PHRASES {
                    break 'grams;
                }
            }
        }
    }
    let from_text = out.len();
    let mut vocab: Vec<&String> = words.iter().collect::<HashSet<_>>().into_iter().collect();
    vocab.sort();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    while out.len() < PHRASES {
        let n = 2 + (rng.next() % 3) as usize;
        let p: Vec<String> = (0..n).map(|_| vocab[(rng.next() % vocab.len() as u64) as usize].clone()).collect();
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }
    (out, from_text)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "usage: phrase_bench TRAIN.conllu EVAL.conllu [EVAL.conllu ...]");
    let train = read_conllu(&args[0]);
    let (list, from_text) = phrases(&train);
    let tokens: usize = list.iter().map(Vec::len).sum();
    println!(
        "phrases: {} ({} n-grams of {}, {} random), {} tokens",
        list.len(),
        from_text,
        args[0],
        list.len() - from_text,
        tokens
    );

    let rss_before = rss_kib();
    let started = Instant::now();
    let mut m = PhraseMatcher::new(Field::Attr(Attr::Lower));
    let mut refs: Vec<&str> = Vec::new();
    for (i, p) in list.iter().enumerate() {
        refs.clear();
        refs.extend(p.iter().map(String::as_str));
        m.add(if i < from_text { "NGRAM" } else { "RANDOM" }, &refs).unwrap();
    }
    let build = started.elapsed();
    let rss_after = rss_kib();
    println!(
        "build: {:.2} s, {} trie states, {} values, {:.1} MiB held by the matcher, RSS grew {:.1} MiB",
        build.as_secs_f64(),
        m.states(),
        m.vocabulary(),
        m.heap_bytes() as f64 / (1 << 20) as f64,
        rss_after.saturating_sub(rss_before) as f64 / 1024.0
    );

    let mut patterns = Matcher::new();
    let title = || Tok::flag(Flag::Title, true);
    patterns.add("NAME", &[title(), title().plus()]).unwrap();
    patterns.add("NUMBER_UNIT", &[Tok::flag(Flag::LikeNum, true), Tok::flag(Flag::Alpha, true)]).unwrap();
    patterns.add("QUOTED", &[Tok::text("\""), (!Tok::text("\"")).repeat(1, Some(8)), Tok::text("\"")]).unwrap();

    for path in &args[1..] {
        let doc = read_conllu(path);
        let input = Input::new(&doc.text, &doc.tokens).unwrap();
        let n = input.len();
        let mut best = f64::MAX;
        let mut found = 0;
        for _ in 0..5 {
            let t = Instant::now();
            found = m.find(&input).unwrap().len();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{path}: {n} tokens, {:.2} MB: {found} phrase matches in {:.1} ms, {:.2} M tokens/s, {:.1} MB/s",
            doc.text.len() as f64 / 1e6,
            best * 1e3,
            n as f64 / best / 1e6,
            doc.text.len() as f64 / best / 1e6
        );
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            found = patterns.find(&input).unwrap().len();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{path}: {found} token pattern matches (3 patterns) in {:.1} ms, {:.2} M tokens/s",
            best * 1e3,
            n as f64 / best / 1e6
        );
    }
}
