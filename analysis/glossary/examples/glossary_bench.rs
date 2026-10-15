//! Times the two ways of finding glossary terms at token edges: the
//! character automaton ([`TextMatcher`]) with the tokens' edges as the
//! allowed boundaries, and turbo-matcher's phrase matcher over the
//! tokens' text ([`TokenMatcher`] with [`TokenValue::Text`]).
//!
//! The glossary and the text are made up from one vocabulary of
//! pseudo-words, so both matchers find the same terms. Run with
//! `cargo run --release -p turbo-glossary --example glossary_bench`.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use turbo_analysis::{Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_glossary::{Boundaries, Case, Glossary, TextMatcher, TokenMatcher, TokenValue};
use turbo_tokenize::word_tokens;

/// A small linear congruential generator, so runs are repeatable.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn vocabulary(rng: &mut Lcg, n: usize) -> Vec<String> {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    (0..n)
        .map(|_| {
            let len = 3 + rng.below(8);
            (0..len).map(|_| LETTERS[rng.below(26)] as char).collect()
        })
        .collect()
}

fn best_of(runs: usize, mut f: impl FnMut() -> usize) -> (Duration, usize) {
    let mut best = Duration::MAX;
    let mut found = 0;
    for _ in 0..runs {
        let t = Instant::now();
        found = f();
        best = best.min(t.elapsed());
    }
    (best, found)
}

fn main() {
    let mut rng = Lcg(202);
    let words = vocabulary(&mut rng, 20_000);
    for terms in [1_000, 50_000] {
        let mut glossary = Glossary::new();
        for i in 0..terms {
            let n = 1 + rng.below(3);
            let term: Vec<&str> = (0..n).map(|_| words[rng.below(words.len())].as_str()).collect();
            glossary.push(&format!("t{i}"), &term.join(" ")).unwrap();
        }
        // About 4 MB of text: the glossary's words and others, a space or
        // a comma and a space between them.
        let mut text = String::new();
        while text.len() < 4 << 20 {
            let w = if rng.below(4) == 0 {
                glossary.term(turbo_glossary::EntryIx(rng.below(terms) as u32))
            } else {
                &words[rng.below(words.len())]
            };
            text.push_str(w);
            text.push_str(if rng.below(10) == 0 { ", " } else { " " });
        }
        let (mut spans, mut types) = (Vec::new(), Vec::new());
        word_tokens(&text, NonZeroU32::new(255).unwrap(), &mut spans, &mut types);
        let mut token_spans = Spans::new(ViewId::SOURCE);
        for s in spans {
            token_spans.push(s);
        }
        let tokens = Tokens { spans: token_spans, sentence: None };
        let boundaries = Boundaries::of_tokens(text.len() as u32, &tokens);

        let t = Instant::now();
        let chars = TextMatcher::new(&glossary, Case::Sensitive).unwrap();
        let chars_build = t.elapsed();
        let t = Instant::now();
        let phrases =
            TokenMatcher::new(&glossary, TokenValue::Text, |term, out| out.extend(term.split(' ').map(str::to_string)))
                .unwrap();
        let phrases_build = t.elapsed();

        let mut out = Vec::new();
        let (chars_time, chars_found) = best_of(5, || {
            out.clear();
            chars.find(&text, &boundaries, &mut out);
            out.len()
        });
        let (scan_time, hits) = best_of(5, || {
            let mut n = 0;
            chars.scan(&text, |_, _| n += 1);
            n
        });
        let (phrases_time, phrases_found) = best_of(5, || {
            out.clear();
            phrases.find(&text, &tokens, None, &mut out).unwrap();
            out.len()
        });
        assert_eq!(chars_found, phrases_found, "the matchers disagree");
        let mb = text.len() as f64 / (1 << 20) as f64;
        println!("{terms} terms, {mb:.1} MB, {} tokens, {chars_found} matches", tokens.spans.len());
        println!("  character automaton scan alone: {:.1} ms, {hits} hits", scan_time.as_secs_f64() * 1e3);
        for (name, build, time, heap) in [
            ("character automaton", chars_build, chars_time, chars.heap_bytes()),
            ("phrase matcher", phrases_build, phrases_time, phrases.heap_bytes()),
        ] {
            println!(
                "  {name:20} build {:7.1} ms  match {:7.1} ms  {:6.0} MB/s  heap {:6.1} MB",
                build.as_secs_f64() * 1e3,
                time.as_secs_f64() * 1e3,
                mb / time.as_secs_f64(),
                heap as f64 / (1 << 20) as f64
            );
        }
    }
}
