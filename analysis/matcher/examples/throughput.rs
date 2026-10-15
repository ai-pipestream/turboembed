//! Throughput of the phrase matcher with a million phrases, and of the
//! token matcher with a thousand rules, on generated text.
//!
//! The words are drawn from a Zipf distribution over a 200,000-word
//! vocabulary (exponent 1, as word frequencies in English roughly
//! follow), and the phrases, one to four words long, from the same
//! distribution, so common words start many phrases and a document
//! touches many of them, as a gazetteer or a term list does.
//!
//! cargo run --release -p turbo-matcher --example throughput

use std::time::Instant;

use turbo_matcher::{Attr, Matcher, PhraseMatcher, PhraseOn, Sequence, Tok};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Word ranks with probability proportional to 1 / rank, by inverting the
/// cumulative sum.
struct Zipf {
    cumulative: Vec<f64>,
}

impl Zipf {
    fn new(n: usize) -> Zipf {
        let mut total = 0.0;
        let cumulative = (1..=n)
            .map(|r| {
                total += 1.0 / r as f64;
                total
            })
            .collect();
        Zipf { cumulative }
    }

    fn draw(&self, rng: &mut Rng) -> usize {
        let x = rng.unit() * self.cumulative.last().unwrap();
        self.cumulative.partition_point(|&c| c < x)
    }
}

fn word(rank: usize) -> String {
    // Distinct lowercase words from the rank, two to eight letters.
    let mut w = String::new();
    let mut r = rank + 27;
    while r > 0 {
        w.push((b'a' + (r % 26) as u8) as char);
        r /= 26;
    }
    w
}

fn main() {
    const VOCABULARY: usize = 200_000;
    const PHRASES: usize = 1_000_000;
    const TOKENS: usize = 5_000_000;
    let zipf = Zipf::new(VOCABULARY);
    let words: Vec<String> = (0..VOCABULARY).map(word).collect();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let t = Instant::now();
    let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Lower));
    let mut phrase = Vec::new();
    for i in 0..PHRASES {
        phrase.clear();
        let len = 1 + (rng.next() % 4) as usize;
        phrase.extend((0..len).map(|_| words[zipf.draw(&mut rng)].as_str()));
        b.add(["PLACE", "ORG", "TERM"][i % 3], &phrase).unwrap();
    }
    let phrases = b.build().unwrap();
    println!("built {PHRASES} phrases: {} trie nodes in {:.2?}", phrases.nodes(), t.elapsed());

    // A document: the words joined by spaces, with every tenth word
    // capitalized so the lowercase form is computed, not copied.
    let mut text = String::new();
    let (mut start, mut end) = (Vec::with_capacity(TOKENS), Vec::with_capacity(TOKENS));
    for i in 0..TOKENS {
        if i > 0 {
            text.push(' ');
        }
        start.push(text.len() as u32);
        let w = &words[zipf.draw(&mut rng)];
        if i % 10 == 0 {
            text.push_str(&w.to_uppercase());
        } else {
            text.push_str(w);
        }
        end.push(text.len() as u32);
    }
    let seq = Sequence::new(&text, &start, &end).unwrap();

    let t = Instant::now();
    let found = phrases.find(&seq).unwrap();
    let secs = t.elapsed().as_secs_f64();
    println!(
        "phrase matcher: {TOKENS} tokens, {} matches in {secs:.2} s: {:.1} M tokens/s, {:.0} MB/s of text",
        found.len(),
        TOKENS as f64 / secs / 1e6,
        text.len() as f64 / secs / 1e6
    );

    // A thousand token rules: a common word, an optional word, then one of
    // a set, by lowercase form.
    let mut b = Matcher::builder();
    for r in 0..1000 {
        let first = &words[zipf.draw(&mut rng)];
        let set: Vec<&str> = (0..5).map(|_| words[zipf.draw(&mut rng)].as_str()).collect();
        let pattern = vec![Tok::lower(first), Tok::any().optional(), Tok::any().is_in(Attr::Lower, &set)];
        b.add(&format!("R{r}"), vec![pattern], None).unwrap();
    }
    let rules = b.build().unwrap();
    let t = Instant::now();
    let found = rules.find(&seq).unwrap();
    let secs = t.elapsed().as_secs_f64();
    println!(
        "token matcher: 1000 rules, {TOKENS} tokens, {} matches in {secs:.2} s: {:.1} M tokens/s",
        found.len(),
        TOKENS as f64 / secs / 1e6
    );
}
