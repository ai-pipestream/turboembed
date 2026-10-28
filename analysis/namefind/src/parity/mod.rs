//! Structural and engine-consistency checks on golden sentences and generated
//! patterns and texts. JVM goldens independently check Java behavior.
//!
//! The texts are seeded, so a failure repeats. `TURBO_TEST_PARITY_ROUNDS`
//! (default 1) multiplies the number of generated cases.

mod builtin;
mod user;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    /// A count from `around`, or one off either side of it.
    fn near(&mut self, around: &[usize]) -> usize {
        let n = *self.pick(around);
        match self.below(3) {
            0 => n.saturating_sub(1),
            1 => n,
            _ => n + 1,
        }
    }
}

fn parse_rounds(raw: Option<&str>) -> usize {
    raw.and_then(|r| r.parse().ok()).filter(|&r| r > 0).unwrap_or(1)
}

fn rounds() -> usize {
    parse_rounds(std::env::var("TURBO_TEST_PARITY_ROUNDS").ok().as_deref())
}

fn read_data(relative: &str) -> String {
    let path = turbo_goldens::data_dir().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The golden sentences, the hand-made ones and those the `generated`
/// golden lists, their tokens joined by spaces as `RegexNameFinder.find`
/// joins them.
fn golden_texts() -> Vec<String> {
    use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape};
    let mut texts: Vec<String> = read_data("inputs/namefind/sentences.txt").lines().map(String::from).collect();
    const SENTENCES: &[ColumnSpec] =
        &[ColumnSpec::new("sentence", ColumnType::Str), ColumnSpec::new("token", ColumnType::Str)];
    let shape = Shape { id: "turbo:sentences".into(), anchor: GoldenAnchor::Document, columns: SENTENCES };
    let path = turbo_goldens::data_dir().join("namefind/generated/namefind/sentences.golden");
    let g = Golden::read_with(&path, &[shape]).unwrap_or_else(|e| panic!("{e}"));
    let l = g.layer("turbo:sentences").unwrap();
    let (keys, tokens) = (l.strs("sentence").unwrap(), l.strs("token").unwrap());
    let mut last: Option<&str> = None;
    let mut sentence: Vec<&str> = Vec::new();
    for (key, token) in keys.iter().zip(tokens.iter()) {
        if last.is_some_and(|k| k != key) {
            texts.push(sentence.join(" "));
            sentence.clear();
        }
        last = Some(key);
        sentence.push(token);
    }
    if !sentence.is_empty() {
        texts.push(sentence.join(" "));
    }
    texts
}

/// The byte offset of each character start, and of the end.
fn char_starts(text: &str) -> Vec<usize> {
    text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect()
}

/// The start offsets to search from: all of them in a short text, a
/// sample in a long one, always with the first and the last.
fn froms(rng: &mut Rng, chars: usize) -> Vec<usize> {
    if chars <= 160 {
        return (0..=chars).collect();
    }
    let mut out = vec![0, chars];
    out.extend((0..24).map(|_| rng.below(chars + 1)));
    out
}
