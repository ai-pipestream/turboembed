//! Viterbi over jieba's trained BMES model (MIT, see data/jieba-hmm).
//! The immutable packed table needs no parsing or heap allocation.

use turbo_analysis::Span;

const MODEL: &[u8] = include_bytes!("../../data/jieba-hmm/bmes.bin");
pub(super) const SHA256: &str = "b72dbb9f25a088473b0f7b4bb892514bbadbd8a510d6360449c29b0e64054bd5";
const INDEX: usize = 20 * 8;
const EMISSIONS: usize = INDEX + 65536 * 2;
// B, E, M, S in lexical order; the larger state wins an exact tie.
const PREVIOUS: [[usize; 2]; 4] = [[1, 3], [0, 2], [0, 2], [1, 3]];

#[derive(Debug, Default, Clone)]
pub(super) struct Scratch {
    previous: Vec<[u8; 4]>,
    path: Vec<u8>,
}

fn number(offset: usize) -> f64 {
    f64::from_le_bytes(MODEL[offset..offset + 8].try_into().unwrap())
}

fn row(c: char) -> usize {
    if c as u32 >= 65536 {
        return 0;
    }
    let i = INDEX + c as usize * 2;
    usize::from(u16::from_le_bytes([MODEL[i], MODEL[i + 1]]))
}

pub(super) fn observes(c: char) -> bool {
    row(c) != 0
}

pub(super) fn cut(chars: &[char], offsets: &[u32], base: u32, scratch: &mut Scratch, out: &mut Vec<Span>) {
    if chars.is_empty() {
        return;
    }
    scratch.previous.clear();
    scratch.previous.resize(chars.len(), [0; 4]);
    let mut scores = [0.0; 4];
    let emission = EMISSIONS + row(chars[0]) * 32;
    for (state, score) in scores.iter_mut().enumerate() {
        *score = number(state * 8) + number(emission + state * 8);
    }
    for (i, &c) in chars.iter().enumerate().skip(1) {
        let emission = EMISSIONS + row(c) * 32;
        let mut next = [0.0; 4];
        for state in 0..4 {
            let [a, b] = PREVIOUS[state];
            let emit = number(emission + state * 8);
            let from = |p| scores[p] + number((4 + p * 4 + state) * 8) + emit;
            let (sa, sb) = (from(a), from(b));
            let p = if sa > sb { a } else { b };
            next[state] = if sa > sb { sa } else { sb };
            scratch.previous[i][state] = p as u8;
        }
        scores = next;
    }
    let mut state = if scores[1] > scores[3] { 1 } else { 3 };
    scratch.path.resize(chars.len(), 0);
    for i in (0..chars.len()).rev() {
        scratch.path[i] = state as u8;
        state = scratch.previous[i][state] as usize;
    }
    let mut start = 0;
    for (i, &state) in scratch.path.iter().enumerate() {
        if state == 1 || state == 3 {
            out.push(Span::new(base + offsets[start], base + offsets[i + 1]));
            start = i + 1;
        }
    }
    debug_assert_eq!(start, chars.len(), "the path ends in E or S");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_model_and_real_unknown_words() {
        assert_eq!(turbo_maxent::sha256(MODEL).to_hex(), SHA256);
        let text = "杭研";
        let chars: Vec<_> = text.chars().collect();
        let offsets: Vec<_> = text.char_indices().map(|(i, _)| i as u32).chain([text.len() as u32]).collect();
        let mut spans = Vec::new();
        cut(&chars, &offsets, 0, &mut Scratch::default(), &mut spans);
        let words: Vec<_> = spans.iter().map(|s| &text[s.range()]).collect();
        assert_eq!(words, ["杭研"]);
        assert!(!observes('𠀀'));
        assert!(!observes('🦀'));
    }

    #[test]
    fn viterbi_matches_exhaustive_paths_on_the_trained_model() {
        for text in ["杭研", "小明", "中国科学", "李小龙", "阿尔法"] {
            let chars: Vec<_> = text.chars().collect();
            let offsets: Vec<_> = text.char_indices().map(|(i, _)| i as u32).chain([text.len() as u32]).collect();
            let mut scratch = Scratch::default();
            let mut spans = Vec::new();
            cut(&chars, &offsets, 0, &mut scratch, &mut spans);
            let mut best = f64::NEG_INFINITY;
            let mut expected = Vec::new();
            for mut code in 0..4usize.pow(chars.len() as u32) {
                let path: Vec<_> = (0..chars.len())
                    .map(|_| {
                        let state = code % 4;
                        code /= 4;
                        state as u8
                    })
                    .collect();
                if !matches!(path[0], 0 | 3) || !matches!(path[chars.len() - 1], 1 | 3) {
                    continue;
                }
                let mut score =
                    number(path[0] as usize * 8) + number(EMISSIONS + row(chars[0]) * 32 + path[0] as usize * 8);
                for i in 1..chars.len() {
                    let (a, b) = (path[i - 1] as usize, path[i] as usize);
                    score = score + number((4 + a * 4 + b) * 8) + number(EMISSIONS + row(chars[i]) * 32 + b * 8);
                }
                if score > best || (score == best && path.iter().rev().cmp(expected.iter().rev()).is_gt()) {
                    best = score;
                    expected = path;
                }
            }
            assert_eq!(scratch.path, expected, "{text}");
        }
    }
}
