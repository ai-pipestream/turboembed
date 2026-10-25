use turbo_analysis_layers::{QualityFlags as F, QualityMetrics};
use turbo_unicode::{category::general_category, java};

pub(super) struct TokenCounts {
    pub tokens: usize,
    pub words: usize,
    pub syllables: usize,
    pub content: usize,
    pub sentences: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub chars: usize,
    pub letters: usize,
    pub alpha: usize,
    pub digits: usize,
    pub space: usize,
    pub symbols: usize,
    pub replacements: usize,
    pub mojibake: usize,
    pub pipes: usize,
    pub terminal: usize,
    pub hyphens: usize,
}

impl Counts {
    pub fn observe(&mut self, c: char, broken_hyphen: bool) {
        self.chars += 1;
        let category = general_category(c);
        let letter = java::is_letter(c);
        let alpha = letter || category.is_mark();
        let digit = java::is_digit(c);
        let space = java::is_whitespace(c) || category.is_separator();
        self.letters += usize::from(letter);
        self.alpha += usize::from(alpha);
        self.digits += usize::from(digit);
        self.space += usize::from(space);
        self.symbols += usize::from(!(alpha || digit || space));
        self.replacements += usize::from(c == '\u{fffd}');
        self.mojibake += usize::from(c == '\u{fffd}' || ('\u{80}'..='\u{9f}').contains(&c));
        self.pipes += usize::from(c == '|');
        self.terminal += usize::from(matches!(c, '.' | '!' | '?'));
        self.hyphens += usize::from(broken_hyphen);
    }

    pub fn metrics(self, counts: TokenCounts, mtld: f32, repetition: f32) -> QualityMetrics {
        let TokenCounts { tokens, words, syllables, content, sentences } = counts;
        let ratio = |n: usize| n as f32 / self.chars.max(1) as f32;
        let visible = self.chars - self.space;
        let mut flags = F::CLEAN;
        if self.mojibake > 0 {
            flags |= F::MOJIBAKE_SUSPECT;
        }
        if self.hyphens >= 2 && self.hyphens as f64 / words.max(1) as f64 >= 0.02 {
            flags |= F::OCR_BROKEN_HYPHENS;
        }
        if visible >= 8 && self.symbols as f64 / visible as f64 > 0.30 {
            flags |= F::HIGH_SYMBOL_NOISE;
        }
        if visible >= 8 && self.digits as f64 / visible as f64 > 0.50 {
            flags |= F::HIGH_DIGIT_DENSITY;
        }
        if words >= 12 && repetition >= 0.50 {
            flags |= F::REPETITIVE_NGRAM;
        }
        if self.pipes >= 2 && words <= (self.pipes + 1) * 5 && self.terminal == 0 {
            flags |= F::NAV_BOILERPLATE;
        }
        if words < 20 {
            flags |= F::TOO_SHORT;
        }
        let density = content as f32 / tokens.max(1) as f32;
        let score = if words == 0 {
            0.0
        } else {
            let base =
                0.60 * self.alpha as f32 / visible.max(1) as f32 + 0.25 * density + 0.15 * (mtld / 50.0).min(1.0);
            let penalties = [
                (F::MOJIBAKE_SUSPECT, 0.25),
                (F::OCR_BROKEN_HYPHENS, 0.75),
                (F::HIGH_SYMBOL_NOISE, 0.4),
                (F::HIGH_DIGIT_DENSITY, 0.6),
                (F::REPETITIVE_NGRAM, 0.5),
                (F::NAV_BOILERPLATE, 0.3),
                (F::TOO_SHORT, 0.75),
            ];
            penalties.into_iter().fold(base, |s, (flag, penalty)| if flags.intersects(flag) { s * penalty } else { s })
        };
        let mut m = QualityMetrics {
            composite_score: score.clamp(0.0, 1.0),
            mtld_lexical_diversity: mtld,
            information_density: density,
            alpha_ratio: ratio(self.alpha),
            digit_ratio: ratio(self.digits),
            symbol_ratio: ratio(self.symbols),
            whitespace_ratio: ratio(self.space),
            replacement_ratio: ratio(self.replacements),
            flags,
            ..QualityMetrics::default()
        };
        if words > 0 {
            let (w, s) = (words as f64, sentences.max(1) as f64);
            m.flesch_kincaid_grade = (0.39 * w / s + 11.8 * syllables as f64 / w - 15.59) as f32;
            m.automated_readability_index =
                (4.71 * (self.letters + self.digits) as f64 / w + 0.5 * w / s - 21.43) as f32;
            m.coleman_liau_index = (5.88 * self.letters as f64 / w - 29.6 * s / w - 15.8) as f32;
        }
        m
    }
}

/// A line-final hyphen between a letter and lowercase continuation.
#[derive(Default)]
pub(super) struct Hyphens {
    letter: bool,
    state: u8,
}

impl Hyphens {
    pub fn observe(&mut self, c: char) -> bool {
        let broken = self.state == 2 && general_category(c) == turbo_unicode::GeneralCategory::Ll;
        self.state = match (self.state, c) {
            (_, '-') if self.letter => 1,
            (1, '\r' | '\n') => 2,
            (1 | 2, ' ' | '\t') => self.state,
            (2, '\r' | '\n') => 2,
            _ => 0,
        };
        self.letter = java::is_letter(c);
        broken
    }
}
