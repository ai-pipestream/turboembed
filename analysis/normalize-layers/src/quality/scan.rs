use turbo_analysis::{Error, Ix, Span, View};
use turbo_analysis_layers::{QualityMetrics, Sentences, Structure, Tokens};

use super::QualityMeasurement;
use super::lexical::{Diversity, Lexeme, Lexicon};
use super::stats::{Counts, Hyphens, TokenCounts};

/// Reusable capacities for token interning, MTLD marks, rolling trigrams,
/// region boundaries and summaries. No borrowed document data is retained.
#[derive(Default)]
pub struct QualityScratch {
    lexicon: Lexicon,
    diversity: Diversity,
    facts: Vec<(Span, Lexeme)>,
    sort_buffer: Vec<(Span, Lexeme)>,
    ids: Vec<usize>,
    regions: Vec<(Span, Counts)>,
    rows: Vec<(Span, QualityMetrics)>,
}

impl QualityScratch {
    pub(super) fn scan(
        &mut self,
        view: &View,
        tokens: &Tokens,
        sentences: &Sentences,
        structure: Option<&Structure>,
    ) -> Result<QualityMeasurement<'_>, Error> {
        self.lexicon.clear();
        self.facts.clear();
        self.regions.clear();
        self.rows.clear();
        let text = view.text();
        if let Some(blocks) = structure.filter(|b| !b.kind.is_empty()) {
            for (i, _) in blocks.spans.iter() {
                // In a validated preorder tree, a non-leaf's first child
                // immediately follows it. Leaves cannot overlap.
                if blocks.parent.get(i.get() + 1) != Some(&Some(i)) {
                    for span in blocks.spans.fragments(i).filter(|s| !s.is_empty()) {
                        self.regions.push((span, Counts::default()));
                    }
                }
            }
        } else {
            self.paragraphs(text);
        }
        for (i, span) in tokens.spans.iter() {
            let lexeme =
                self.lexicon.intern(tokens.spans.fragments(i).map(|s| &text[s.start as usize..s.end as usize]));
            self.facts.push((span, lexeme));
        }
        self.order_tokens();
        let mut all = Counts::default();
        let mut region = 0;
        let mut hyphens = Hyphens::default();
        for (offset, c) in text.char_indices() {
            let broken = hyphens.observe(c);
            all.observe(c, broken);
            while region < self.regions.len() && self.regions[region].0.end as usize <= offset {
                region += 1;
            }
            if let Some((span, counts)) = self.regions.get_mut(region)
                && span.start as usize <= offset
            {
                counts.observe(c, broken);
            }
        }
        let summary =
            self.summarize(all, 0, self.facts.len(), sentences.spans.iter().filter(|(_, s)| !s.is_empty()).count());
        let (mut tok, mut sent) = (0, 0);
        for region in 0..self.regions.len() {
            let (span, counts) = self.regions[region];
            while tok < self.facts.len() && self.facts[tok].0.start < span.start {
                tok += 1;
            }
            let mut end = tok;
            while end < self.facts.len() && self.facts[end].0.start < span.end {
                end += 1;
            }
            while sent < sentences.spans.len() && sentences.spans.get(Ix::new(sent as u32)).end <= span.start {
                sent += 1;
            }
            let mut sent_end = sent;
            while sent_end < sentences.spans.len() && sentences.spans.get(Ix::new(sent_end as u32)).start < span.end {
                sent_end += 1;
            }
            let metrics = self.summarize(counts, tok, end, sent_end - sent);
            self.rows.push((span, metrics));
        }
        Ok((summary, &self.rows))
    }

    fn order_tokens(&mut self) {
        if self.facts.windows(2).all(|w| w[0].0.start <= w[1].0.start) {
            return;
        }
        // Tokens legitimately allow overlaps and arbitrary order. Stable
        // four-pass radix sorting restores text order in O(T), including
        // widened UTF-16 tokens with identical starting offsets.
        self.sort_buffer.resize(self.facts.len(), (Span::new(0, 0), Lexeme { id: None, syllables: 0, content: false }));
        for shift in [0, 8, 16, 24] {
            let mut next = [0usize; 256];
            for (s, _) in &self.facts {
                next[((s.start >> shift) & 255) as usize] += 1;
            }
            let mut sum = 0;
            for n in &mut next {
                let count = *n;
                *n = sum;
                sum += count;
            }
            for &fact in &self.facts {
                let bucket = ((fact.0.start >> shift) & 255) as usize;
                self.sort_buffer[next[bucket]] = fact;
                next[bucket] += 1;
            }
            std::mem::swap(&mut self.facts, &mut self.sort_buffer);
        }
    }

    fn summarize(&mut self, counts: Counts, start: usize, end: usize, sentences: usize) -> QualityMetrics {
        self.ids.clear();
        let (mut syllables, mut content) = (0, 0);
        for &(_, lexeme) in &self.facts[start..end] {
            if let Some(id) = lexeme.id {
                self.ids.push(id);
            }
            syllables += lexeme.syllables;
            content += usize::from(lexeme.content);
        }
        let (mtld, repetition) = self.diversity.measure(&self.ids);
        counts.metrics(
            TokenCounts { tokens: end - start, words: self.ids.len(), syllables, content, sentences },
            mtld,
            repetition,
        )
    }

    fn paragraphs(&mut self, text: &str) {
        let (mut offset, mut start) = (0, None);
        for line in text.split_inclusive('\n') {
            if line.trim().is_empty() {
                if let Some(start) = start.take() {
                    self.regions.push((Span::new(start, offset as u32), Counts::default()));
                }
            } else if start.is_none() {
                start = Some(offset as u32);
            }
            offset += line.len();
        }
        if let Some(start) = start {
            self.regions.push((Span::new(start, text.len() as u32), Counts::default()));
        }
    }
}
