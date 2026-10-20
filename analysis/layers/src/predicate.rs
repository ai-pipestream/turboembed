//! [`FilterableLayer`] implementations for the standard layers in
//! `turbo-analysis-layers`.

use std::sync::Arc;

use turbo_analysis::{
    Anchor, Document, Error, FilterableLayer, Ix, KeyId, Layer, Provenance, co_written_aligned_anchors,
    retain_aligned_anchor,
};

use crate::{
    Abbreviations, Boilerplate, Categories, CategoryScope, ChunkTags, Confidence, Entities, LanguageSpans, Lemmas,
    Matches, PhraseChunks, Pii, PosTags, Sentences, Stems, Strs, Terms, Tokens,
};

impl FilterableLayer for Sentences {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Sentences { spans: self.spans.retain_indices(kept) }
    }
}

impl FilterableLayer for Tokens {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Tokens {
            spans: self.spans.retain_indices(kept),
            sentence: self.sentence.as_ref().map(|s| kept.iter().map(|&ix| s[ix.get()]).collect()),
        }
    }
}

impl FilterableLayer for Entities {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Entities {
            spans: self.spans.retain_indices(kept),
            tokens: kept.iter().map(|&ix| self.tokens[ix.get()]).collect(),
            kind: kept.iter().map(|&ix| self.kind[ix.get()]).collect(),
            kinds: Arc::clone(&self.kinds),
            score: kept.iter().map(|&ix| self.score[ix.get()]).collect(),
        }
    }
}

impl FilterableLayer for PhraseChunks {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        PhraseChunks {
            spans: self.spans.retain_indices(kept),
            tokens: kept.iter().map(|&ix| self.tokens[ix.get()]).collect(),
            kind: kept.iter().map(|&ix| self.kind[ix.get()]).collect(),
            kinds: Arc::clone(&self.kinds),
        }
    }
}

impl FilterableLayer for Matches {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Matches {
            spans: self.spans.retain_indices(kept),
            tokens: kept.iter().map(|&ix| self.tokens[ix.get()]).collect(),
            label: kept.iter().map(|&ix| self.label[ix.get()]).collect(),
            labels: Arc::clone(&self.labels),
            rule: kept.iter().map(|&ix| self.rule[ix.get()]).collect(),
        }
    }
}

impl FilterableLayer for Pii {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        let mut normalized = Strs::default();
        for &ix in kept {
            normalized.push(self.normalized.get(ix.get()));
        }
        Pii {
            spans: self.spans.retain_indices(kept),
            kind: kept.iter().map(|&ix| self.kind[ix.get()]).collect(),
            confidence: kept.iter().map(|&ix| self.confidence[ix.get()]).collect(),
            normalized,
        }
    }
}

impl FilterableLayer for LanguageSpans {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        LanguageSpans {
            spans: self.spans.retain_indices(kept),
            language: kept.iter().map(|&ix| self.language[ix.get()]).collect(),
            languages: Arc::clone(&self.languages),
            script: kept.iter().map(|&ix| self.script[ix.get()]).collect(),
            confidence: kept.iter().map(|&ix| self.confidence[ix.get()]).collect(),
        }
    }
}

impl FilterableLayer for Abbreviations {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Abbreviations {
            spans: self.spans.retain_indices(kept),
            form: kept.iter().map(|&ix| self.form[ix.get()]).collect(),
        }
    }
}

impl<L: FilterableLayer> FilterableLayer for Confidence<L> {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Confidence::new(kept.iter().map(|&ix| self.value[ix.get()]).collect())
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<L>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, L>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl<L: FilterableLayer> FilterableLayer for Boilerplate<L> {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Boilerplate::new(
            kept.iter().map(|&ix| self.kind[ix.get()]).collect(),
            kept.iter().map(|&ix| self.score[ix.get()]).collect(),
        )
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.kind == other.kind && self.score == other.score
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<L>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, L>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl FilterableLayer for PosTags {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        PosTags {
            tagset: Arc::clone(&self.tagset),
            tags: kept.iter().map(|&ix| self.tags[ix.get()]).collect(),
            prob: self.prob.as_ref().map(|p| kept.iter().map(|&ix| p[ix.get()]).collect()),
        }
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.tags == other.tags && self.prob == other.prob
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<Tokens>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, Tokens>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl FilterableLayer for ChunkTags {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        ChunkTags {
            tagset: Arc::clone(&self.tagset),
            tags: kept.iter().map(|&ix| self.tags[ix.get()]).collect(),
            prob: self.prob.as_ref().map(|p| kept.iter().map(|&ix| p[ix.get()]).collect()),
        }
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.tags == other.tags && self.prob == other.prob
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<Tokens>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, Tokens>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl FilterableLayer for Lemmas {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        let mut lemma = Strs::default();
        for &ix in kept {
            lemma.push(self.lemma.get(ix.get()));
        }
        Lemmas { lemma }
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.lemma.get(i) == other.lemma.get(i))
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<Tokens>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, Tokens>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl FilterableLayer for Stems {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        let mut stem = Strs::default();
        for &ix in kept {
            stem.push(self.stem.get(ix.get()));
        }
        Stems { stem }
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.stem.get(i) == other.stem.get(i))
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<Tokens>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, Tokens>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl FilterableLayer for Terms {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        let mut surface = Strs::default();
        let mut folded = Strs::default();
        for &ix in kept {
            surface.push(self.surface.get(ix.get()));
            folded.push(self.folded.get(ix.get()));
        }
        let stem = self.stem.as_ref().map(|s| {
            let mut out = Strs::default();
            for &ix in kept {
                out.push(s.get(ix.get()));
            }
            out
        });
        let lemma = self.lemma.as_ref().map(|l| {
            let mut out = Strs::default();
            for &ix in kept {
                out.push(l.get(ix.get()));
            }
            out
        });
        Terms { surface, folded, stem, lemma }
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.len() == other.len()
            && (0..self.len())
                .all(|i| self.surface.get(i) == other.surface.get(i) && self.folded.get(i) == other.folded.get(i))
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        co_written_aligned_anchors::<Tokens>(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        retain_aligned_anchor::<Self, Tokens>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl<L: CategoryScope> FilterableLayer for Categories<L> {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        let mut out = Categories::with_capacity(Arc::clone(&self.categories), kept.len());
        for &ix in kept {
            let i = ix.get();
            out.push_selected(self.category[i], &self.selected[i], self.row_scores(i));
        }
        out
    }

    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.category == other.category && self.selected == other.selected && self.scores == other.scores
    }

    fn co_written_anchors(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        L::categories_co_written_anchors(target_name)
    }

    fn retain_anchor(
        doc: &mut Document,
        source_anchor: KeyId,
        target_name: &'static str,
        kept: &[Ix<Self>],
        provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        L::retain_categories_anchor::<Self>(doc, source_anchor, target_name, kept, provenance)
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{
        Analyzer, ConditionalAnnotator, Document, FilterAnnotator, IxRange, Key, Layer, Provenance, Span, Spans, Tag,
        TagSet, ViewId,
    };

    use super::*;
    use crate::Language;

    #[test]
    fn filter_entities_preserves_discontinuous_fragments_and_token_indexes() {
        let mut doc = Document::new("New York and Los Angeles");
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        tok_spans.push(Span::new(0, 3));
        tok_spans.push(Span::new(4, 8));
        tok_spans.push(Span::new(9, 12));
        tok_spans.push(Span::new(13, 16));
        tok_spans.push(Span::new(17, 24));
        doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

        let kinds = TagSet::new("ner", &["LOC", "ORG"]);
        let mut ent_spans = Spans::new(ViewId::SOURCE);
        // Item 0: discontinuous entity [0..3, 17..24]
        ent_spans.push_fragments(&[Span::new(0, 3), Span::new(17, 24)]);
        // Item 1: continuous entity [4..8]
        ent_spans.push(Span::new(4, 8));
        doc.add(
            &Key::DEFAULT,
            Entities {
                spans: ent_spans,
                tokens: vec![IxRange::new(0, 5), IxRange::new(1, 2)],
                kind: vec![Tag(0), Tag(1)],
                kinds: Arc::clone(&kinds),
                score: vec![0.95, 0.40],
            },
            Provenance::by("test"),
        )
        .unwrap();

        let high_conf: Key<Entities> = Key::named("high_conf");
        let filter = FilterAnnotator::new(Key::DEFAULT, high_conf, |e, ix| e.score[ix.get()] >= 0.8)
            .unwrap()
            .named("score>=0.8");
        let cond = ConditionalAnnotator::new(
            "is_en",
            |d| d.get::<Language>().and_then(|l| l.best()).is_some_and(|(code, _)| code == "en"),
            filter,
        )
        .require(Key::<Language>::DEFAULT);

        let mut lang_code = Strs::default();
        lang_code.push("en");
        doc.add(&Key::DEFAULT, Language { code: lang_code, score: vec![0.99] }, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Entities>::DEFAULT)
            .given(Key::<Language>::DEFAULT)
            .step(cond)
            .build()
            .unwrap();
        let mut docs = vec![doc];
        assert!(analyzer.run(&mut docs)[0].is_ok());

        let filtered = docs[0].get_key(&high_conf).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.spans.is_discontinuous(Ix::new(0)));
        assert_eq!(filtered.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), vec![Span::new(0, 3), Span::new(17, 24)]);
        assert_eq!(docs[0].made_on(high_conf.id()), [Key::<Tokens>::DEFAULT.id()]);
    }
}
