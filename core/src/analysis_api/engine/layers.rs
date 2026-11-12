//! The layers the bundle's heads add to the analyzed document, each with
//! the provenance of the head that wrote it, as the analysis annotators
//! name theirs (`turbo-tagger`).

use std::sync::Arc;

use turbo_analysis::{Document, Execution, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Strs, Tokens, Words};

use super::{AnalyzerEngine, forms, sequences};
use crate::analysis_api::heads::BundleHeads;
use crate::status::{BUNDLE_INVALID, Error, Result};

/// The token layer of `spans`, made per document.
pub(super) fn tokens_of(spans: &[Span]) -> Tokens {
    let mut out = Spans::new(ViewId::SOURCE);
    for s in spans {
        out.push(*s);
    }
    Tokens { spans: out, sentence: None }
}

/// The syntactic words of `tok_spans` over `text` from each of `seqs`'
/// decoded words, which cover every token.
pub(super) fn word_layer(
    text: &str,
    tok_spans: &[Span],
    seqs: &[Vec<usize>],
    split: Vec<Vec<Vec<String>>>,
) -> Result<Words> {
    let mut of: Vec<Vec<String>> = vec![Vec::new(); tok_spans.len()];
    for (s, words) in seqs.iter().zip(split) {
        for (&i, ws) in s.iter().zip(words) {
            of[i] = ws;
        }
    }
    Words::from_splits(text, &tokens_of(tok_spans), &of).map_err(|e| Error::new(BUNDLE_INVALID, format!("words: {e}")))
}

impl AnalyzerEngine {
    /// The provenance of a layer the bundle's heads wrote.
    pub(super) fn head_provenance(heads: &BundleHeads, annotator: &'static str) -> Provenance {
        let mut p = Provenance::by(annotator);
        p.model = Some(ModelRef {
            id: heads.model_id.clone(),
            revision: heads.model_revision.clone(),
            sha256: heads.manifest_sha256.clone(),
        });
        p.execution = Some(Execution {
            backend: heads.backend.clone(),
            device: heads.device.clone(),
            precision: "f32 (model)".into(),
        });
        p
    }

    /// Every layer the bundle's heads write, added to `doc` over its
    /// tokens, read a sentence at a time.
    pub(super) fn add_head_layers(
        &self,
        doc: &mut Document,
        text: &str,
        tok_spans: &[Span],
        sent_spans: &[Span],
    ) -> Result<()> {
        let Some(heads) = &self.heads else { return Ok(()) };
        let added = |e: turbo_analysis::Error| Error::new(BUNDLE_INVALID, format!("add layer: {e}"));
        let tok_strs = forms(text, tok_spans)?;
        let seqs = sequences(tok_spans, sent_spans);
        let words: Vec<Vec<&str>> = seqs.iter().map(|s| s.iter().map(|&i| tok_strs[i]).collect()).collect();
        let written = heads.run(&words, &heads.has())?;
        let n = tok_spans.len();
        if let Some(decoded) = written.pos {
            let (mut tags, mut prob) = (vec![Tag(0); n], vec![0.0; n]);
            for (s, d) in seqs.iter().zip(&decoded) {
                for (k, &i) in s.iter().enumerate() {
                    (tags[i], prob[i]) = (Tag(d.tags[k] as u16), d.prob[k]);
                }
            }
            let tagset: Arc<TagSet> = heads.pos_tagset()?.clone();
            let layer = PosTags { tagset, tags, prob: Some(prob) };
            doc.add(&Key::<PosTags>::DEFAULT, layer, Self::head_provenance(heads, "turbo-tagger-pos"))
                .map_err(added)?;
        }
        if let Some(decoded) = written.morphology {
            let tagset: Arc<TagSet> = heads.morph_tagset()?.clone();
            let mut of: Vec<&str> = vec![""; n];
            for (s, d) in seqs.iter().zip(&decoded) {
                for (k, &i) in s.iter().enumerate() {
                    of[i] = tagset.label(Tag(d.tags[k] as u16));
                }
            }
            let mut feats = Strs::default();
            for f in of {
                feats.push(f);
            }
            doc.add(
                &Key::<MorphFeatures<Tokens>>::DEFAULT,
                MorphFeatures::new(feats),
                Self::head_provenance(heads, "turbo-tagger-morphology"),
            )
            .map_err(added)?;
        }
        if let Some(lemmas) = written.lemmas {
            let mut of: Vec<&str> = vec![""; n];
            for (s, l) in seqs.iter().zip(&lemmas) {
                for (&i, lemma) in s.iter().zip(l) {
                    of[i] = lemma;
                }
            }
            let mut lemma = Strs::default();
            for l in of {
                lemma.push(l);
            }
            doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma }, Self::head_provenance(heads, "turbo-tagger-lemmas"))
                .map_err(added)?;
        }
        if let Some(parses) = written.dependencies {
            let (mut head, mut relation) = (vec![None; n], vec![Tag(0); n]);
            for (s, p) in seqs.iter().zip(&parses) {
                for (k, &i) in s.iter().enumerate() {
                    let h = p.heads[k] as usize;
                    head[i] = (h > 0).then(|| turbo_analysis::Ix::new(s[h - 1] as u32));
                    relation[i] = Tag(p.relations[k] as u16);
                }
            }
            let layer = Dependencies::<Tokens> { head, relation, relations: heads.relations()?.clone() };
            doc.add(
                &Key::<Dependencies<Tokens>>::DEFAULT,
                layer,
                Self::head_provenance(heads, "turbo-tagger-dependencies"),
            )
            .map_err(added)?;
        }
        if let Some(found) = written.entities {
            let mut layer = Entities {
                spans: Spans::new(ViewId::SOURCE),
                tokens: Vec::new(),
                kind: Vec::new(),
                kinds: heads.entity_kinds()?.clone(),
                score: Vec::new(),
            };
            for (s, found) in seqs.iter().zip(&found) {
                for e in found {
                    let (a, b) = (s[e.start as usize], s[e.end as usize - 1]);
                    layer.spans.push(Span::new(tok_spans[a].start, tok_spans[b].end));
                    layer.tokens.push(IxRange::new(a as u32, b as u32 + 1));
                    layer.kind.push(Tag(e.kind as u16));
                    layer.score.push(e.prob);
                }
            }
            let annotator = if heads.entities_by_spans() { "turbo-tagger-spans" } else { "turbo-tagger-entities" };
            doc.add(&Key::<Entities>::DEFAULT, layer, Self::head_provenance(heads, annotator)).map_err(added)?;
        }
        if let Some(split) = written.words {
            let words = word_layer(text, tok_spans, &seqs, split)?;
            doc.add(&Key::<Words>::DEFAULT, words, Self::head_provenance(heads, "turbo-tagger-words"))
                .map_err(added)?;
        }
        Ok(())
    }
}
