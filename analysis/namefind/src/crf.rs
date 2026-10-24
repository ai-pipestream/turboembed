//! CRF document integration. The companion alternatives layer is a union
//! of candidate spans with unit-token fallback edges. Exact ranked paths
//! and entity kinds remain available through `CrfNameFinder::best_sequences`;
//! arbitrary recombinations of lattice edges are not ranked CRF paths.
use std::collections::BTreeMap;
use std::sync::Arc;
use turbo_analysis::{Alternatives, Annotator, Error, Ix, IxRange, Layer, ModeSet, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Entities, Sentences, TagSet, Tokens};
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
pub use turbo_sequence::crf::{CrfSequenceModel, FeatureRow};
use turbo_sequence::namefind::NameContextFactory;
pub use turbo_sequence::namefind::{CrfNameFinder, CrfNameSequence};

pub struct CrfNameAnnotator {
    context: NameContextFactory,
    kinds: Arc<TagSet>,
    model: CrfSequenceModel,
    k: usize,
    hash: u64,
    model_ref: ModelRef,
}
impl CrfNameAnnotator {
    pub fn new(file: ModelFile, k: usize) -> Result<Self, String> {
        if k == 0 || k > 4096 {
            return Err("k must be in 1..=4096".into());
        }
        let model = CrfNameFinder::read_model(&file).map_err(|e| e.to_string())?;
        let context = NameContextFactory::for_model(&file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let kinds = crate::entity_kinds::from_labels("turbo:crf-entities", model.labels().iter().map(String::as_str));
        let hash = file
            .bytes()
            .iter()
            .copied()
            .chain((k as u64).to_le_bytes())
            .fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
        // The model's identity, computed once: the provenance of every
        // document it runs on names it.
        let model_ref = ModelRef {
            id: format!("turbo-namefind:crf:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(Self { context, kinds, model, k, hash, model_ref })
    }
}
impl Annotator for CrfNameAnnotator {
    const ID: &'static str = "turbo-namefind:crf";
    const VERSION: &'static str = "2";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Entities, Alternatives<Entities>);
    fn config_hash(&self) -> u64 {
        self.hash
    }
    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }
    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        tokens.check().map_err(fail)?;
        if tokens.spans.view() != view.id() || sentences.spans.view() != view.id() {
            return Err(fail("input view mismatch".into()));
        }
        for (starts, ends) in [sentences.spans.columns(), tokens.spans.columns()] {
            for (&start, &end) in starts.iter().zip(ends) {
                if view.text().get(start as usize..end as usize).is_none() {
                    return Err(fail("input byte boundary".into()));
                }
            }
        }
        for i in 0..sentences.len() {
            if sentences.spans.is_discontinuous(Ix::new(i as u32)) {
                return Err(fail("discontinuous sentence unsupported".into()));
            }
        }
        for i in 0..tokens.len() {
            if tokens.spans.is_discontinuous(Ix::new(i as u32)) {
                return Err(fail("discontinuous token unsupported".into()));
            }
        }
        let kinds = self.kinds.clone();
        let mut entities =
            Entities { spans: Spans::new(view.id()), tokens: Vec::new(), kind: Vec::new(), kinds, score: Vec::new() };
        let mut alternatives = Alternatives::new(view.id());
        let mut context = self.context.create().map_err(|e| fail(e.to_string()))?;
        let mut finder = CrfNameFinder::new(&self.model, &mut context);
        let (starts, ends) = tokens.spans.columns();
        let mut base = 0;
        for sentence in 0..sentences.len() {
            let span = sentences.spans.get(Ix::new(sentence as u32));
            let mut end = base;
            while end < tokens.len() && starts[end] < span.end {
                if starts[end] < span.start
                    || ends[end] > span.end
                    || tokens.sentence.as_ref().is_some_and(|s| s[end].get() != sentence)
                {
                    return Err(fail("token/sentence mismatch".into()));
                }
                end += 1;
            }
            let words: Vec<_> = (base..end).map(|i| &view.text()[starts[i] as usize..ends[i] as usize]).collect();
            let paths = finder.best_sequences(&words, &[], &[], self.k).map_err(|e| fail(e.to_string()))?;
            let mut candidates = BTreeMap::<(usize, usize, String), (f64, bool)>::new();
            let mut covered = vec![false; end - base];
            for (rank, path) in paths.iter().enumerate() {
                for name in &path.names {
                    let (a, b) = (base + name.span.start, base + name.span.end);
                    let kind = name.span.kind.as_deref().unwrap_or("default");
                    let item = candidates.entry((a, b, kind.to_owned())).or_insert((name.prob, false));
                    item.1 |= rank == 0;
                    if rank == 0 {
                        covered[name.span.start..name.span.end].fill(true);
                        entities.spans.push(Span::new(starts[a], ends[b - 1]));
                        entities.tokens.push(IxRange::new(a as u32, b as u32));
                        entities.kind.push(entities.kinds.tag(kind).expect("kind from model"));
                        entities.score.push(name.prob);
                    }
                }
            }
            // A complete unit-token backbone keeps the union lattice valid,
            // including positions traversed only inside a multi-token entity.
            let mut union = BTreeMap::<(usize, usize), (f64, bool)>::new();
            for ((a, b, _), (prob, best)) in candidates {
                let entry = union.entry((a, b)).or_default();
                entry.0 += prob;
                entry.1 |= best;
            }
            for i in base..end {
                alternatives.push_full(
                    Span::new(starts[i], ends[i]),
                    i as u32,
                    1,
                    if !covered[i - base] { ModeSet::BEST } else { ModeSet::EMPTY },
                    Some(0.0),
                    None,
                );
                for (&(a, b), &(score, best)) in union.range((i, 0)..=(i, usize::MAX)) {
                    alternatives.push_full(
                        Span::new(starts[a], ends[b - 1]),
                        a as u32,
                        (b - a) as u32,
                        if best { ModeSet::BEST } else { ModeSet::EMPTY },
                        Some(score.min(1.0)),
                        None,
                    );
                }
            }
            base = end;
        }
        if base != tokens.len() {
            return Err(fail("tokens outside sentences".into()));
        }
        entities.check().map_err(fail)?;
        alternatives.check().map_err(fail)?;
        Ok((entities, alternatives))
    }
}
