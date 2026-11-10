use std::sync::Arc;

use turbo_analysis::export::Codecs;
use turbo_analysis::{
    Anchor, Annotator, Document, Error, Execution, Fnv1a, Ix, IxRange, Key, Layer, ModelRef, Refs, Span, Spans,
    Support, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Entities, Pii, PiiKind, Tokens};
use turbo_entity_linker::{KbProvider, prior_score, type_compatibility};

use crate::Decisions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionRejection {
    NonEntity,
    PiiConflict(PiiKind),
    NoCompatibleLink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionDecision {
    Accepted,
    Changed,
    Rejected(PrecisionRejection),
}

impl PrecisionDecision {
    pub fn is_rejected(self) -> bool {
        matches!(self, PrecisionDecision::Rejected(_))
    }
}

#[derive(Clone, Debug)]
pub struct PrecisionCandidates {
    pub spans: Spans<PrecisionCandidates>,
    pub original_spans: Vec<Span>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub original_tokens: Vec<IxRange<Tokens>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    pub logits: Vec<f32>,
    pub score: Vec<f64>,
    pub decision: Vec<PrecisionDecision>,
}

#[derive(Clone, Copy, Debug)]
pub struct PrecisionCandidate {
    pub original_span: Span,
    pub span: Span,
    pub original_tokens: IxRange<Tokens>,
    pub tokens: IxRange<Tokens>,
    pub kind: Tag,
    pub logit: f32,
    pub score: f64,
    pub decision: PrecisionDecision,
}

impl PrecisionCandidates {
    pub fn new(view: ViewId, kinds: Arc<TagSet>) -> Self {
        Self {
            spans: Spans::new(view),
            original_spans: Vec::new(),
            tokens: Vec::new(),
            original_tokens: Vec::new(),
            kind: Vec::new(),
            kinds,
            logits: Vec::new(),
            score: Vec::new(),
            decision: Vec::new(),
        }
    }

    pub fn get(&self, i: Ix<PrecisionCandidates>) -> PrecisionCandidate {
        let at = i.get();
        PrecisionCandidate {
            original_span: self.original_spans[at],
            span: self.spans.get(i),
            original_tokens: self.original_tokens[at],
            tokens: self.tokens[at],
            kind: self.kind[at],
            logit: self.logits[at],
            score: self.score[at],
            decision: self.decision[at],
        }
    }

    pub fn accepted(&self) -> impl Iterator<Item = PrecisionCandidate> {
        (0..self.spans.len()).map(|i| self.get(Ix::new(i as u32))).filter(|c| !c.decision.is_rejected())
    }

    pub fn accepted_entities(&self) -> Entities {
        let mut out = Entities {
            spans: Spans::new(self.spans.view()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: Arc::clone(&self.kinds),
            score: Vec::new(),
        };
        for c in self.accepted() {
            out.spans.push(c.span);
            out.tokens.push(c.tokens);
            out.kind.push(c.kind);
            out.score.push(c.score);
        }
        out
    }
}

impl Layer for PrecisionCandidates {
    const ID: &'static str = "turbo:gliner-precision";
    const OVERLAP: turbo_analysis::Overlap = turbo_analysis::Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.decision.len()
    }

    fn indexes(&self) -> Vec<(turbo_analysis::KeyId, usize)> {
        let bound = self.original_tokens.iter().chain(&self.tokens).map(|r| r.end.get()).max().unwrap_or(0);
        vec![(Key::<Tokens>::DEFAULT.id(), bound)]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.original_spans.len() != n
            || self.tokens.len() != n
            || self.original_tokens.len() != n
            || self.kind.len() != n
            || self.logits.len() != n
            || self.score.len() != n
            || self.decision.len() != n
        {
            return Err("every precision column must have one row per candidate".into());
        }
        self.spans.check(Self::OVERLAP)?;
        for (i, c) in (0..n).map(|i| (i, self.get(Ix::new(i as u32)))) {
            if c.original_span.is_empty() || c.span.is_empty() {
                return Err(format!("precision candidate {i} is empty"));
            }
            if c.original_tokens.start > c.original_tokens.end || c.tokens.start > c.tokens.end {
                return Err(format!("precision candidate {i} has reversed token bounds"));
            }
            if c.tokens.start < c.original_tokens.start || c.tokens.end > c.original_tokens.end {
                return Err(format!("precision candidate {i} expands beyond its original tokens"));
            }
            if c.kind.0 as usize >= self.kinds.len() {
                return Err(format!("precision candidate {i} has an unknown kind"));
            }
            if !c.logit.is_finite() || !c.score.is_finite() {
                return Err(format!("precision candidate {i} has a nonfinite score"));
            }
            let changed = c.original_span != c.span || c.original_tokens != c.tokens;
            match (c.decision, changed) {
                (PrecisionDecision::Accepted, true) => {
                    return Err(format!("precision candidate {i} is accepted but changed"));
                }
                (PrecisionDecision::Changed, false) => {
                    return Err(format!("precision candidate {i} is marked changed without an edit"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl turbo_analysis::Evidenced for PrecisionCandidates {
    fn evidence_spans(
        &self,
        _doc: &Document,
        _provenance: &turbo_analysis::Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error> {
        for (_, span) in self.spans.iter() {
            spans.push(span);
            row.push(spans.len() as u32);
        }
        Ok(self.spans.view())
    }

    fn raw_scores(&self, _provenance: &turbo_analysis::Provenance, out: &mut Vec<(u32, f64)>) {
        out.extend(self.logits.iter().enumerate().map(|(i, &logit)| (i as u32, f64::from(logit))));
    }

    fn evidence_support(
        &self,
        provenance: &turbo_analysis::Provenance,
        layers: &mut Vec<turbo_analysis::KeyId>,
        row: &mut Vec<u32>,
        support: &mut Vec<Support>,
    ) {
        let mut slot = |key| {
            layers.iter().position(|k| *k == key).unwrap_or_else(|| {
                layers.push(key);
                layers.len() - 1
            }) as u32
        };
        let entities =
            slot(turbo_analysis::evidence::read_of::<Entities>(provenance).unwrap_or(Key::<Entities>::DEFAULT.id()));
        let decisions =
            slot(turbo_analysis::evidence::read_of::<Decisions>(provenance).unwrap_or(Key::<Decisions>::DEFAULT.id()));
        let tokens =
            slot(turbo_analysis::evidence::read_of::<Tokens>(provenance).unwrap_or(Key::<Tokens>::DEFAULT.id()));
        for (i, range) in self.tokens.iter().enumerate() {
            let at = i as u32;
            support.push(Support { layer: entities, start: at, end: at + 1 });
            support.push(Support { layer: decisions, start: at, end: at + 1 });
            support.push(Support { layer: tokens, start: range.start.get() as u32, end: range.end.get() as u32 });
            row.push(support.len() as u32);
        }
    }
}

#[derive(Clone, Debug)]
struct PiiNegativeRule {
    label: String,
    kind: PiiKind,
}

#[derive(Clone)]
pub struct HybridPrecision {
    providers: Vec<Arc<dyn KbProvider>>,
    link_threshold: f64,
    linked_labels: Option<Vec<String>>,
    trim_articles: bool,
    reject_non_alpha: bool,
    negative: Vec<PiiNegativeRule>,
}

impl HybridPrecision {
    pub fn new(link_threshold: f64) -> Result<Self, String> {
        if !link_threshold.is_finite() || !(0.0..=1.0).contains(&link_threshold) {
            return Err("link threshold must be finite in [0,1]".into());
        }
        Ok(Self {
            providers: Vec::new(),
            link_threshold,
            linked_labels: None,
            trim_articles: true,
            reject_non_alpha: true,
            negative: Vec::new(),
        })
    }

    pub fn with_provider(mut self, provider: Arc<dyn KbProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Requires a compatible link only for `label`. The first call switches
    /// from all-label linking to an explicit label set, preserving other types.
    pub fn require_link(mut self, label: impl Into<String>) -> Self {
        self.linked_labels.get_or_insert_with(Vec::new).push(label.into());
        self
    }

    fn links(&self, label: &str) -> bool {
        !self.providers.is_empty()
            && self.linked_labels.as_ref().is_none_or(|labels| labels.iter().any(|linked| linked == label))
    }

    pub fn with_negative_rule(mut self, label: impl Into<String>, kind: PiiKind) -> Self {
        self.negative.push(PiiNegativeRule { label: label.into(), kind });
        self
    }

    pub fn trim_articles(mut self, enabled: bool) -> Self {
        self.trim_articles = enabled;
        self
    }

    pub fn reject_non_alpha(mut self, enabled: bool) -> Self {
        self.reject_non_alpha = enabled;
        self
    }

    pub fn filter(
        &self,
        view: &View,
        tokens: &Tokens,
        entities: &Entities,
        decisions: &Decisions,
        pii: Option<&Pii>,
    ) -> Result<PrecisionCandidates, Error> {
        validate_inputs(view, tokens, entities, decisions)?;
        let mut out = PrecisionCandidates::new(view.id(), Arc::clone(&entities.kinds));
        for (ix, original_span) in entities.spans.iter() {
            let original_tokens = entities.tokens[ix.get()];
            let (span, token_range) = trim(view, tokens, original_tokens, self.trim_articles);
            let kind = entities.kind[ix.get()];
            let label = entities.kinds.label(kind);
            let text = &view.text()[span.range()];
            let changed = span != original_span || token_range != original_tokens;
            let mut rejection = (self.reject_non_alpha && !text.chars().any(char::is_alphanumeric))
                .then_some(PrecisionRejection::NonEntity);
            if rejection.is_none()
                && let Some(layer) = pii
            {
                for rule in &self.negative {
                    if rule.label != label {
                        continue;
                    }
                    if layer
                        .mentions()
                        .any(|m| m.kind == rule.kind && m.span.start < span.end && span.start < m.span.end)
                    {
                        rejection = Some(PrecisionRejection::PiiConflict(rule.kind));
                        break;
                    }
                }
            }
            if rejection.is_none() && self.links(label) {
                let mut best = 0.0_f64;
                for provider in &self.providers {
                    for candidate in provider.generate_candidates(text, None, 4, 1.0) {
                        let link =
                            0.5 * prior_score(&candidate) + 0.5 * type_compatibility(label, &candidate.entity_types);
                        best = best.max(link);
                    }
                }
                if best < self.link_threshold {
                    rejection = Some(PrecisionRejection::NoCompatibleLink);
                }
            }
            let decision = match rejection {
                Some(reason) => PrecisionDecision::Rejected(reason),
                None if changed => PrecisionDecision::Changed,
                None => PrecisionDecision::Accepted,
            };
            out.original_spans.push(original_span);
            out.spans.push(span);
            out.original_tokens.push(original_tokens);
            out.tokens.push(token_range);
            out.kind.push(kind);
            out.logits.push(decisions.logits[ix.get()]);
            out.score.push(entities.score[ix.get()]);
            out.decision.push(decision);
        }
        out.check().map_err(|detail| Error::Invalid { layer: Key::<PrecisionCandidates>::DEFAULT.id(), detail })?;
        Ok(out)
    }
}

fn validate_inputs(view: &View, tokens: &Tokens, entities: &Entities, decisions: &Decisions) -> Result<(), Error> {
    let fail = |detail: String| Error::Invalid { layer: Key::<PrecisionCandidates>::DEFAULT.id(), detail };
    tokens.check().map_err(fail)?;
    entities.check().map_err(fail)?;
    decisions.check().map_err(fail)?;
    if entities.spans.view() != view.id() || tokens.spans.view() != view.id() {
        return Err(fail("precision inputs must share one view".into()));
    }
    if entities.spans.len() != decisions.logits.len() {
        return Err(fail("precision decisions must align to entities".into()));
    }
    for (_, span) in tokens.spans.iter() {
        if view.text().get(span.range()).is_none() {
            return Err(fail(format!("token span {}..{} is outside its view", span.start, span.end)));
        }
    }
    let (starts, ends) = tokens.spans.columns();
    for (i, range) in entities.tokens.iter().enumerate() {
        let (start, end) = (range.start.get(), range.end.get());
        if start >= end || end > tokens.spans.len() {
            return Err(fail(format!("entity {i} has an empty or out-of-bounds token range")));
        }
        let expected = Span::new(starts[start], ends[end - 1]);
        let span = entities.spans.get(turbo_analysis::Ix::new(i as u32));
        if span != expected || view.text().get(span.range()).is_none() {
            return Err(fail(format!("entity {i} does not match its token span")));
        }
    }
    Ok(())
}

fn trim(view: &View, tokens: &Tokens, range: IxRange<Tokens>, articles: bool) -> (Span, IxRange<Tokens>) {
    let (starts, ends) = tokens.spans.columns();
    let (mut start, mut end) = (range.start.get(), range.end.get());
    while end - start > 1
        && !view.text()[starts[start] as usize..ends[start] as usize].chars().any(char::is_alphanumeric)
    {
        start += 1;
    }
    while end - start > 1
        && !view.text()[starts[end - 1] as usize..ends[end - 1] as usize].chars().any(char::is_alphanumeric)
    {
        end -= 1;
    }
    if articles && end - start > 1 {
        let first = &view.text()[starts[start] as usize..ends[start] as usize];
        if matches!(first.to_ascii_lowercase().as_str(), "a" | "an" | "the") {
            start += 1;
        }
    }
    let range = IxRange::new(start as u32, end as u32);
    (Span::new(starts[start], ends[end - 1]), range)
}

impl Annotator for HybridPrecision {
    const ID: &'static str = "turbo:gliner-precision";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens, Entities, Decisions, Option<Pii>);
    type Writes = (PrecisionCandidates,);

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.link_threshold.to_bits());
        h.u8(self.trim_articles as u8).u8(self.reject_non_alpha as u8).usize(self.providers.len());
        for provider in &self.providers {
            h.u64(provider.fingerprint());
        }
        match &self.linked_labels {
            None => {
                h.u8(0);
            }
            Some(labels) => {
                h.u8(1).usize(labels.len());
                for label in labels {
                    h.str(label);
                }
            }
        };
        for rule in &self.negative {
            h.str(&rule.label).u32(rule.kind.code());
        }
        h.finish()
    }

    fn model(&self) -> Option<ModelRef> {
        None
    }

    fn execution(&self) -> Option<Execution> {
        None
    }

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (view, tokens, entities, decisions, pii) = input;
        Ok((self.filter(view, tokens, entities, decisions, pii)?,))
    }
}

pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<PrecisionCandidates>();
    codecs.register::<turbo_analysis::Evidence<PrecisionCandidates>>();
}
