//! The [`LinkedEntities`] output layer (`turbo:linked_entities`), aligned 1-to-1
//! with [`Entities`], together with its [`LayerCodec`] implementation.

use std::fmt;

use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer};
use turbo_analysis::{Anchor, Document, Groups, Ix, Layer};
use turbo_analysis_layers::{Entities, Vectors};

/// Reason why an entity mention resolved to `NIL` instead of a knowledge-base entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NilReason {
    /// No knowledge-base provider returned any candidate for the mention.
    NoCandidates,
    /// Candidates were generated, but all fell below the minimum entity-type compatibility threshold.
    TypeIncompatible,
    /// Compatible candidates were found, but the top combined disambiguation score fell below `nil_threshold`.
    BelowThreshold,
}

impl NilReason {
    pub(crate) const fn to_byte(self) -> u8 {
        match self {
            NilReason::NoCandidates => 1,
            NilReason::TypeIncompatible => 2,
            NilReason::BelowThreshold => 3,
        }
    }

    pub(crate) const fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(NilReason::NoCandidates),
            2 => Some(NilReason::TypeIncompatible),
            3 => Some(NilReason::BelowThreshold),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            NilReason::NoCandidates => "no_candidates",
            NilReason::TypeIncompatible => "type_incompatible",
            NilReason::BelowThreshold => "below_threshold",
        }
    }
}

impl fmt::Display for NilReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One scored candidate knowledge-base link for an entity mention.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateLink {
    /// Canonical entity identifier in the provider's knowledge base.
    pub kb_id: String,
    /// Identifier of the [`crate::KbProvider`] that supplied this candidate.
    pub provider_id: String,
    /// Canonical display name of the KB entity.
    pub canonical_name: String,
    /// Surface alias that matched the mention.
    pub matched_alias: String,
    /// Combined disambiguation score in `[0.0, 1.0]`.
    pub score: f64,
    /// Combined alias and popularity prior component in `[0.0, 1.0]`.
    pub prior_score: f64,
    /// Entity-type compatibility component in `[0.0, 1.0]`.
    pub type_score: f64,
    /// Local and document context overlap (plus optional dense vector) component in `[0.0, 1.0]`.
    pub context_score: f64,
    /// Document-level entity coherence and coreference component in `[0.0, 1.0]`.
    pub coherence_score: f64,
}

/// Borrowed view of one row in [`LinkedEntities`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkedEntityView<'a> {
    pub kb_id: Option<&'a str>,
    pub provider_id: &'a str,
    pub canonical_name: &'a str,
    pub matched_alias: &'a str,
    pub score: f64,
    pub nil_reason: Option<NilReason>,
    pub nil_cluster: Option<u32>,
    pub candidates: &'a [CandidateLink],
}

impl LinkedEntityView<'_> {
    /// Returns `true` if the mention was linked to a KB entity (`kb_id.is_some()`).
    pub fn is_linked(&self) -> bool {
        self.kb_id.is_some()
    }

    /// Returns `true` if the mention was resolved to `NIL` (`kb_id.is_none()`).
    pub fn is_nil(&self) -> bool {
        self.kb_id.is_none()
    }
}

/// Output layer (`turbo:linked_entities`) aligned 1-to-1 with [`Entities`].
///
/// For each entity `i` in [`Entities`], records either the resolved KB identifier,
/// provider, canonical name, matched alias, and confidence score, or an explicit
/// `NIL` assignment with a [`NilReason`] and surface/coreference `nil_cluster` ID,
/// along with the top-$k$ ranked [`CandidateLink`]s.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinkedEntities {
    pub kb_id: Vec<Option<String>>,
    pub provider_id: Vec<String>,
    pub canonical_name: Vec<String>,
    pub matched_alias: Vec<String>,
    pub score: Vec<f64>,
    pub nil_reason: Vec<Option<NilReason>>,
    pub nil_cluster: Vec<Option<u32>>,
    pub candidates: Vec<Vec<CandidateLink>>,
}

impl LinkedEntities {
    /// Creates an empty [`LinkedEntities`] layer with capacity for `cap` entities.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            kb_id: Vec::with_capacity(cap),
            provider_id: Vec::with_capacity(cap),
            canonical_name: Vec::with_capacity(cap),
            matched_alias: Vec::with_capacity(cap),
            score: Vec::with_capacity(cap),
            nil_reason: Vec::with_capacity(cap),
            nil_cluster: Vec::with_capacity(cap),
            candidates: Vec::with_capacity(cap),
        }
    }

    /// Returns the view for entity `i`.
    pub fn get(&self, i: Ix<Entities>) -> LinkedEntityView<'_> {
        let idx = i.get();
        LinkedEntityView {
            kb_id: self.kb_id[idx].as_deref(),
            provider_id: &self.provider_id[idx],
            canonical_name: &self.canonical_name[idx],
            matched_alias: &self.matched_alias[idx],
            score: self.score[idx],
            nil_reason: self.nil_reason[idx],
            nil_cluster: self.nil_cluster[idx],
            candidates: &self.candidates[idx],
        }
    }

    /// Iterates over all linked entity rows in entity index order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Entities>, LinkedEntityView<'_>)> {
        (0..self.kb_id.len()).map(|i| {
            let ix = Ix::new(i as u32);
            (ix, self.get(ix))
        })
    }
}

impl Layer for LinkedEntities {
    const ID: &'static str = "turbo:linked_entities";

    fn anchor() -> Anchor {
        Anchor::aligned::<Entities>()
    }

    fn len(&self) -> usize {
        self.kb_id.len()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.kb_id.len();
        if self.provider_id.len() != n
            || self.canonical_name.len() != n
            || self.matched_alias.len() != n
            || self.score.len() != n
            || self.nil_reason.len() != n
            || self.nil_cluster.len() != n
            || self.candidates.len() != n
        {
            return Err("parallel columns in LinkedEntities must all match entity count".into());
        }
        for i in 0..n {
            let s = self.score[i];
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(format!("entity {i} has score {s} outside [0.0, 1.0]"));
            }
            match (&self.kb_id[i], self.nil_reason[i], self.nil_cluster[i]) {
                (Some(id), None, None) => {
                    if id.trim().is_empty() {
                        return Err(format!("linked entity {i} has a blank kb_id"));
                    }
                    if self.provider_id[i].trim().is_empty() {
                        return Err(format!("linked entity {i} has a blank provider_id"));
                    }
                }
                (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                    return Err(format!("linked entity {i} must not carry a nil_reason or nil_cluster"));
                }
                (None, Some(_), Some(_)) => {}
                (None, None, _) => {
                    return Err(format!("NIL entity {i} must carry a NilReason"));
                }
                (None, _, None) => {
                    return Err(format!("NIL entity {i} must carry a nil_cluster id"));
                }
            }
            let cands = &self.candidates[i];
            for (j, c) in cands.iter().enumerate() {
                if c.kb_id.trim().is_empty() || c.provider_id.trim().is_empty() {
                    return Err(format!("entity {i} candidate {j} has blank kb_id or provider_id"));
                }
                for (name, val) in [
                    ("score", c.score),
                    ("prior_score", c.prior_score),
                    ("type_score", c.type_score),
                    ("context_score", c.context_score),
                    ("coherence_score", c.coherence_score),
                ] {
                    if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                        return Err(format!("entity {i} candidate {j} has {name} {val} outside [0.0, 1.0]"));
                    }
                }
                if j > 0 && cands[j - 1].score < c.score {
                    return Err(format!("entity {i} candidates are not sorted in descending score order at index {j}"));
                }
            }
        }
        Ok(())
    }
}

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn invalid(cx: &ImportCx, detail: String) -> ExportError {
    ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail }
}

impl LayerCodec for LinkedEntities {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        let n = self.kb_id.len();
        w.usize(n)?;
        for i in 0..n {
            match &self.kb_id[i] {
                None => w.u8(0),
                Some(id) => {
                    w.u8(1);
                    w.str(id)?;
                }
            }
            w.str(&self.provider_id[i])?;
            w.str(&self.canonical_name[i])?;
            w.str(&self.matched_alias[i])?;
            w.f64(self.score[i]);
            match self.nil_reason[i] {
                None => w.u8(0),
                Some(r) => w.u8(r.to_byte()),
            }
            match self.nil_cluster[i] {
                None => w.u8(0),
                Some(c) => {
                    w.u8(1);
                    w.u32(c);
                }
            }
            w.usize(self.candidates[i].len())?;
            for c in &self.candidates[i] {
                w.str(&c.kb_id)?;
                w.str(&c.provider_id)?;
                w.str(&c.canonical_name)?;
                w.str(&c.matched_alias)?;
                w.f64(c.score);
                w.f64(c.prior_score);
                w.f64(c.type_score);
                w.f64(c.context_score);
                w.f64(c.coherence_score);
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let n = r.usize()?;
        let mut layer = LinkedEntities::with_capacity(n.min(1024));
        for _ in 0..n {
            let kb_id = match r.u8()? {
                0 => None,
                1 => Some(r.string()?),
                tag => return Err(malformed(format!("invalid optional kb_id tag {tag}"))),
            };
            let provider_id = r.string()?;
            let canonical_name = r.string()?;
            let matched_alias = r.string()?;
            let score = r.f64()?;
            let nil_reason = match r.u8()? {
                0 => None,
                b => Some(NilReason::from_byte(b).ok_or_else(|| malformed(format!("invalid NilReason byte {b}")))?),
            };
            let nil_cluster = match r.u8()? {
                0 => None,
                1 => Some(r.u32()?),
                tag => return Err(malformed(format!("invalid optional nil_cluster tag {tag}"))),
            };
            let cand_count = r.usize()?;
            let mut cands = Vec::with_capacity(cand_count.min(64));
            for _ in 0..cand_count {
                cands.push(CandidateLink {
                    kb_id: r.string()?,
                    provider_id: r.string()?,
                    canonical_name: r.string()?,
                    matched_alias: r.string()?,
                    score: r.f64()?,
                    prior_score: r.f64()?,
                    type_score: r.f64()?,
                    context_score: r.f64()?,
                    coherence_score: r.f64()?,
                });
            }
            layer.kb_id.push(kb_id);
            layer.provider_id.push(provider_id);
            layer.canonical_name.push(canonical_name);
            layer.matched_alias.push(matched_alias);
            layer.score.push(score);
            layer.nil_reason.push(nil_reason);
            layer.nil_cluster.push(nil_cluster);
            layer.candidates.push(cands);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in LinkedEntities payload"));
        }
        layer.check().map_err(|detail| invalid(cx, detail))?;
        // The layer is aligned to `Entities`: one row per entity of the
        // instance this record's provenance resolves `Entities` to.
        let entities = LayerRef {
            layer: Entities::NAME.to_string(),
            instance: cx.resolved_instance(&Entities::NAME.to_string()).to_owned(),
        };
        let want = cx.layer_len(&entities)?;
        if layer.len() != want {
            return Err(invalid(cx, format!("{} linked rows for an aligned Entities layer of {want}", layer.len())));
        }
        Ok(layer)
    }
}

/// Registers [`LinkedEntities`], `Vectors<Entities>`, `Groups<Entities>`, and
/// coreference codecs on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    turbo_coref::register_codecs(codecs);
    codecs.register::<LinkedEntities>().register::<Vectors<Entities>>().register::<Groups<Entities>>();
}

/// Returns the shared layer codecs with [`register_codecs`] applied.
pub fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}
