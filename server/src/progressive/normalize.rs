//! Cross-view normalization step adapters (issue `#1325`) that execute
//! text-reading annotators on a normalized child [`View`] and project their
//! produced [`ProjectableLayer`] spans bi-directionally back to
//! [`ViewId::SOURCE`] so downstream layers (`Chunks`, `Levels`,
//! `Vectors<Sentences>`, `Vectors<Chunks>`, `SparseTerms<View>`) and wire
//! results align with the caller's source text across UTF-8, UTF-16, and
//! Unicode codepoint offset modes.

use std::any::Any;
use std::marker::PhantomData;

use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::{Annotator, Document, Error, Key, KeyId, ProjectableLayer, Provenance, Status, View, ViewId};
use turbo_analysis_embed::{SubwordAnnotator, align};
use turbo_analysis_layers::{Entities, Sentences, Structure, Subwords, Tokens};
use turbo_geo::NameAnnotator;
use turbo_structure::StructureProjector;
use turbo_tokenize::uax29::UntypedWordTokenizer;

fn check_required_inputs(doc: &Document, keys: &[KeyId]) -> Option<String> {
    keys.iter().find_map(|&k| match doc.status(k) {
        Some(Status::Skipped(why)) => Some(format!("{k} was skipped: {why}")),
        Some(Status::Failed(e)) => Some(format!("{k} failed: {e}")),
        _ => None,
    })
}

fn store_single<L: ProjectableLayer, A: Annotator>(
    out: Box<dyn Any + Send>,
    write_key: Key<L>,
    annotator: &A,
    reads: impl Fn(&Document) -> Vec<KeyId>,
    docs: &mut [Document],
    status: &mut [Result<(), Error>],
) {
    let out = *out.downcast::<Vec<Option<Result<L, Error>>>>().expect("step output");
    let key_id = write_key.id();
    let config = annotator.config_hash();
    let model = annotator.model();
    let execution = annotator.execution();
    for (i, r) in out.into_iter().enumerate() {
        let Some(r) = r else { continue };
        let p = Provenance {
            annotator: A::ID,
            version: A::VERSION,
            config,
            model: model.clone(),
            execution: execution.clone(),
            reads: reads(&docs[i]),
            on: Vec::new(),
        };
        let failure = match r {
            Ok(layer) => docs[i].add(&write_key, layer, p.clone()).err(),
            Err(Error::Skipped(why)) => docs[i].skip(key_id, why, p.clone()).err(),
            Err(e) => Some(e),
        };
        if let Some(e) = failure {
            if !docs[i].has(key_id) {
                let _ = docs[i].fail(key_id, e.clone(), p);
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}

/// Runs a `(View,) -> (L,)` annotator on the normalized view `norm_view` and
/// projects the resulting [`ProjectableLayer`] `L` back to [`ViewId::SOURCE`].
pub(super) struct NormalizedSourceStep<A, L> {
    pub(super) annotator: A,
    pub(super) norm_view: &'static str,
    pub(super) _marker: PhantomData<fn() -> L>,
}

impl<A, L> Step for NormalizedSourceStep<A, L>
where
    A: Annotator<Reads = (View,), Writes = (L,)>,
    L: ProjectableLayer,
{
    fn id(&self) -> &'static str {
        A::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![Key::<View>::named(self.norm_view).id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<L>::DEFAULT.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let view_key = Key::<View>::named(self.norm_view);
        let req_keys = [view_key.id()];
        let out: Vec<Option<Result<L, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let view = doc.require(&view_key)?;
                    let (layer,) = self.annotator.annotate((view,))?;
                    doc.project_layer(&layer, ViewId::SOURCE)
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let view_id = Key::<View>::named(self.norm_view).id();
        store_single(out, Key::<L>::DEFAULT, &self.annotator, |_| vec![view_id], docs, status);
    }
}

/// Runs [`UntypedWordTokenizer`] on `norm_view` (with optional projected
/// [`Sentences`]) and projects [`Tokens`] back to [`ViewId::SOURCE`].
pub(super) struct NormalizedTokensStep {
    pub(super) annotator: UntypedWordTokenizer,
    pub(super) norm_view: &'static str,
}

impl Step for NormalizedTokensStep {
    fn id(&self) -> &'static str {
        UntypedWordTokenizer::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![Key::<View>::named(self.norm_view).id(), Key::<Sentences>::DEFAULT.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<Tokens>::DEFAULT.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let view_key = Key::<View>::named(self.norm_view);
        let req_keys = [view_key.id()];
        let out: Vec<Option<Result<Tokens, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let view = doc.require(&view_key)?;
                    let norm_sents = match doc.get::<Sentences>() {
                        Some(s) => Some(doc.project_layer(s, view.id())?),
                        None => None,
                    };
                    let (tokens,) = self.annotator.annotate((view, norm_sents.as_ref()))?;
                    doc.project_layer(&tokens, ViewId::SOURCE)
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let view_id = Key::<View>::named(self.norm_view).id();
        let sent_id = Key::<Sentences>::DEFAULT.id();
        store_single(
            out,
            Key::<Tokens>::DEFAULT,
            &self.annotator,
            |doc| {
                let mut r = vec![view_id];
                if doc.present(sent_id) {
                    r.push(sent_id);
                }
                r
            },
            docs,
            status,
        );
    }
}

/// Runs [`SubwordAnnotator`] on `norm_view`, projects [`Subwords`] back to
/// [`ViewId::SOURCE`], and aligns piece-to-token indices with [`Tokens`] in
/// [`ViewId::SOURCE`] when present.
pub(super) struct NormalizedSubwordsStep {
    pub(super) annotator: SubwordAnnotator,
    pub(super) norm_view: &'static str,
}

impl Step for NormalizedSubwordsStep {
    fn id(&self) -> &'static str {
        SubwordAnnotator::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![Key::<View>::named(self.norm_view).id(), Key::<Tokens>::DEFAULT.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<Subwords>::DEFAULT.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let view_key = Key::<View>::named(self.norm_view);
        let req_keys = [view_key.id()];
        let out: Vec<Option<Result<Subwords, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let view = doc.require(&view_key)?;
                    let mut sub = self.annotator.pieces(view)?;
                    sub = doc.project_layer(&sub, ViewId::SOURCE)?;
                    if let Some(tokens) = doc.get::<Tokens>()
                        && tokens.spans.view() == ViewId::SOURCE
                        && tokens.spans.is_ordered()
                    {
                        sub.token = align(&sub.spans, &tokens.spans);
                    }
                    Ok(sub)
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let view_id = Key::<View>::named(self.norm_view).id();
        let tok_id = Key::<Tokens>::DEFAULT.id();
        store_single(
            out,
            Key::<Subwords>::DEFAULT,
            &self.annotator,
            |doc| {
                let mut r = vec![view_id];
                if doc.present(tok_id) {
                    r.push(tok_id);
                }
                r
            },
            docs,
            status,
        );
    }
}

/// Runs [`NameAnnotator`] on `norm_view` with projected [`Tokens`] and projects
/// the resulting [`Entities`] back to [`ViewId::SOURCE`].
pub(super) struct NormalizedNameStep {
    pub(super) annotator: NameAnnotator,
    pub(super) norm_view: &'static str,
}

impl Step for NormalizedNameStep {
    fn id(&self) -> &'static str {
        NameAnnotator::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![Key::<View>::named(self.norm_view).id(), Key::<Tokens>::DEFAULT.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<Entities>::DEFAULT.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let view_key = Key::<View>::named(self.norm_view);
        let tok_key = Key::<Tokens>::DEFAULT;
        let req_keys = [view_key.id(), tok_key.id()];
        let out: Vec<Option<Result<Entities, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let view = doc.require(&view_key)?;
                    let tokens = doc.require(&tok_key)?;
                    let norm_tokens = doc.project_layer(tokens, view.id())?;
                    let (entities,) = self.annotator.annotate((view, &norm_tokens))?;
                    doc.project_layer(&entities, ViewId::SOURCE)
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let view_id = Key::<View>::named(self.norm_view).id();
        let tok_id = Key::<Tokens>::DEFAULT.id();
        store_single(out, Key::<Entities>::DEFAULT, &self.annotator, |_| vec![view_id, tok_id], docs, status);
    }
}

/// Runs [`StructureProjector`] across the normalized view chain `norm_chain`
/// and stores the validated [`Structure`] back on [`ViewId::SOURCE`] (or
/// `target_name` when `Key::<Structure>::DEFAULT` was pre-populated).
pub(super) struct NormalizedStructureProjectStep {
    pub(super) norm_chain: Vec<&'static str>,
    pub(super) source_name: &'static str,
    pub(super) target_name: &'static str,
}

impl Step for NormalizedStructureProjectStep {
    fn id(&self) -> &'static str {
        StructureProjector::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        let last_view = *self.norm_chain.last().unwrap_or(&"normalized");
        vec![Key::<View>::named(last_view).id(), Key::<Structure>::named(self.source_name).id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<Structure>::named(self.target_name).id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let req_keys = self.reads();
        let src_key = Key::<Structure>::named(self.source_name);
        let out: Vec<Option<Result<Structure, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let src = doc.require(&src_key)?;
                    let mut current: Option<Structure> = None;
                    for &vname in &self.norm_chain {
                        let view = doc.require(&Key::<View>::named(vname))?;
                        let input = current.as_ref().unwrap_or(src);
                        let (projected,) = StructureProjector.annotate((view, input))?;
                        current = Some(projected);
                    }
                    let final_struct = current.expect("non-empty norm_chain");
                    if self.target_name.is_empty() {
                        doc.project_layer(&final_struct, ViewId::SOURCE)
                    } else {
                        Ok(final_struct)
                    }
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let reads = self.reads();
        store_single(
            out,
            Key::<Structure>::named(self.target_name),
            &StructureProjector,
            move |_| reads.clone(),
            docs,
            status,
        );
    }
}
