//! Which layers a search projection may read (#1932, #2345). When the
//! document holds PII and only a rewrite view's text may leave, a term,
//! an entity name or a value column made from another view carries that
//! view's words, so the projection reads a layer only when its whole
//! lineage may leave ([`disclosure::withheld_layers`]).

use std::any::TypeId;
use std::collections::{HashMap, HashSet};

use turbo_analysis::{Document, Key, KeyId, Layer, View, ViewId};
use turbo_analysis_layers::{
    Aspects, Boilerplate, Categories, Chunks, Containment, Dependencies, DocumentDate, Entities, Lemmas, Modality,
    Money, Pii, Quantities, Sentences, Stems, Temporals, Times, TokenGraph, Tokens,
};
use turbo_entity_linker::LinkedEntities;
use turbo_geo::Locations;
use turbo_group::Sketches;

use crate::results::disclosure::{self, Disclosure};

/// The layers held back from one projection; `None` when all may leave.
pub(crate) struct Gate(Option<HashSet<KeyId>>);

/// Records the view each text-anchored instance of `L` sits on.
fn own<L: Layer>(doc: &Document, out: &mut HashMap<KeyId, ViewId>) {
    for k in doc.layers().filter(|k| k.ty == TypeId::of::<L>()) {
        if let Some(ts) = doc.get_key(&Key::<L>::named(k.name)).and_then(Layer::text_spans) {
            out.insert(k, ts.view);
        }
    }
}

impl Gate {
    pub(crate) fn new(doc: &Document, disclosure: &Disclosure) -> Gate {
        if disclosure.graph(doc) {
            return Gate(None);
        }
        let mut views = HashMap::new();
        own::<Tokens>(doc, &mut views);
        own::<Sentences>(doc, &mut views);
        own::<Chunks>(doc, &mut views);
        own::<TokenGraph>(doc, &mut views);
        own::<Stems>(doc, &mut views);
        own::<Lemmas>(doc, &mut views);
        own::<Entities>(doc, &mut views);
        own::<LinkedEntities>(doc, &mut views);
        own::<Dependencies>(doc, &mut views);
        own::<Modality>(doc, &mut views);
        own::<Money>(doc, &mut views);
        own::<Quantities>(doc, &mut views);
        own::<DocumentDate>(doc, &mut views);
        own::<Temporals>(doc, &mut views);
        own::<Times>(doc, &mut views);
        own::<Locations>(doc, &mut views);
        own::<Containment>(doc, &mut views);
        own::<Aspects>(doc, &mut views);
        own::<Pii>(doc, &mut views);
        own::<Categories<View>>(doc, &mut views);
        own::<Categories<Chunks>>(doc, &mut views);
        own::<Categories<Sentences>>(doc, &mut views);
        own::<Boilerplate<Sentences>>(doc, &mut views);
        own::<Boilerplate<Chunks>>(doc, &mut views);
        own::<Sketches<View>>(doc, &mut views);
        own::<Sketches<Chunks>>(doc, &mut views);
        Gate(Some(disclosure::withheld_layers(doc, disclosure, &views)))
    }

    /// Whether the data of `key` may leave.
    pub(crate) fn keeps(&self, key: KeyId) -> bool {
        self.0.as_ref().is_none_or(|w| !w.contains(&key))
    }
}
