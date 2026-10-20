//! The name finder as an analysis step: the gazetteer's names in a view,
//! written as location entities.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Entities, Tokens};

use super::{NameFinder, NameScratch};
use crate::geocode::DEFAULT_LOCATION_TYPE;

/// Writes `opennlp:entities` of type `location` for the gazetteer names a
/// [`NameFinder`] finds in a view, without overlaps (leftmost, then the
/// longest), so [`GeocodeAnnotator`](crate::GeocodeAnnotator) and
/// [`GeoEntityLinker`](crate::GeoEntityLinker) can resolve mentions no
/// name finder model tagged. An entity's tokens are the view's tokens
/// inside its span; scores are 0, as for the dictionary name finder.
///
/// The finder borrows its image for as long as the step lives, so the
/// image is `'static`: a buffer or a mapped file kept for the process
/// (`Box::leak`), as a model is.
pub struct NameAnnotator {
    finder: NameFinder<'static>,
    kinds: Arc<TagSet>,
}

impl NameAnnotator {
    pub fn new(finder: NameFinder<'static>) -> NameAnnotator {
        NameAnnotator { finder, kinds: TagSet::new("turbo-geo:names", &[DEFAULT_LOCATION_TYPE]) }
    }
}

impl Annotator for NameAnnotator {
    const ID: &'static str = "turbo-geo:names";
    type Reads = (View, Tokens);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let mut found = Vec::new();
        self.finder.find(view.text(), &mut NameScratch::new(), &mut found);
        let mut entities = Entities {
            spans: Spans::with_capacity(view.id(), found.len()),
            tokens: Vec::with_capacity(found.len()),
            kind: Vec::with_capacity(found.len()),
            kinds: self.kinds.clone(),
            score: Vec::with_capacity(found.len()),
        };
        for m in found {
            entities.spans.push(m.span);
            entities.tokens.push(tokens.spans.within(m.span));
            entities.kind.push(Tag(0));
            entities.score.push(0.0);
        }
        Ok((entities,))
    }

    fn config_hash(&self) -> u64 {
        self.finder.matcher.fingerprint() ^ self.finder.keys() as u64
    }
}
