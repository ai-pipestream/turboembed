//! `EventExtractor`: predicate-argument events from dependencies, an
//! SRL-lite pass with no model of its own.
//!
//! A predicate is a verb (a `VERB` or `VB*` tag when there are tags,
//! else a word with a subject, object, `obl` or `aux` dependent), the
//! head of a copula (the copula is the trigger and the head its
//! `Attribute`), or a noun whose lemma is one of the configured
//! nominalizations. Each dependent maps to a role by its relation and
//! preposition: `nsubj` is the `Agent`, or the `Patient` in the passive
//! (`nsubj:pass`, `aux:pass`, or a form of *be* over a `VBN`), and a
//! `by` obl of a passive is the `Agent`; `obj` is the `Patient`, `iobj`
//! and `to`/`for` obls the `Recipient`, `in`/`at`/`from` obls the
//! `Location`, `with`/`using` the `Instrument`, `because`/`due to` the
//! `Cause`, and a `to` or `so` adverbial clause the `Purpose`. The
//! subject of a copula is its `Patient`. A verb
//! conjunct with no subject of its own shares its first conjunct's.
//!
//! Anchoring: an argument whose head word is in a mention
//! of `Temporals`, `Money`, `Quantities` or `Entities` carries that
//! layer and row. A temporal argument other than a subject becomes
//! `Time`, money or a quantity becomes `Measure`, and a location entity
//! in an obl becomes `Location`. A temporal, money, quantity or location
//! mention deeper in the predicate's subtree (governed by no nearer
//! predicate) that no argument already anchors is added as an argument
//! of its own.

mod frames;

use turbo_analysis::{Annotator, Document, Error, Key, Refs, Span, View};
use turbo_analysis_layers::{
    Dependencies, Entities, Events, Lemmas, Money, PosTags, Quantities, Sentences, Temporals, Tokens,
};

/// What [`EventExtractor`] treats as a predicate and whether it anchors.
#[derive(Debug, Clone, PartialEq)]
pub struct EventConfig {
    /// Lower-case lemmas of eventive nouns (`acquisition`, `merger`).
    pub nominal_triggers: Vec<String>,
    /// Anchor arguments to `Temporals`, `Money`, `Quantities` and
    /// `Entities`, and add governed mentions as arguments.
    pub anchor: bool,
    /// Share a first conjunct's subject with its verb conjuncts.
    pub propagate_conj: bool,
}

impl Default for EventConfig {
    fn default() -> Self {
        let nominal = [
            "acquisition",
            "merger",
            "purchase",
            "sale",
            "investment",
            "payment",
            "launch",
            "attack",
            "election",
            "bankruptcy",
            "layoff",
            "takeover",
        ];
        EventConfig {
            nominal_triggers: nominal.iter().map(|s| s.to_string()).collect(),
            anchor: true,
            propagate_conj: true,
        }
    }
}

/// Extracts [`Events`] from token dependencies, anchored to the numeric,
/// temporal and entity layers that are present.
#[derive(Debug, Clone, Default)]
pub struct EventExtractor {
    pub config: EventConfig,
}

impl EventExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: EventConfig) -> Self {
        EventExtractor { config }
    }

    /// The events of the default layers of `doc`, without adding them.
    pub fn events(&self, doc: &Document) -> Result<Events, Error> {
        let (events,) = self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<Sentences>::DEFAULT)?,
            doc.require(&Key::<Tokens>::DEFAULT)?,
            doc.require(&Key::<Dependencies>::DEFAULT)?,
            doc.get_key(&Key::<PosTags>::DEFAULT),
            doc.get_key(&Key::<Lemmas>::DEFAULT),
            doc.get_key(&Key::<Temporals>::DEFAULT),
            doc.get_key(&Key::<Money>::DEFAULT),
            doc.get_key(&Key::<Quantities>::DEFAULT),
            doc.get_key(&Key::<Entities>::DEFAULT),
        ))?;
        Ok(events)
    }

    fn fail(detail: String) -> Error {
        Error::Annotator { annotator: <Self as Annotator>::ID, detail }
    }
}

impl Annotator for EventExtractor {
    const ID: &'static str = "turbo:event-extractor";
    const VERSION: &'static str = "1";
    type Reads = (
        View,
        Sentences,
        Tokens,
        Dependencies,
        Option<PosTags>,
        Option<Lemmas>,
        Option<Temporals>,
        Option<Money>,
        Option<Quantities>,
        Option<Entities>,
    );
    type Writes = (Events,);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, pos, lemmas, temporals, money, quantities, entities): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let n = tokens.spans.len();
        if deps.head.len() != n {
            return Err(Self::fail(format!("{} dependencies for {n} tokens", deps.head.len())));
        }
        // The frame walks climb heads and descend children; a cycle would never end.
        deps.check_forest().map_err(Self::fail)?;
        if pos.is_some_and(|p| p.tags.len() != n) || lemmas.is_some_and(|l| l.lemma.len() != n) {
            return Err(Self::fail("tags or lemmas not aligned to the tokens".into()));
        }
        let spans: Vec<Span> = tokens.spans.iter().map(|(_, s)| s).collect();
        let sentence_of = (0..n)
            .map(|t| match &tokens.sentence {
                Some(col) => col[t].get() as u32,
                None => sentences
                    .spans
                    .iter()
                    .position(|(_, s)| s.start <= spans[t].start && spans[t].start < s.end)
                    .unwrap_or(0) as u32,
            })
            .collect();
        let anchors =
            if self.config.anchor { frames::anchors(temporals, money, quantities, entities) } else { Vec::new() };
        let input = frames::Input::new(view.text(), spans, deps, pos, lemmas, sentence_of, anchors);
        let mut events = Events::new(view.id());
        for f in frames::extract(&input, &self.config) {
            events.push(f.as_frame());
        }
        Ok((events,))
    }

    fn config_hash(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        eat(Self::VERSION.as_bytes());
        eat(&[u8::from(self.config.anchor), u8::from(self.config.propagate_conj)]);
        for t in &self.config.nominal_triggers {
            eat(t.as_bytes());
            eat(&[0]);
        }
        h
    }
}
