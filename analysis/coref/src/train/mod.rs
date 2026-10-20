//! Training (`CorefTrainer`): a pair classifier from the events of
//! documents with gold chains, or a mention ranker ([`train_ranking`]).
//!
//! Training uses the annotator's mention detection, candidate order and
//! speaker and string sieves, so a model learns the decisions left after
//! them. A corpus that annotates singleton entities annotates every
//! mention of its scheme; there, a detected mention outside the gold is
//! not an anaphor to learn from.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use turbo_analysis::{Groups, Span};
use turbo_train::trainer::train_events;
use turbo_train::{Event, JavaString, TrainingConfiguration, TrainingParameters};

use crate::annotator::CorefAnnotator;
use crate::input::CorefInput;
use crate::layer::Mentions;
use crate::model::{APART, CorefModel, LINK};
use crate::sieve::Resolver;

mod ranking;

pub use ranking::{DEFAULT_EPOCHS, DEFAULT_L2, DEFAULT_LEARNING_RATE, train_ranking};

/// One gold mention: its envelope in the input's view and its chain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GoldMention {
    pub span: Span,
    pub chain: usize,
}

/// A training document: the layers coreference reads, and the gold.
#[derive(Clone, Debug)]
pub struct TrainingDocument<'a> {
    pub input: CorefInput<'a>,
    /// Gold mentions with chain ids numbered from zero in first-mention
    /// order (start ascending, wider first), each span once.
    pub gold: Vec<GoldMention>,
}

/// OpenNLP's `Span.compareTo`: start ascending, the wider span first.
pub(crate) fn span_order(a: &Span, b: &Span) -> std::cmp::Ordering {
    a.start.cmp(&b.start).then(b.end.cmp(&a.end))
}

/// The gold mentions of a gold `Mentions` layer and its chains, chain
/// ids renumbered in first-mention order; a mention in no chain is left
/// out, and a discontinuous mention is matched by its envelope.
pub fn gold_from_layers(mentions: &Mentions, chains: &Groups<Mentions>) -> Vec<GoldMention> {
    let mut entries: Vec<(Span, usize)> = Vec::new();
    for (g, members) in chains.iter() {
        for &m in members {
            entries.push((mentions.spans.get(m), g.get()));
        }
    }
    entries.sort_by(|a, b| span_order(&a.0, &b.0));
    let mut renumber: HashMap<usize, usize> = HashMap::new();
    entries
        .into_iter()
        .map(|(span, g)| {
            let next = renumber.len();
            GoldMention { span, chain: *renumber.entry(g).or_insert(next) }
        })
        .collect()
}

/// The gold chain of each detected mention, `None` outside the gold;
/// `Err` for a repeated gold span or chain ids out of first-mention order.
fn gold_chains(gold: &[GoldMention], resolver: &Resolver<'_>) -> Result<Vec<Option<usize>>, String> {
    let mut sorted = gold.to_vec();
    sorted.sort_by(|a, b| span_order(&a.span, &b.span));
    let mut chain_of_span: HashMap<Span, usize> = HashMap::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for g in &sorted {
        if chain_of_span.insert(g.span, g.chain).is_some() {
            return Err(format!("duplicate gold mention span: [{}..{})", g.span.start, g.span.end));
        }
        if seen.insert(g.chain) && g.chain != seen.len() - 1 {
            return Err(format!(
                "gold chain identifiers must start at zero in first-mention order: expected {}, found {} at [{}..{})",
                seen.len() - 1,
                g.chain,
                g.span.start,
                g.span.end
            ));
        }
    }
    Ok(resolver.mentions.iter().map(|m| chain_of_span.get(&m.span).copied()).collect())
}

/// What the gold says about a document beyond each mention's chain.
pub(crate) struct Gold {
    pub chains: Vec<Option<usize>>,
    /// The first offset of every gold chain.
    pub starts: Vec<u32>,
    /// Whether any gold chain has one mention.
    pub singletons: bool,
}

impl Gold {
    pub fn of(gold: &[GoldMention], resolver: &Resolver<'_>) -> Result<Gold, String> {
        let chains = gold_chains(gold, resolver)?;
        let n = gold.iter().map(|g| g.chain + 1).max().unwrap_or(0);
        let mut starts = vec![u32::MAX; n];
        let mut sizes = vec![0usize; n];
        for g in gold {
            starts[g.chain] = starts[g.chain].min(g.span.start);
            sizes[g.chain] += 1;
        }
        Ok(Gold { chains, starts, singletons: sizes.contains(&1) })
    }

    /// Whether anaphor `j`'s gold antecedents all lie outside the detected
    /// candidates: its chain starts before it.
    pub fn unlearnable(&self, resolver: &Resolver<'_>, j: usize) -> bool {
        self.chains[j].is_some_and(|c| self.starts[c] < resolver.mentions[j].span.start)
    }

    /// Whether anaphor `j` is skipped: not heading its cluster, not
    /// rankable, or outside a gold that annotates every mention.
    pub fn skips(&self, resolver: &mut Resolver<'_>, j: usize) -> bool {
        resolver.clusters.find(j) != j || !resolver.rankable(j) || (self.singletons && self.chains[j].is_none())
    }
}

/// The training pairs of one document, in text order of their anaphors.
pub fn pairs(doc: &TrainingDocument<'_>, annotator: &CorefAnnotator) -> Result<Vec<Event>, String> {
    let Some(prepared) = doc.input.prepare()? else { return Ok(Vec::new()) };
    let mut resolver = annotator.resolver(&prepared)?;
    resolver.resolve_precise();
    let gold = Gold::of(&doc.gold, &resolver)?;
    let mut events = Vec::new();
    for j in 0..resolver.mentions.len() {
        if gold.skips(&mut resolver, j) {
            continue;
        }
        let candidates = resolver.ranker_candidates(j);
        let antecedent =
            gold.chains[j].and_then(|c| candidates.iter().copied().filter(|&i| gold.chains[i] == Some(c)).max());
        if antecedent.is_none() && gold.unlearnable(&resolver, j) {
            continue;
        }
        for i in candidates {
            if antecedent.is_some_and(|a| i < a) {
                continue;
            }
            let (names, values) = resolver.features(i, j)?.expand()?;
            let outcome = if Some(i) == antecedent { LINK } else { APART };
            let context = names.iter().map(|n| JavaString::new(n)).collect();
            events.push(Event::with_values(outcome, context, values).map_err(|e| e.to_string())?);
        }
        if let Some(a) = antecedent {
            resolver.clusters.union(a, j);
        }
    }
    Ok(events)
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// Trains a pair classifier with the given parameters (`Algorithm`,
/// `Iterations`, `Cutoff`, ...; real-valued features need
/// `DataIndexer=OnePassRealValue`).
pub fn train(
    language: &str,
    documents: &[TrainingDocument<'_>],
    params: &TrainingParameters,
    annotator: &CorefAnnotator,
) -> Result<CorefModel, String> {
    if language.chars().all(turbo_unicode::java::is_whitespace) {
        return Err("languageCode must not be null or blank".into());
    }
    let mut events = Vec::new();
    for doc in documents {
        events.extend(pairs(doc, annotator)?);
    }
    let (model, report) =
        train_events(&events, params, &mut TrainingConfiguration::default()).map_err(|e| e.to_string())?;
    let entry = model.write().map_err(|e| e.to_string())?;
    CorefModel::package(language, entry, false, annotator.token_vector_dimension(), Some(&report), now_ms())
}
