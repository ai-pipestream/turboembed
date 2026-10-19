use crate::{
    ArcStandardOracle, ArcStandardState, DependencyContextGenerator, DependencyGraph, ParserInput, Result, error,
};
use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::{Dependencies, Tokens, WordOrTokenAnchor};
use turbo_formats::conllu::ConlluDependencySample;
use turbo_formats::sample::dependency::{DependencySample as FormatSample, DependencyTree, Head};
use turbo_formats::source::Field;
use turbo_train::Event;
use turbo_unicode::WhitespaceMode;

/// Sentence data and its tree. Optional CoNLL-U metadata survives a graph/layer
/// round trip through this carrier; a bare dependency layer has no text offsets.
pub struct DependencySample<A: WordOrTokenAnchor = Tokens> {
    tokens: Vec<String>,
    tags: Vec<String>,
    graph: DependencyGraph<A>,
    conllu: Option<ConlluDependencySample>,
}
impl<A: WordOrTokenAnchor> Clone for DependencySample<A> {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            tags: self.tags.clone(),
            graph: self.graph.clone(),
            conllu: self.conllu.clone(),
        }
    }
}
impl<A: WordOrTokenAnchor> DependencySample<A> {
    pub fn new(tokens: Vec<String>, tags: Vec<String>, graph: DependencyGraph<A>) -> Result<Self> {
        ParserInput::check(
            &tokens.iter().map(String::as_str).collect::<Vec<_>>(),
            &tags.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        graph.check_tree()?;
        if tokens.len() != graph.len() {
            return Err(error("one graph node per token required"));
        }
        Ok(Self { tokens, tags, graph, conllu: None })
    }
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
    pub fn tags(&self) -> &[String] {
        &self.tags
    }
    pub fn graph(&self) -> &DependencyGraph<A> {
        &self.graph
    }
    pub fn from_conllu(sample: ConlluDependencySample) -> Result<Self> {
        sample.check().map_err(error)?;
        let heads = sample
            .sample
            .tree()
            .heads()
            .iter()
            .map(|h| match h {
                Head::Root => None,
                Head::Word(i) => Some(Ix::new(*i)),
            })
            .collect();
        let relations = sample.sample.tree().relations().iter().map(|r| r.as_str().to_owned()).collect();
        let graph = DependencyGraph::new(heads, relations)?;
        let mut out = Self::new(
            sample.sample.tokens().iter().map(|f| f.as_str().to_owned()).collect(),
            sample.sample.tags().iter().map(|f| f.as_str().to_owned()).collect(),
            graph,
        )?;
        out.conllu = Some(sample);
        Ok(out)
    }
    /// Replaces the tree without dropping source fields or view/MWT alignment.
    pub fn with_layer(mut self, layer: Dependencies<A>) -> Result<Self> {
        if layer.len() != self.tokens.len() {
            return Err(error("layer length differs from sentence"));
        }
        let graph = DependencyGraph::from_layer(layer)?;
        graph.check_tree()?;
        self.graph = graph;
        Ok(self)
    }
    pub fn to_conllu(&self) -> Result<ConlluDependencySample> {
        let mut out = self.conllu.clone().ok_or_else(|| error("CoNLL-U alignment metadata required"))?;
        let heads = (0..self.graph.len())
            .map(|i| self.graph.arc(Ix::new(i as u32)).unwrap().head.map_or(Head::Root, |h| Head::Word(h.get() as u32)))
            .collect();
        let relations = (0..self.graph.len())
            .map(|i| {
                Field::made(
                    self.graph.arc(Ix::new(i as u32)).unwrap().relation.to_owned(),
                    out.sample.tree().relations()[i].span,
                )
            })
            .collect();
        let tree = DependencyTree::new(heads, relations, WhitespaceMode::Unicode).map_err(error)?;
        out.sample =
            FormatSample::new(out.sample.tokens().to_vec(), out.sample.tags().to_vec(), tree, out.sample.source())
                .map_err(error)?;
        Ok(out)
    }
}
/// Oracle events in sentence/transition order. Non-projective samples are
/// skipped before any events are emitted, matching the Java event stream.
#[derive(Default)]
pub struct DependencyEventStream {
    pub events: Vec<Event>,
    pub skipped_nonprojective: usize,
}
impl DependencyEventStream {
    pub fn generate<A: WordOrTokenAnchor>(samples: &[DependencySample<A>]) -> Result<Self> {
        let mut out = Self::default();
        for sample in samples {
            if !sample.graph.is_projective() {
                out.skipped_nonprojective += 1;
                continue;
            }
            let steps = ArcStandardOracle::transitions(&sample.graph)?;
            let mut state = ArcStandardState::<A>::new(sample.tokens.len())?;
            let tokens: Vec<_> = sample.tokens.iter().map(String::as_str).collect();
            let tags: Vec<_> = sample.tags.iter().map(String::as_str).collect();
            for step in steps {
                let context = DependencyContextGenerator::context(&state, &tokens, &tags)?;
                out.events.push(Event::new(step.encode(), context.into_iter().map(Into::into).collect()));
                state.apply(&step)?;
            }
        }
        Ok(out)
    }
}
