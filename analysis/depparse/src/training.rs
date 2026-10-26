use crate::{ArcStandardOracle, DependencyGraph, LabelEncoding, Result, SwapOracle, Transition, error};
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainingMode {
    Projective,
    PseudoProjective,
    Swap,
}
impl TrainingMode {
    pub(crate) fn graph<A: WordOrTokenAnchor>(self, graph: &DependencyGraph<A>) -> Result<DependencyGraph<A>> {
        if self == Self::PseudoProjective {
            return graph.projectivize();
        }
        if graph.label_encoding != LabelEncoding::Plain {
            return Err(error("training requires unencoded source labels"));
        }
        if self == Self::Projective && !graph.is_projective() {
            return Err(error("projective training requires plain projective trees"));
        }
        Ok(graph.clone())
    }
    pub(crate) fn transitions<A: WordOrTokenAnchor>(self, graph: &DependencyGraph<A>) -> Result<Vec<Transition>> {
        if self == Self::Swap { SwapOracle::transitions(graph) } else { ArcStandardOracle::transitions(graph) }
    }
    pub(crate) fn encoding(self) -> LabelEncoding {
        if self == Self::PseudoProjective { LabelEncoding::EscapedV1 } else { LabelEncoding::Plain }
    }
}
