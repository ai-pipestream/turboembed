use crate::{ArcStandardState, ParserInput, Position, Result, error};
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureLayout {
    /// The pinned OpenNLP implementation: 14 words, 14 POS, 8 labels.
    OpenNlp14,
    /// Chen and Manning: 18 words, 18 POS, 12 labels.
    ChenManning18,
}
impl FeatureLayout {
    pub fn positions(self) -> usize {
        match self {
            Self::OpenNlp14 => 14,
            Self::ChenManning18 => 18,
        }
    }
    pub fn feature_count(self) -> usize {
        self.positions() * 3 - 6
    }
}

/// Typed feature groups before concatenation into the network input order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedforwardFeatureIds {
    pub word_ids: Vec<usize>,
    pub pos_ids: Vec<usize>,
    pub label_ids: Vec<usize>,
}
impl FeedforwardFeatureIds {
    pub fn flatten(&self) -> Vec<usize> {
        self.word_ids.iter().chain(&self.pos_ids).chain(&self.label_ids).copied().collect()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedforwardContext {
    pub words: Vec<Option<String>>,
    pub pos: Vec<Option<String>>,
    pub labels: Vec<Option<String>>,
}
impl FeedforwardContext {
    /// Extracts the word, POS and label ID groups for the model's template.
    pub fn extract_ids<A: WordOrTokenAnchor>(
        state: &ArcStandardState<A>,
        words: &[&str],
        tags: &[&str],
        model: &super::FeedforwardDependencyModel,
    ) -> Result<FeedforwardFeatureIds> {
        let context = Self::extract(state, words, tags, model.layout())?;
        let ids = model.feature_ids(&context)?;
        let n = model.layout().positions();
        Ok(FeedforwardFeatureIds {
            word_ids: ids[..n].to_vec(),
            pos_ids: ids[n..2 * n].to_vec(),
            label_ids: ids[2 * n..].to_vec(),
        })
    }
    /// Slots: s0,s1,s2,b0,b1,b2; for s0 then s1, first left/right children
    /// followed by second left/right children in the expanded layout;
    /// finally left-of-left and right-of-right children of s0,s1.
    pub fn extract<A: WordOrTokenAnchor>(
        state: &ArcStandardState<A>,
        words: &[&str],
        tags: &[&str],
        layout: FeatureLayout,
    ) -> Result<Self> {
        ParserInput::check(words, tags)?;
        if state.token_count() != words.len() {
            return Err(error("state/input length mismatch"));
        }
        let mut positions = Vec::with_capacity(layout.positions());
        Self::positions(state, layout, &mut positions);
        let strings = |source: &[&str]| {
            positions
                .iter()
                .map(|p| match p {
                    Position::Root => Some("*ROOT*".into()),
                    Position::Missing => None,
                    Position::Node(i) => Some(source[i.get()].into()),
                })
                .collect()
        };
        let labels = positions[6..]
            .iter()
            .map(|p| match p {
                Position::Node(i) => state.relations[i.get()].clone(),
                _ => None,
            })
            .collect();
        Ok(Self { words: strings(words), pos: strings(tags), labels })
    }
    /// The layout's positions for `state`, in slot order, into `out`
    /// (cleared first).
    pub(super) fn positions<A: WordOrTokenAnchor>(
        state: &ArcStandardState<A>,
        layout: FeatureLayout,
        out: &mut Vec<Position<A>>,
    ) {
        out.clear();
        let s = [state.stack(0), state.stack(1)];
        out.extend([s[0], s[1], state.stack(2), state.buffer(0), state.buffer(1), state.buffer(2)]);
        for node in s {
            out.extend([child(state, node, false, 0, layout), child(state, node, true, 0, layout)]);
            if layout == FeatureLayout::ChenManning18 {
                out.extend([child(state, node, false, 1, layout), child(state, node, true, 1, layout)]);
            }
        }
        for node in s {
            out.push(child(state, child(state, node, false, 0, layout), false, 0, layout));
            out.push(child(state, child(state, node, true, 0, layout), true, 0, layout));
        }
    }
}
fn child<A: WordOrTokenAnchor>(
    state: &ArcStandardState<A>,
    node: Position<A>,
    right: bool,
    rank: usize,
    layout: FeatureLayout,
) -> Position<A> {
    let Position::Node(n) = node else {
        return Position::Missing;
    };
    let id = if layout == FeatureLayout::OpenNlp14 {
        if right { state.right[n.get()] } else { state.left[n.get()] }
    } else if right {
        state.directional_right[n.get()][rank]
    } else {
        state.directional_left[n.get()][rank]
    };
    id.map_or(Position::Missing, Position::Node)
}
