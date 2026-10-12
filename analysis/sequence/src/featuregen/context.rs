//! Generators that read the token's place in the sentence or document,
//! its neighbours, or the outcomes before it.

use super::FeatureGenerator;
use crate::features::Features;
use crate::pattern::{PatternMode, token_feature};

/// `SentenceFeatureGenerator`: `S=begin` on the first token and `S=end`
/// on the last, each when asked for.
pub struct SentencePosition {
    begin: bool,
    end: bool,
}

impl SentencePosition {
    pub fn new(begin: bool, end: bool) -> SentencePosition {
        SentencePosition { begin, end }
    }
}

impl FeatureGenerator for SentencePosition {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        if self.begin && index == 0 {
            out.add("S=begin");
        }
        if self.end && tokens.len() == index + 1 {
            out.add("S=end");
        }
    }
}

/// `OutcomePriorFeatureGenerator`: the feature `def` on every token, whose
/// parameters are the outcomes' prior. `DefinitionFeatureGeneratorFactory`
/// makes this generator too.
pub struct OutcomePrior;

impl FeatureGenerator for OutcomePrior {
    fn create_features(&mut self, out: &mut Features, _tokens: &[&str], _index: usize, _previous: &[&str]) {
        out.add("def");
    }
}

/// `DocumentBeginFeatureGenerator`: `D=begin` on the first token of the
/// first sentence since the last
/// [`clear_adaptive_data`](FeatureGenerator::clear_adaptive_data).
///
/// Java remembers the first sentence's `String[]` and compares later
/// arrays with it by identity, so tagging that same array again gives
/// `D=begin` again. Here sentences are told apart by
/// [`start_sequence`](FeatureGenerator::start_sequence): every call to it
/// is a new sentence, the same array or not.
#[derive(Default)]
pub struct DocumentBegin {
    first: Option<u64>,
    current: u64,
}

impl DocumentBegin {
    pub fn new() -> DocumentBegin {
        DocumentBegin::default()
    }
}

impl FeatureGenerator for DocumentBegin {
    fn create_features(&mut self, out: &mut Features, _tokens: &[&str], index: usize, _previous: &[&str]) {
        let first = *self.first.get_or_insert(self.current);
        if first == self.current && index == 0 {
            out.add("D=begin");
        }
    }

    fn clear_adaptive_data(&mut self) {
        self.first = None;
    }

    fn start_sequence(&mut self) {
        self.current += 1;
    }
}

/// `AdditionalContextFeatureGenerator`: `ne=` and each string the caller
/// gave for the token, from
/// [`set_additional_context`](FeatureGenerator::set_additional_context).
/// `NameFinderME` puts one in a window of eight tokens on each side after
/// the model's own generators.
///
/// # Panics
///
/// When a non-empty additional context has no entry for the token, as
/// Java throws `ArrayIndexOutOfBoundsException`.
#[derive(Default)]
pub struct AdditionalContext {
    context: Vec<Vec<String>>,
}

impl AdditionalContext {
    pub fn new() -> AdditionalContext {
        AdditionalContext::default()
    }
}

impl FeatureGenerator for AdditionalContext {
    fn create_features(&mut self, out: &mut Features, _tokens: &[&str], index: usize, _previous: &[&str]) {
        if self.context.is_empty() {
            return;
        }
        for s in &self.context[index] {
            out.push_str("ne=").push_str(s).finish();
        }
    }

    fn set_additional_context(&mut self, context: &[Vec<String>]) {
        self.context.clear();
        self.context.extend_from_slice(context);
    }
}

/// `BigramNameFeatureGenerator`: the token with the one before it
/// (`pw,w=`) and their classes (`pwc,wc=`), then with the one after it
/// (`w,nw=`, `wc,nc=`).
pub struct BigramName {
    mode: PatternMode,
}

impl BigramName {
    pub fn new(mode: PatternMode) -> BigramName {
        BigramName { mode }
    }
}

impl FeatureGenerator for BigramName {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let wc = token_feature(token, self.mode);
        if index > 0 {
            let prev = tokens[index - 1];
            out.push_str("pw,w=").push_str(prev).push_str(",").push_str(token).finish();
            out.push_str("pwc,wc=").push_str(token_feature(prev, self.mode)).push_str(",").push_str(wc).finish();
        }
        if index + 1 < tokens.len() {
            let next = tokens[index + 1];
            out.push_str("w,nw=").push_str(token).push_str(",").push_str(next).finish();
            out.push_str("wc,nc=").push_str(wc).push_str(",").push_str(token_feature(next, self.mode)).finish();
        }
    }
}

/// `TrigramNameFeatureGenerator`: the token with the two before it
/// (`ppw,pw,w=`) and their classes (`ppwc,pwc,wc=`), then with the two
/// after it (`w,nw,nnw=`, `wc,nwc,nnwc=`).
pub struct TrigramName {
    mode: PatternMode,
}

impl TrigramName {
    pub fn new(mode: PatternMode) -> TrigramName {
        TrigramName { mode }
    }
}

impl FeatureGenerator for TrigramName {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        let token = tokens[index];
        let class = |t: &str| token_feature(t, self.mode);
        let wc = class(token);
        if index > 1 {
            let (pp, p) = (tokens[index - 2], tokens[index - 1]);
            out.push_str("ppw,pw,w=").push_str(pp).push_str(",").push_str(p).push_str(",").push_str(token).finish();
            out.push_str("ppwc,pwc,wc=")
                .push_str(class(pp))
                .push_str(",")
                .push_str(class(p))
                .push_str(",")
                .push_str(wc)
                .finish();
        }
        if index + 2 < tokens.len() {
            let (n, nn) = (tokens[index + 1], tokens[index + 2]);
            out.push_str("w,nw,nnw=").push_str(token).push_str(",").push_str(n).push_str(",").push_str(nn).finish();
            out.push_str("wc,nwc,nnwc=")
                .push_str(wc)
                .push_str(",")
                .push_str(class(n))
                .push_str(",")
                .push_str(class(nn))
                .finish();
        }
    }
}

/// `PosTaggerFeatureGenerator`: the previous outcome (`t=`) and the two
/// before the token (`t2=`, the earlier first). Nothing on the first
/// token.
pub struct PreviousTags;

impl FeatureGenerator for PreviousTags {
    fn create_features(&mut self, out: &mut Features, _tokens: &[&str], index: usize, previous: &[&str]) {
        if index >= 1 {
            out.push_str("t=").push_str(previous[index - 1]).finish();
            if index >= 2 {
                out.push_str("t2=").push_str(previous[index - 2]).push_str(",").push_str(previous[index - 1]).finish();
            }
        }
    }
}
