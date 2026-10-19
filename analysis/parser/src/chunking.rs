//! Bottom-up shift-reduce chunking constituency parser and its context generators
//! (`opennlp.tools.parser.chunking.*`). The event stream and the trainer are
//! `turbo_train_components::parser::chunking`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use std::fmt;
use turbo_analysis::Span;
use turbo_jvm::strict_math;
use turbo_sequence::Features;

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, collapse_punctuation_in_place,
    parse_bottom_up,
};
use crate::context::{
    ConsRef, DisplayFn, FeatureSink, check_cons_into as check_cons, check_cons_pair_into as check_cons_pair,
    cons2_into as cons2, cons3_into as cons3, surround_into as surround,
};
use crate::head_rules::HeadRules;
use crate::model::ParserModel;
use crate::parse::{COMPLETE, CONT, INCOMPLETE, Parse, ParseError, START, TOP_NODE};

/// Top-start outcome (`AbstractBottomUpParser.TOP_START = "S-TOP"`).
pub const TOP_START: &str = "S-TOP";

// Decode length-prefixed dictionary keys once, so n-gram lookup needs no temporary key.
#[derive(Clone, Debug, Default)]
struct DictionaryNode {
    terminal: bool,
    children: HashMap<String, DictionaryNode>,
}

impl DictionaryNode {
    fn from_keys(keys: HashSet<String>) -> Self {
        let mut root = Self::default();
        for key in keys {
            let mut rest = key.as_str();
            let mut words = Vec::new();
            while let Some((length, tail)) = rest.split_once(':') {
                let Ok(n) = length.parse::<usize>() else {
                    break;
                };
                if n.to_string() != length {
                    break;
                }
                let Some(word) = tail.get(..n) else {
                    break;
                };
                words.push(word);
                rest = &tail[n..];
            }
            if !rest.is_empty() {
                continue;
            }
            let mut node = &mut root;
            for word in words {
                node = node.children.entry(word.to_owned()).or_default();
            }
            node.terminal = true;
        }
        root
    }

    fn contains(&self, words: &[&Parse]) -> bool {
        let mut node = self;
        for word in words {
            let Some(next) = node.children.get(word.head_text()) else {
                return false;
            };
            node = next;
        }
        node.terminal
    }
}

/// Context generator for the shift-reduce build phase (`opennlp.tools.parser.chunking.BuildContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct BuildContextGenerator {
    dict: Option<DictionaryNode>,
}

impl BuildContextGenerator {
    /// Creates a [`BuildContextGenerator`] without a dictionary, so every lexical feature is
    /// used (`new BuildContextGenerator()`).
    pub fn new() -> Self {
        Self { dict: None }
    }

    /// Creates a [`BuildContextGenerator`] whose lexical features are limited to the n-grams of
    /// `dict`, keyed by [`crate::bottom_up::dictionary_key`] (`new BuildContextGenerator(Dictionary)`).
    pub fn with_dictionary(dict: HashSet<String>) -> Self {
        Self { dict: Some(DictionaryNode::from_keys(dict)) }
    }

    /// Generates build features for `constituents[index]` (`BuildContextGenerator.getContext(Parse[], int)`).
    pub fn get_context(&self, constituents: &[Parse], index: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
        self.populate(constituents, index, &mut features);
        features
    }

    /// Replaces `out` with build features, reusing its byte storage.
    pub fn get_features(&self, constituents: &[Parse], index: usize, out: &mut Features) {
        out.clear();
        self.populate(constituents, index, out);
    }

    fn populate<S: FeatureSink>(&self, constituents: &[Parse], index: usize, features: &mut S) {
        let ps = constituents.len();

        let p_2 = index.checked_sub(2).map(|j| &constituents[j]);
        let p_1 = index.checked_sub(1).map(|j| &constituents[j]);
        let punct_2s = p_1.map(Parse::previous_punctuation);
        let p0 = &constituents[index];
        let punct_1s = Some(p0.previous_punctuation());
        let punct1s = Some(p0.next_punctuation());
        let p1 = constituents.get(index + 1);
        let punct2s = p1.map(Parse::next_punctuation);
        let p2 = constituents.get(index + 2);

        let mut u_2 = true;
        let mut u_1 = true;
        let mut u0 = true;
        let mut u1 = true;
        let mut u2 = true;
        let mut b_2_1 = true;
        let mut b_10 = true;
        let mut b01 = true;
        let mut b12 = true;
        let mut t_2_10 = true;
        let mut t_101 = true;
        let mut t012 = true;

        if let Some(dict) = &self.dict {
            let has = |ngram: &[&Parse]| dict.contains(ngram);
            if let Some(p_2) = p_2 {
                u_2 = has(&[p_2]);
            }
            if let Some(p2) = p2 {
                u2 = has(&[p2]);
            }
            u0 = has(&[p0]);
            if let (Some(p_2), Some(p_1)) = (p_2, p_1) {
                b_2_1 = has(&[p_2, p_1]);
                t_2_10 = has(&[p_2, p_1, p0]);
            }
            if let (Some(p_1), Some(p1)) = (p_1, p1) {
                t_101 = has(&[p_1, p0, p1]);
            }
            if let Some(p_1) = p_1 {
                u_1 = has(&[p_1]);
                // OpenNLP's "extra check for 2==null case"; `u1` is still true here.
                b_2_1 = b_2_1 && u_1 && u_2;
                t_2_10 = t_2_10 && u_1 && u_2 && u0;
                t_101 = t_101 && u_1 && u0 && u1;
                b_10 = has(&[p_1, p0]) && u_1 && u0;
            }
            if let (Some(p1), Some(p2)) = (p1, p2) {
                b12 = has(&[p1, p2]);
                t012 = has(&[p0, p1, p2]);
            }
            if let Some(p1) = p1 {
                u1 = has(&[p1]);
                b12 = b12 && u1 && u2;
                t012 = t012 && u1 && u2 && u0;
                t_101 = t_101 && u0 && u_1 && u1;
                b01 = has(&[p0, p1]) && u0 && u1;
            }
        }

        let c_2 = ConsRef::new(p_2, -2, true, u_2, features);
        let c_1 = ConsRef::new(p_1, -1, true, u_1, features);
        let c0 = ConsRef::new(Some(p0), 0, true, u0, features);
        let c1 = ConsRef::new(p1, 1, true, u1, features);
        let c2 = ConsRef::new(p2, 2, true, u2, features);

        features.emit(format_args!("default"));
        for c in [&c0, &c_2, &c_1, &c1, &c2] {
            if c.unigram {
                features.emit(format_args!("{}", c.cons));
            }
            features.emit(format_args!("{}", c.cons_bo));
        }

        cons2(features, &c0, &c1, punct1s, b01);
        cons2(features, &c_1, &c0, punct_1s, b_10);
        cons3(features, &c0, &c1, &c2, punct1s, punct2s, t012, b01, b12);
        cons3(features, &c_2, &c_1, &c0, punct_2s, punct_1s, t_2_10, b_2_1, b_10);
        cons3(features, &c_1, &c0, &c1, punct_1s, punct1s, t_101, b_10, b01);

        // Walks back to the closest start; a node before `index` always has a label here, where
        // OpenNLP would throw `NullPointerException` on one without.
        let starts_constituent = |p: &Parse| p.label().is_some_and(|l| l.starts_with(START));
        let matches_back = |open: &str, feature: &str, features: &mut S| {
            for p in constituents[..index].iter().rev() {
                if p.kind() == open {
                    features.emit(format_args!("{feature}"));
                    break;
                }
                if starts_constituent(p) {
                    break;
                }
            }
        };
        match p0.kind() {
            "-RRB-" => matches_back("-LRB-", "bracketsmatch", features),
            "-RCB-" => matches_back("-LCB-", "bracketsmatch", features),
            "''" => matches_back("``", "quotesmatch", features),
            "'" => matches_back("`", "quotesmatch", features),
            "," => matches_back(",", "iscomma", features),
            "." if index + 1 == ps => {
                if let Some(pi) = constituents[..index].iter().rposition(starts_constituent)
                    && pi == 0
                {
                    features.emit(format_args!("endofsentence"));
                }
            }
            _ => {}
        }
    }
}

/// Context generator for the shift-reduce check phase (`opennlp.tools.parser.chunking.CheckContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct CheckContextGenerator;

impl CheckContextGenerator {
    /// Creates a new [`CheckContextGenerator`].
    pub fn new() -> Self {
        Self
    }

    /// Generates completeness check features for reducing `constituents[start..=end]` into `kind`
    /// (`CheckContextGenerator.getContext(Parse[], String, int, int)`).
    pub fn get_context(&self, constituents: &[Parse], kind: &str, start: usize, end: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
        self.populate(constituents, kind, start, end, constituents[0].label().unwrap_or("null"), &mut features);
        features
    }

    /// Replaces `out` with check features, reusing its byte storage.
    pub fn get_features(&self, constituents: &[Parse], kind: &str, start: usize, end: usize, out: &mut Features) {
        out.clear();
        self.populate(constituents, kind, start, end, constituents[0].label().unwrap_or("null"), out);
    }

    fn populate(
        &self,
        constituents: &[Parse],
        kind: &str,
        start: usize,
        end: usize,
        first_label: &str,
        features: &mut impl FeatureSink,
    ) {
        let ps = constituents.len();
        features.emit(format_args!("default"));
        features.emit(format_args!("fl={first_label}"));
        let pstart = &constituents[start];
        let pend = &constituents[end];
        check_cons(pstart, "begin", kind, features);
        check_cons(pend, "last", kind, features);
        for p in &constituents[start..end] {
            check_cons_pair(p, pend, kind, features);
        }
        for (prefix, include_punctuation) in [("p", false), ("pp", true)] {
            let production = DisplayFn(|f: &mut fmt::Formatter<'_>| {
                write!(f, "{prefix}={kind}->")?;
                for p in &constituents[start..end] {
                    write!(f, "{},", p.kind())?;
                    if include_punctuation {
                        for punct in p.next_punctuation() {
                            write!(f, "{},", punct.kind())?;
                        }
                    }
                }
                f.write_str(pend.kind())
            });
            features.emit(format_args!("{production}"));
        }

        let p1s = Some(constituents[end].next_punctuation());
        let p_1s = Some(constituents[start].previous_punctuation());
        let p_2 = start.checked_sub(2).map(|j| &constituents[j]);
        let p_1 = start.checked_sub(1).map(|j| &constituents[j]);
        let p_2s = p_1.map(Parse::previous_punctuation);
        let p1 = (end + 1 < ps).then(|| &constituents[end + 1]);
        let p2s = p1.map(Parse::next_punctuation);
        let p2 = (end + 2 < ps).then(|| &constituents[end + 2]);

        surround(p_1, -1, kind, p_1s, features);
        surround(p_2, -2, kind, p_2s, features);
        surround(p1, 1, kind, p1s, features);
        surround(p2, 2, kind, p2s, features);
    }
}

/// Shift-reduce constituency parser (`opennlp.tools.parser.chunking.Parser`).
#[derive(Debug)]
pub struct Parser {
    model: Arc<ParserModel>,
    beam_size: usize,
    advance_percentage: f64,
    build_cg: BuildContextGenerator,
    check_cg: CheckContextGenerator,
    start_type_map: HashMap<String, String>,
    cont_type_map: HashMap<String, String>,
    top_start_index: Option<usize>,
    complete_index: usize,
    incomplete_index: usize,
}

impl Parser {
    /// Creates a shift-reduce [`Parser`] with default beam size (`20`) and advance percentage (`0.95`).
    pub fn new(model: impl Into<Arc<ParserModel>>) -> Self {
        Self::with_params(model, DEFAULT_BEAM_SIZE, DEFAULT_ADVANCE_PERCENTAGE)
    }

    /// Creates a shift-reduce [`Parser`] with explicit `beam_size` and `advance_percentage`.
    pub fn with_params(model: impl Into<Arc<ParserModel>>, beam_size: usize, advance_percentage: f64) -> Self {
        let model = model.into();
        let build_model = model.build_model();
        let check_model = model.check_model();

        let mut start_type_map = HashMap::new();
        let mut cont_type_map = HashMap::new();
        for oi in 0..build_model.num_outcomes() {
            let outcome_cow = build_model.outcome(oi).to_str().unwrap_or_default();
            let outcome = outcome_cow.as_ref();
            if let Some(rest) = outcome.strip_prefix(START) {
                start_type_map.insert(outcome.to_owned(), rest.to_owned());
            } else if let Some(rest) = outcome.strip_prefix(CONT) {
                cont_type_map.insert(outcome.to_owned(), rest.to_owned());
            }
        }
        let top_start_index = build_model.outcome_index(TOP_START);
        let complete_index = check_model.outcome_index(COMPLETE).unwrap_or(0);
        let incomplete_index = check_model.outcome_index(INCOMPLETE).unwrap_or(0);

        Self {
            model,
            beam_size,
            advance_percentage,
            build_cg: BuildContextGenerator::new(),
            check_cg: CheckContextGenerator::new(),
            start_type_map,
            cont_type_map,
            top_start_index,
            complete_index,
            incomplete_index,
        }
    }

    /// Returns the underlying [`ParserModel`].
    pub fn model(&self) -> &ParserModel {
        &self.model
    }

    /// Merges the children of `p` into a single `TOP` root node (`chunking.Parser.merge(Parse)`).
    pub fn merge(p: &mut Parse) {
        let ptype = p.kind().to_owned();
        p.set_kind(TOP_NODE);
        if ptype != TOP_NODE && p.child_count() > 1 {
            let span = p.span();
            let text = Arc::clone(p.text_arc());
            let mut root = Parse::new(text, span, ptype, 0.0, 0);
            let kids: Vec<Parse> = p.children().to_vec();
            for kid in kids {
                root.insert(kid).expect("kid lies inside p");
            }
            while p.child_count() > 0 {
                p.remove(0);
            }
            p.insert(root).expect("root has p's span");
        }
    }

    fn advance_top(&self, p: &mut Parse, features: &mut Features) {
        let build_model = self.model.build_model();
        let check_model = self.model.check_model();
        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut cprobs = vec![0.0; check_model.num_outcomes()];

        self.build_cg.get_features(p.children(), 0, features);
        build_model.eval_features(features.iter(), &mut bprobs);
        if let Some(tsi) = self.top_start_index {
            p.add_prob(strict_math::log(bprobs[tsi]));
        }

        self.check_cg.get_features(p.children(), TOP_NODE, 0, 0, features);
        check_model.eval_features(features.iter(), &mut cprobs);
        p.add_prob(strict_math::log(cprobs[self.complete_index]));
        p.set_kind(TOP_NODE);
    }

    /// `chunking.Parser.advanceParses`. `None` is OpenNLP's `null`, for a parse without
    /// non-punctuation children. Where every child is labeled already OpenNLP indexes past the
    /// children and throws; this returns no derivations instead.
    fn advance_parses(&self, p: &mut Parse, prob_mass: f64, features: &mut Features) -> Option<Vec<Parse>> {
        let q = 1.0 - prob_mass;
        let punct_set = self.model.head_rules().punctuation_tags();
        let non_punct_indices = collapse_punctuation_in_place(p.children_mut(), punct_set);
        let children: Vec<Parse> = non_punct_indices.iter().map(|&idx| p.children()[idx].clone()).collect();
        let num_nodes = children.len();
        if num_nodes == 0 {
            return None;
        }

        let mut last_start_node: Option<&Parse> = None;
        let mut last_start_index: Option<usize> = None;
        let mut last_start_type: Option<&str> = None;
        let mut advance_node_index = num_nodes;

        for (pi, np) in children.iter().enumerate() {
            if let Some(lbl) = np.label() {
                if let Some(st) = self.start_type_map.get(lbl) {
                    last_start_type = Some(st.as_str());
                    last_start_node = Some(np);
                    last_start_index = Some(pi);
                }
            } else {
                advance_node_index = pi;
                break;
            }
        }
        if advance_node_index >= num_nodes {
            return Some(Vec::new());
        }
        let advance_node = &children[advance_node_index];
        let original_advance_index = non_punct_indices[advance_node_index];

        let build_model = self.model.build_model();
        let check_model = self.model.check_model();
        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut cprobs = vec![0.0; check_model.num_outcomes()];

        self.build_cg.get_features(&children, advance_node_index, features);
        build_model.eval_features(features.iter(), &mut bprobs);

        let mut new_parses_list = Vec::with_capacity(build_model.num_outcomes());
        let mut bprob_sum = 0.0;

        while bprob_sum < prob_mass {
            let mut max = 0;
            for pi in 1..bprobs.len() {
                if bprobs[pi] > bprobs[max] {
                    max = pi;
                }
            }
            if bprobs[max] == 0.0 {
                break;
            }
            let bprob = bprobs[max];
            bprobs[max] = 0.0;
            bprob_sum += bprob;

            let tag_cow = build_model.outcome(max).to_str().unwrap_or_default();
            let tag = tag_cow.as_ref();
            if Some(max) == self.top_start_index {
                continue;
            }
            if let Some(st) = self.start_type_map.get(tag) {
                last_start_index = Some(advance_node_index);
                last_start_node = Some(advance_node);
                last_start_type = Some(st.as_str());
            } else if let Some(ct) = self.cont_type_map.get(tag)
                && (last_start_node.is_none() || last_start_type != Some(ct.as_str()))
            {
                continue;
            }

            if let (Some(ls_idx), Some(ls_type), Some(ls_node)) = (last_start_index, last_start_type, last_start_node) {
                // Check features read only the first child's label. Supply the candidate
                // label directly; punctuation was already collapsed before this outcome loop.
                let first_label = if advance_node_index == 0 { tag } else { children[0].label().unwrap_or("null") };
                features.clear();
                self.check_cg.populate(&children, ls_type, ls_idx, advance_node_index, first_label, features);
                check_model.eval_features(features.iter(), &mut cprobs);

                let incomplete = cprobs[self.incomplete_index] > q && advance_node_index != num_nodes - 1;
                let flat = children[ls_idx..=advance_node_index].iter().all(Parse::is_pos_tag);
                let candidate = || {
                    let mut branch = p.clone();
                    branch.set_child(original_advance_index, tag);
                    branch.add_prob(strict_math::log(bprob));
                    branch
                };
                if cprobs[self.complete_index] > q && !flat {
                    let mut complete = candidate();
                    complete.add_prob(strict_math::log(cprobs[self.complete_index]));
                    let head = self.model.head_rules().get_head(&children[ls_idx..=advance_node_index], ls_type);
                    let node_prob = cprobs.get(1).copied().unwrap_or(cprobs[self.complete_index]);
                    let span = if ls_idx == 0 && advance_node_index == num_nodes - 1 {
                        p.span()
                    } else {
                        Span::new(ls_node.span().start, advance_node.span().end)
                    };
                    complete
                        .insert(Parse::with_head(Arc::clone(p.text_arc()), span, ls_type, node_prob, head))
                        .expect("reduced constituent span inside root");
                    new_parses_list.push(complete);
                }
                if incomplete {
                    let mut incomplete = candidate();
                    incomplete.add_prob(strict_math::log(cprobs[self.incomplete_index]));
                    new_parses_list.push(incomplete);
                }
            } else {
                // When tag is `O` and no prior start exists, OpenNLP calls checkContextGenerator
                // with `lastStartIndex = -1`, which would throw ArrayIndexOutOfBoundsException in
                // Java unless a start outcome was already seen or tag is in startTypeMap.
            }
        }

        Some(new_parses_list)
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        let mut features = Features::new();
        let mut top_features = Features::new();
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| Ok(self.advance_parses(p, q, &mut features)),
            |p| self.advance_top(p, &mut top_features),
            |_| {},
        )
    }
}
