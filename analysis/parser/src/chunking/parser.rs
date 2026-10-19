//! The shift-reduce beam-search parser (`opennlp.tools.parser.chunking.Parser`).

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_jvm::strict_math;
use turbo_sequence::Features;

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, collapse_punctuation_in_place,
    parse_bottom_up,
};
use crate::head_rules::HeadRules;
use crate::model::ParserModel;
use crate::parse::{COMPLETE, CONT, INCOMPLETE, Parse, ParseError, START, TOP_NODE};

use super::*;

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
                p.remove(0).expect("child 0 exists");
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
                    branch.set_child(original_advance_index, tag).expect("the advance index is a child of the clone");
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
