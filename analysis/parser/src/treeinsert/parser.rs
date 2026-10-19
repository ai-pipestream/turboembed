//! The tree-insertion beam-search parser (`opennlp.tools.parser.treeinsert.Parser`).

use std::collections::HashSet;
use std::sync::Arc;

use turbo_jvm::strict_math;
use turbo_sequence::Features;

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, collapse_punctuation_in_place,
    parse_bottom_up,
};
use crate::head_rules::HeadRules;
use crate::model::ParserModel;
use crate::parse::{COMPLETE, Parse, ParseError, TOP_NODE};

use super::*;

/// Tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.Parser`).
///
/// OpenNLP's `checkComplete` switch is a package-private static that is always `false`, so the
/// check model is never consulted while parsing and its branches are not ported.
#[derive(Debug)]
pub struct Parser {
    model: Arc<ParserModel>,
    beam_size: usize,
    advance_percentage: f64,
    build_cg: BuildContextGenerator,
    attach_cg: AttachContextGenerator,
    /// Outcome indices, `None` where OpenNLP's `getIndex` gives `-1`; reading one fails as
    /// OpenNLP's read of index `-1` does.
    done_index: Option<usize>,
    sister_attach_index: Option<usize>,
    daughter_attach_index: Option<usize>,
}

impl Parser {
    /// Creates a tree-insertion [`Parser`] with default beam size (`20`) and advance percentage (`0.95`).
    ///
    /// # Panics
    ///
    /// As [`Parser::with_params`].
    pub fn new(model: impl Into<Arc<ParserModel>>) -> Self {
        Self::with_params(model, DEFAULT_BEAM_SIZE, DEFAULT_ADVANCE_PERCENTAGE)
    }

    /// Creates a tree-insertion [`Parser`] with explicit `beam_size` and `advance_percentage`.
    ///
    /// # Panics
    ///
    /// When [`Parser::try_with_params`] fails.
    pub fn with_params(model: impl Into<Arc<ParserModel>>, beam_size: usize, advance_percentage: f64) -> Self {
        Self::try_with_params(model, beam_size, advance_percentage).expect("TREEINSERT parser model")
    }

    /// Creates a tree-insertion [`Parser`], failing when the model has no `attach.model`.
    pub fn try_with_params(
        model: impl Into<Arc<ParserModel>>,
        beam_size: usize,
        advance_percentage: f64,
    ) -> Result<Self, TreeInsertError> {
        let model = model.into();
        let attach_model = model.attach_model().ok_or(TreeInsertError::MissingAttachModel)?;
        let done_index = model.build_model().outcome_index(DONE);
        let sister_attach_index = attach_model.outcome_index(ATTACH_SISTER);
        let daughter_attach_index = attach_model.outcome_index(ATTACH_DAUGHTER);
        let punct_set = model.head_rules().punctuation_tags().clone();
        Ok(Self {
            model,
            beam_size,
            advance_percentage,
            build_cg: BuildContextGenerator::new(),
            attach_cg: AttachContextGenerator::new(punct_set),
            done_index,
            sister_attach_index,
            daughter_attach_index,
        })
    }

    /// Returns the underlying [`ParserModel`].
    pub fn model(&self) -> &ParserModel {
        &self.model
    }

    /// Returns the right frontier of `root`, deepest node first (`treeinsert.Parser.getRightFrontier`).
    ///
    /// A `TOP` or `INC` root starts from its first child that is not in `punct_set`, any other
    /// root from itself; the walk follows last children down to the first part-of-speech node,
    /// which is left out.
    pub fn get_right_frontier<'a>(root: &'a Parse, punct_set: &HashSet<String>) -> Vec<&'a Parse> {
        right_frontier_paths(root, punct_set).iter().map(|path| root.node_at_path(path)).collect()
    }

    fn is_built(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.starts_with(BUILT))
    }

    fn is_complete(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.ends_with(COMPLETE))
    }

    /// `treeinsert.Parser.setComplete`.
    fn set_complete(p: &mut Parse) {
        let label = if Self::is_built(p) { format!("{BUILT}.{COMPLETE}") } else { COMPLETE.to_owned() };
        p.set_label(Some(label));
    }

    /// Marks every top-level child of each parse complete, as `treeinsert.Parser.advanceChunks`
    /// does after the chunker has run.
    pub fn mark_chunks_complete(parses: &mut [Parse]) {
        for parse in parses {
            for chunk in parse.children_mut() {
                Self::set_complete(chunk);
            }
        }
    }

    /// Clears the punctuation sets of `p`, as a node copied by `Parse.clone` has none.
    fn clear_punctuation(p: &mut Parse) {
        p.set_previous_punctuation(Vec::new());
        p.set_next_punctuation(Vec::new());
    }

    /// Advances `p` one build or attach step (`treeinsert.Parser.advanceParses`).
    ///
    /// Returns `None` where OpenNLP returns `null`: when `p` has no constituent that is not
    /// punctuation, or a single part-of-speech node. The caller then completes `p` as it is. A
    /// single other constituent takes the punctuation around it as children and `p` is returned.
    /// Otherwise the derivations are returned in OpenNLP's order: one per build outcome taken,
    /// then one per daughter or sister attachment taken, frontier node by frontier node.
    ///
    /// Fails where OpenNLP throws: when the head rules give no head for a node an attachment
    /// changes, or when every top-level constituent is already built, where OpenNLP reads past
    /// the end of the constituents.
    pub fn advance_parses(&self, p: &mut Parse, prob_mass: f64) -> Result<Option<Vec<Parse>>, ParseError> {
        self.advance_parses_with_features(p, prob_mass, &mut Features::new())
    }

    fn advance_parses_with_features(
        &self,
        p: &mut Parse,
        prob_mass: f64,
        features: &mut Features,
    ) -> Result<Option<Vec<Parse>>, ParseError> {
        let q = 1.0 - prob_mass;
        let rules = self.model.head_rules();
        let punct_set = rules.punctuation_tags();
        let build_model = self.model.build_model();
        let attach_model = self.model.attach_model().expect("checked by Parser::try_with_params");

        let kept = collapse_punctuation_in_place(p.children_mut(), punct_set);
        let num_nodes = kept.len();
        if num_nodes == 0 {
            return Ok(None);
        }
        if num_nodes == 1 {
            if p.children()[kept[0]].is_pos_tag() {
                return Ok(None);
            }
            p.expand_top_node(kept[0])?;
            return Ok(Some(vec![p.clone()]));
        }
        let advance_node_index = (0..num_nodes)
            .find(|&i| !Self::is_built(&p.children()[kept[i]]))
            .ok_or(ParseError::TokenIndex { index: num_nodes as i64, len: num_nodes })?;
        let original_zero_index = kept[0];
        let original_advance_index = kept[advance_node_index];

        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut rf: Vec<Vec<usize>> = Vec::new();
        let mut aprobs: Vec<Vec<f64>> = Vec::new();
        {
            let children: Vec<&Parse> = kept.iter().map(|&i| &p.children()[i]).collect();
            self.build_cg.get_features(&children, advance_node_index, features);
            build_model.eval_features(features.iter(), &mut bprobs);
        }
        let done_prob = bprobs[outcome_index(self.done_index, &bprobs)?];

        if done_prob > q && advance_node_index != 0 {
            rf = right_frontier_paths(p, punct_set);
            {
                let children: Vec<&Parse> = kept.iter().map(|&i| &p.children()[i]).collect();
                let rf_nodes: Vec<&Parse> = rf.iter().map(|path| p.node_at_path(path)).collect();
                for fi in 0..rf.len() {
                    self.attach_cg.get_features(&children, advance_node_index, &rf_nodes, fi, features);
                    let mut probs = vec![0.0; attach_model.num_outcomes()];
                    attach_model.eval_features(features.iter(), &mut probs);
                    aprobs.push(probs);
                }
            }
            // `production(fn, false)` collapses the children of every frontier node.
            for path in &rf {
                collapse_children(p.node_at_path_mut(path), punct_set);
            }
        }

        let advance_node = p.children()[original_advance_index].clone();
        let mut new_parses = Vec::new();

        if 1.0 - done_prob > q {
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
                let tag = build_model.outcome(max).to_str().unwrap_or_default();
                if tag != DONE {
                    let mut new_parse = p.clone();
                    let new_node = Parse::with_head(
                        Arc::clone(p.text_arc()),
                        advance_node.span(),
                        tag.as_ref(),
                        bprob,
                        Some(advance_node.head()),
                    );
                    new_parse.insert(new_node)?;
                    new_parse.add_prob(strict_math::log(bprob));
                    new_parses.push(new_parse);
                }
            }
        }

        if done_prob > q {
            let prepare_done = || {
                let mut branch = p.clone();
                let built = &mut branch.children_mut()[original_advance_index];
                built.set_label(Some(BUILT.to_owned()));
                Self::clear_punctuation(built);
                branch.add_prob(strict_math::log(done_prob));
                branch
            };
            if advance_node_index == 0 {
                new_parses.push(prepare_done());
            } else {
                for (fi, probs) in aprobs.iter().enumerate() {
                    let daughter = outcome_index(self.daughter_attach_index, probs)?;
                    let sister = outcome_index(self.sister_attach_index, probs)?;
                    let fn_complete = Self::is_complete(p.node_at_path(&rf[fi]));
                    for attachment in [daughter, sister] {
                        let prob = probs[attachment];
                        if !(prob > q && (attachment != daughter || !fn_complete)) {
                            continue;
                        }
                        // `cloneRoot(fn, originalZeroIndex)` copies the nodes from the top down to fn.
                        let mut new_parse = prepare_done();
                        for path in &rf[fi..] {
                            Self::clear_punctuation(new_parse.node_at_path_mut(path));
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        for _ in original_zero_index + 1..=original_advance_index {
                            new_parse.remove(original_zero_index + 1)?;
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        let crf = right_frontier_paths(&new_parse, punct_set);
                        if attachment == daughter {
                            new_parse.node_at_path_mut(&crf[fi]).add(advance_node.clone(), rules)?;
                        } else if fi + 1 < crf.len() {
                            new_parse.node_at_path_mut(&crf[fi + 1]).adjoin(advance_node.clone(), rules)?;
                        } else {
                            new_parse.adjoin_root(advance_node.clone(), rules, original_zero_index)?;
                        }
                        for path in &crf[fi + 1..] {
                            new_parse.node_at_path_mut(path).update_span();
                        }
                        new_parse.add_prob(strict_math::log(prob));
                        new_parses.push(new_parse);
                    }
                }
            }
        }
        Ok(Some(new_parses))
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        let mut features = Features::new();
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| self.advance_parses_with_features(p, q, &mut features),
            |p| p.set_kind(TOP_NODE),
            Self::mark_chunks_complete,
        )
    }
}
