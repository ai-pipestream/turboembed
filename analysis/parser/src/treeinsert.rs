//! Bottom-up tree-insertion constituency parser, context generators, event stream, and trainer
//! (`opennlp.tools.parser.treeinsert.*`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_formats::{Field, PosSample};
use turbo_jvm::strict_math;
use turbo_train::trainer;
use turbo_train::{Event, TrainingConfiguration};
use turbo_train_components::package::Stamp;
use turbo_train_components::postag::{BEAM_SIZE_KEY, PosTaggerFactory, train_pos_model};

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, build_dictionary, collapse_punctuation,
    collapse_punctuation_in_place, parse_bottom_up,
};
use crate::chunking::{ParserTrainingParameters, collect_pos_samples, event_from_strings, train_parser_chunker_bytes};
use crate::context::{
    ChunkContextGenerator, Cons, ConsWindow, check_cons, check_cons_pair, cons, cons_bo, cons2, cons3, get_nodes,
    production, surround,
};
use crate::head_rules::HeadRules;
use crate::model::{HeadRulesArtifact, ParserEventType, ParserModel, ParserModelError, ParserType};
use crate::parse::{COMPLETE, INC_NODE, INCOMPLETE, Parse, TOP_NODE};
use crate::stream::{ParserChunkSample, ParserPosSample};

/// Outcome indicating no further unary build levels (`treeinsert.Parser.DONE = "d"`).
pub const DONE: &str = "d";

/// Outcome indicating sister adjunction (`treeinsert.Parser.ATTACH_SISTER = "s"`).
pub const ATTACH_SISTER: &str = "s";

/// Outcome indicating daughter attachment (`treeinsert.Parser.ATTACH_DAUGHTER = "d"`).
pub const ATTACH_DAUGHTER: &str = "d";

/// Outcome indicating non-attachment (`treeinsert.Parser.NON_ATTACH = "n"`).
pub const NON_ATTACH: &str = "n";

/// Label prefix marking a constituent as already built (`treeinsert.Parser.BUILT = "b"`).
pub const BUILT: &str = "b";

/// Context generator for the tree-insertion build phase (`opennlp.tools.parser.treeinsert.BuildContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct BuildContextGenerator;

impl BuildContextGenerator {
    /// Creates a new [`BuildContextGenerator`].
    pub fn new() -> Self {
        Self
    }

    /// Generates build features for `constituents[index]` (`BuildContextGenerator.getContext(Parse[], int)`).
    pub fn get_context(&self, constituents: &[Parse], index: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(64);
        let nodes = get_nodes(constituents, index);
        let [_, cf_1, cf0, cf1, _] = nodes;
        let c0_node = cf0.expect("index in bounds of constituents");

        let punct_2s = cf_1.map(Parse::previous_punctuation);
        let punct_1s = Some(c0_node.previous_punctuation());
        let punct1s = Some(c0_node.next_punctuation());
        let punct2s = cf1.map(Parse::next_punctuation);

        let w = ConsWindow::build(nodes, None);

        features.push("default".to_owned());

        // unigrams
        features.push(w.c_2.cons_bo.clone());
        features.push(w.c_2.cons.clone());
        features.push(w.c_1.cons_bo.clone());
        features.push(w.c_1.cons.clone());
        features.push(w.c0.cons_bo.clone());
        features.push(w.c0.cons.clone());
        features.push(w.c1.cons_bo.clone());
        features.push(w.c1.cons.clone());
        features.push(w.c2.cons_bo.clone());
        features.push(w.c2.cons.clone());

        // bigrams
        cons2(&mut features, &w.c_2, &w.c_1, punct_2s, true);
        cons2(&mut features, &w.c_1, &w.c0, punct_1s, true);
        cons2(&mut features, &w.c0, &w.c1, punct1s, true);
        cons2(&mut features, &w.c1, &w.c2, punct2s, true);

        // trigrams (note: OpenNLP passes `punct_1s` twice in the middle trigram)
        cons3(&mut features, &w.c_2, &w.c_1, &w.c0, punct_2s, punct_1s, true, true, true);
        cons3(&mut features, &w.c_1, &w.c0, &w.c1, punct_1s, punct_1s, true, true, true);
        cons3(&mut features, &w.c0, &w.c1, &w.c2, punct1s, punct2s, true, true, true);

        if cf1.is_none() {
            features.push("eos".to_owned());
        }

        features
    }
}

/// Context generator for the tree-insertion attachment phase (`opennlp.tools.parser.treeinsert.AttachContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct AttachContextGenerator;

impl AttachContextGenerator {
    /// Creates a new [`AttachContextGenerator`].
    pub fn new() -> Self {
        Self
    }

    /// Generates attachment features for attaching `constituents[index]` to `right_frontier[rfi]`
    /// (`AttachContextGenerator.getContext(Parse[], int, List<Parse>, int)`).
    pub fn get_context(
        &self,
        constituents: &[Parse],
        index: usize,
        right_frontier: &[Parse],
        rfi: usize,
    ) -> Vec<String> {
        let mut features = Vec::with_capacity(64);
        let fn_node = &right_frontier[rfi];
        let fp = right_frontier.get(rfi + 1);
        let p0 = &constituents[index];
        let p1 = constituents.get(index + 1);

        let punct_1s = Some(fn_node.previous_punctuation());
        let punct1s = Some(p0.previous_punctuation());
        let punct_1_fs = Some(p0.next_punctuation());

        let cons_fp = cons(fp, -2);
        let cons_fn = cons(Some(fn_node), -1);
        let cons_p0 = cons(Some(p0), 0);
        let cons_p1 = cons(p1, 1);

        let cons_bo_fp = cons_bo(fp, -2);
        let cons_bo_fn = cons_bo(Some(fn_node), -1);
        let cons_bo_p0 = cons_bo(Some(p0), 0);
        let cons_bo_p1 = cons_bo(p1, 1);

        let c_2 = Cons::new(cons_fp, cons_bo_fp, -2, true);
        let c_1 = Cons::new(cons_fn, cons_bo_fn, -1, true);
        let c0 = Cons::new(cons_p0, cons_bo_p0, 0, true);
        let c1 = Cons::new(cons_p1, cons_bo_p1, 1, true);

        // default
        features.push("default".to_owned());

        // unigrams
        features.push(c_2.cons_bo.clone());
        features.push(c_2.cons.clone());
        features.push(c_1.cons_bo.clone());
        features.push(c_1.cons.clone());
        features.push(c0.cons_bo.clone());
        features.push(c0.cons.clone());
        features.push(c1.cons_bo.clone());
        features.push(c1.cons.clone());

        // productions
        let prod = production(fn_node, false);
        let punct_prod = production(fn_node, true);
        features.push(format!("pn={prod}"));
        features.push(format!("pkt={punct_prod}"));

        // bigrams
        cons2(&mut features, &c_2, &c_1, punct_1s, true);
        cons2(&mut features, &c_1, &c0, punct1s, true);
        cons2(&mut features, &c0, &c1, punct_1_fs, true);

        // trigrams
        cons3(&mut features, &c_2, &c_1, &c0, punct_1s, punct1s, true, true, true);
        cons3(&mut features, &c_1, &c0, &c1, punct1s, punct_1_fs, true, true, true);

        // frontier distance
        let mut fdist = String::new();
        for node in &right_frontier[..rfi] {
            fdist.push_str(node.kind());
            fdist.push(',');
        }
        features.push(format!("fd={fdist}"));
        features.push(format!("sn={rfi}"));

        features
    }
}

/// Context generator for the tree-insertion check phase (`opennlp.tools.parser.treeinsert.CheckContextGenerator`).
#[derive(Clone, Debug)]
pub struct CheckContextGenerator {
    punct_set: HashSet<String>,
}

impl CheckContextGenerator {
    /// Creates a new [`CheckContextGenerator`] with `punct_set`.
    pub fn new(punct_set: HashSet<String>) -> Self {
        Self { punct_set }
    }

    /// Generates completeness check features for `parent` at `constituents[index]`
    /// (`CheckContextGenerator.getContext(Parse, Parse[], int, boolean)`).
    pub fn get_context(
        &self,
        parent: &Parse,
        constituents: &[Parse],
        index: usize,
        trim_frontier: bool,
    ) -> Vec<String> {
        let mut features = Vec::with_capacity(32);
        features.push("default".to_owned());

        let children = collapse_punctuation(parent.children(), &self.punct_set);
        let pstart = &children[0];
        let pend = &children[children.len() - 1];
        let kind = parent.kind();

        check_cons(pstart, "begin", kind, &mut features);
        check_cons(pend, "last", kind, &mut features);
        features.push(format!("p={}", production(parent, false)));
        features.push(format!("pp={}", production(parent, true)));

        for ci in 1..children.len() {
            check_cons_pair(&children[ci - 1], &children[ci], kind, &mut features);
        }

        let p_1 = Some(parent);
        let p_2 = index.checked_sub(1).and_then(|j| constituents.get(j));
        let p1 = constituents.get(index + 1);
        let p2 = constituents.get(index + 2);

        let punct1s = Some(constituents[index].next_punctuation());
        let punct2s = p1.map(Parse::next_punctuation);
        let punct_1s = Some(parent.previous_punctuation());
        let punct_2s = p_2.map(Parse::previous_punctuation);

        surround(p_1, -1, kind, punct_1s, &mut features);
        surround(p_2, -2, kind, punct_2s, &mut features);
        surround(p1, 1, kind, punct1s, &mut features);
        surround(p2, 2, kind, punct2s, &mut features);

        if trim_frontier {
            features.push("tf".to_owned());
        }

        features
    }
}

/// Tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.Parser`).
#[derive(Debug)]
pub struct Parser {
    model: Arc<ParserModel>,
    beam_size: usize,
    advance_percentage: f64,
    build_cg: BuildContextGenerator,
    attach_cg: AttachContextGenerator,
    check_cg: CheckContextGenerator,
    done_index: usize,
    sister_attach_index: usize,
    daughter_attach_index: usize,
    non_attach_index: usize,
    complete_index: usize,
    check_complete: bool,
}

impl Parser {
    /// Creates a tree-insertion [`Parser`] with default beam size (`20`) and advance percentage (`0.95`).
    pub fn new(model: impl Into<Arc<ParserModel>>) -> Self {
        Self::with_params(model, DEFAULT_BEAM_SIZE, DEFAULT_ADVANCE_PERCENTAGE)
    }

    /// Creates a tree-insertion [`Parser`] with explicit `beam_size` and `advance_percentage`.
    pub fn with_params(model: impl Into<Arc<ParserModel>>, beam_size: usize, advance_percentage: f64) -> Self {
        let model = model.into();
        let build_model = model.build_model();
        let attach_model = model.attach_model().expect("TREEINSERT model has attach.model");
        let check_model = model.check_model();
        let punct_set = model.head_rules().punctuation_tags().clone();

        let done_index = build_model.outcome_index(DONE).unwrap_or(0);
        let sister_attach_index = attach_model.outcome_index(ATTACH_SISTER).unwrap_or(0);
        let daughter_attach_index = attach_model.outcome_index(ATTACH_DAUGHTER).unwrap_or(0);
        let non_attach_index = attach_model.outcome_index(NON_ATTACH).unwrap_or(0);
        let complete_index = check_model.outcome_index(COMPLETE).unwrap_or(0);

        Self {
            model,
            beam_size,
            advance_percentage,
            build_cg: BuildContextGenerator::new(),
            attach_cg: AttachContextGenerator::new(),
            check_cg: CheckContextGenerator::new(punct_set),
            done_index,
            sister_attach_index,
            daughter_attach_index,
            non_attach_index,
            complete_index,
            check_complete: false,
        }
    }

    /// Returns the underlying [`ParserModel`].
    pub fn model(&self) -> &ParserModel {
        &self.model
    }

    /// Returns the right frontier of `root`, ordered from deepest right-most non-POS constituent
    /// to top-most (`treeinsert.Parser.getRightFrontier`).
    pub fn get_right_frontier(root: &Parse, punct_set: &HashSet<String>) -> Vec<Parse> {
        Self::get_right_frontier_with_paths(root, punct_set).into_iter().map(|(node, _)| node).collect()
    }

    pub(crate) fn get_right_frontier_with_paths(root: &Parse, punct_set: &HashSet<String>) -> Vec<(Parse, Vec<usize>)> {
        let mut rf: Vec<(Parse, Vec<usize>)> = Vec::new();
        let (mut cur, mut cur_path) = if root.kind() == TOP_NODE || root.kind() == INC_NODE {
            let children = root.children();
            let Some(first_idx) = (0..children.len()).find(|&i| !punct_set.contains(children[i].kind())) else {
                return rf;
            };
            (&children[first_idx], vec![first_idx])
        } else {
            (root, Vec::new())
        };

        while !cur.is_pos_tag() {
            rf.insert(0, (cur.clone(), cur_path.clone()));
            let kids = cur.children();
            if kids.is_empty() {
                break;
            }
            let mut ci = kids.len() - 1;
            if punct_set.contains(kids[ci].kind()) {
                while ci > 0 && punct_set.contains(kids[ci].kind()) {
                    ci -= 1;
                }
            }
            cur = &kids[ci];
            cur_path.push(ci);
        }
        rf
    }

    fn set_built(p: &mut Parse) {
        let next = match p.label() {
            None => BUILT.to_owned(),
            Some(l) => format!("{BUILT}.{l}"),
        };
        p.set_label(Some(next));
    }

    fn set_complete(p: &mut Parse) {
        let next = match p.label() {
            None => COMPLETE.to_owned(),
            Some(l) => format!("{l}.{COMPLETE}"),
        };
        p.set_label(Some(next));
    }

    fn set_incomplete(p: &mut Parse) {
        let next = match p.label() {
            None => INCOMPLETE.to_owned(),
            Some(l) => format!("{l}.{INCOMPLETE}"),
        };
        p.set_label(Some(next));
    }

    fn is_built(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.starts_with(BUILT))
    }

    fn is_complete(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.ends_with(COMPLETE))
    }

    fn advance_top(&self, p: &mut Parse) {
        p.set_kind(TOP_NODE);
    }

    fn advance_parses(&self, p: &mut Parse, prob_mass: f64) -> Vec<Parse> {
        let q = 1.0 - prob_mass;
        let punct_set = self.model.head_rules().punctuation_tags();
        let non_punct_indices = collapse_punctuation_in_place(p.children_mut(), punct_set);
        let children: Vec<Parse> = non_punct_indices.iter().map(|&idx| p.children()[idx].clone()).collect();
        let num_nodes = children.len();
        if num_nodes == 0 {
            return Vec::new();
        }

        let mut advance_node_index = 0usize;
        while advance_node_index < num_nodes {
            if !Self::is_built(&children[advance_node_index]) {
                break;
            }
            advance_node_index += 1;
        }
        if advance_node_index >= num_nodes {
            return Vec::new();
        }

        let original_zero_index = non_punct_indices[0];
        let original_advance_index = non_punct_indices[advance_node_index];
        let advance_node = children[advance_node_index].clone();

        let build_model = self.model.build_model();
        let attach_model = self.model.attach_model().expect("TREEINSERT has attach.model");
        let check_model = self.model.check_model();
        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut aprobs = vec![0.0; attach_model.num_outcomes()];
        let mut cprobs = vec![0.0; check_model.num_outcomes()];

        let mut new_parses_list: Vec<Parse> = Vec::new();

        // Call build model
        let b_ctx = self.build_cg.get_context(&children, advance_node_index);
        let b_refs: Vec<&str> = b_ctx.iter().map(String::as_str).collect();
        build_model.eval(&b_refs, &mut bprobs);
        let done_prob = bprobs[self.done_index];

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
            if max == self.done_index {
                let mut new_parse1 = p.clone();
                Self::set_built(&mut new_parse1.children_mut()[original_advance_index]);
                new_parse1.add_prob(strict_math::log(bprob));
                if advance_node_index == 0 {
                    new_parses_list.push(new_parse1);
                } else {
                    let rf_with_paths = Self::get_right_frontier_with_paths(p, punct_set);
                    let rf_nodes: Vec<Parse> = rf_with_paths.iter().map(|(n, _)| n.clone()).collect();
                    for (fi, (fn_node, fn_path)) in rf_with_paths.iter().enumerate() {
                        if self.check_complete && Self::is_complete(fn_node) {
                            continue;
                        }
                        let a_ctx = self.attach_cg.get_context(&children, advance_node_index, &rf_nodes, fi);
                        let a_refs: Vec<&str> = a_ctx.iter().map(String::as_str).collect();
                        attach_model.eval(&a_refs, &mut aprobs);

                        let attachments = if aprobs[self.daughter_attach_index] > aprobs[self.sister_attach_index] {
                            [self.daughter_attach_index, self.sister_attach_index]
                        } else {
                            [self.sister_attach_index, self.daughter_attach_index]
                        };

                        for &att_idx in &attachments {
                            let prob = aprobs[att_idx];
                            if prob > q
                                && ((!self.check_complete
                                    && (fi != rf_with_paths.len() - 1 || att_idx != self.sister_attach_index))
                                    || (self.check_complete && !Self::is_complete(fn_node)))
                            {
                                let mut new_parse2 = new_parse1.clone();
                                let adv_clone = new_parse2.children()[original_advance_index].clone();
                                let updated_path = if att_idx == self.sister_attach_index {
                                    if fi + 1 == rf_with_paths.len() {
                                        new_parse2.adjoin_root(adv_clone, self.model.head_rules(), original_zero_index);
                                        vec![original_zero_index]
                                    } else {
                                        let parent_path = &rf_with_paths[fi + 1].1;
                                        let parent = new_parse2.node_at_path_mut(parent_path);
                                        let _ = parent.adjoin(adv_clone, self.model.head_rules());
                                        let last_idx = parent.child_count() - 1;
                                        let mut up = parent_path.clone();
                                        up.push(last_idx);
                                        up
                                    }
                                } else {
                                    let target = new_parse2.node_at_path_mut(fn_path);
                                    let _ = target.add(adv_clone, self.model.head_rules());
                                    fn_path.clone()
                                };

                                new_parse2.remove(original_advance_index);
                                new_parse2.add_prob(strict_math::log(prob));
                                if advance_node_index == num_nodes - 1 {
                                    new_parse2.expand_top_node(original_zero_index);
                                }

                                if self.check_complete {
                                    let updated_node = new_parse2.node_at_path(&updated_path).clone();
                                    let c_ctx =
                                        self.check_cg.get_context(&updated_node, &children, advance_node_index, true);
                                    let c_refs: Vec<&str> = c_ctx.iter().map(String::as_str).collect();
                                    check_model.eval(&c_refs, &mut cprobs);
                                    if cprobs[self.complete_index] > prob_mass {
                                        let mut new_parse3 = new_parse2.clone();
                                        Self::set_complete(new_parse3.node_at_path_mut(&updated_path));
                                        new_parse3.add_prob(strict_math::log(cprobs[self.complete_index]));
                                        new_parses_list.push(new_parse3);
                                    }
                                    if 1.0 - cprobs[self.complete_index] > prob_mass {
                                        Self::set_incomplete(new_parse2.node_at_path_mut(&updated_path));
                                        new_parse2.add_prob(strict_math::log(1.0 - cprobs[self.complete_index]));
                                        new_parses_list.push(new_parse2);
                                    }
                                } else {
                                    new_parses_list.push(new_parse2);
                                }
                            }
                        }

                        if self.check_complete && !Self::is_complete(fn_node) {
                            break;
                        }
                    }
                }
            } else {
                let mut new_parse1 = p.clone();
                let mut new_node = Parse::new(Arc::clone(advance_node.text_arc()), advance_node.span(), tag, bprob, 0);
                let _ = new_node.add(advance_node.clone(), self.model.head_rules());
                new_node.set_previous_punctuation(advance_node.previous_punctuation().to_vec());
                new_node.set_next_punctuation(advance_node.next_punctuation().to_vec());
                new_parse1.insert(new_node.clone()).expect("new_node span equals advance_node span");
                new_parse1.add_prob(strict_math::log(bprob));

                if self.check_complete {
                    let c_ctx = self.check_cg.get_context(&new_node, &children, advance_node_index, false);
                    let c_refs: Vec<&str> = c_ctx.iter().map(String::as_str).collect();
                    check_model.eval(&c_refs, &mut cprobs);
                    if cprobs[self.complete_index] > prob_mass {
                        let mut new_parse2 = new_parse1.clone();
                        Self::set_complete(&mut new_parse2.children_mut()[original_advance_index]);
                        new_parse2.add_prob(strict_math::log(cprobs[self.complete_index]));
                        new_parses_list.push(new_parse2);
                    }
                    if 1.0 - cprobs[self.complete_index] > prob_mass {
                        Self::set_incomplete(&mut new_parse1.children_mut()[original_advance_index]);
                        new_parse1.add_prob(strict_math::log(1.0 - cprobs[self.complete_index]));
                        new_parses_list.push(new_parse1);
                    }
                } else {
                    new_parses_list.push(new_parse1);
                }
            }
        }

        // Advance exclusions (`done_index` when `done_prob == 0.0`)
        if advance_node_index != 0 && done_prob == 0.0 {
            let new_parse1 = p.clone();
            let rf_with_paths = Self::get_right_frontier_with_paths(&new_parse1, punct_set);
            let rf_nodes: Vec<Parse> = rf_with_paths.iter().map(|(n, _)| n.clone()).collect();
            for (fi, (fn_node, _)) in rf_with_paths.iter().enumerate() {
                if self.check_complete && Self::is_complete(fn_node) {
                    continue;
                }
                let a_ctx = self.attach_cg.get_context(&children, advance_node_index, &rf_nodes, fi);
                let a_refs: Vec<&str> = a_ctx.iter().map(String::as_str).collect();
                attach_model.eval(&a_refs, &mut aprobs);
                if aprobs[self.non_attach_index] > q {
                    let mut new_parse2 = new_parse1.clone();
                    for (_, path) in &rf_with_paths[..=fi] {
                        Self::set_complete(new_parse2.node_at_path_mut(path));
                    }
                    new_parse2.add_prob(strict_math::log(aprobs[self.non_attach_index]));
                    new_parses_list.push(new_parse2);
                }
                if self.check_complete && !Self::is_complete(fn_node) {
                    break;
                }
            }
        }

        new_parses_list
    }

    /// Trains a `TREEINSERT` [`ParserModel`] from `samples` (`treeinsert.Parser.train`).
    pub fn train(
        language: &str,
        samples: &[Parse],
        rules: HeadRulesArtifact,
        params: &ParserTrainingParameters,
    ) -> Result<ParserModel, ParserModelError> {
        let dict = build_dictionary(samples, &rules, params.dict_cutoff);

        // 1. Train build model
        let build_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Build, Some(&dict));
        let (build_trained, _) =
            trainer::train_events(&build_events, &params.build, &mut TrainingConfiguration::default())
                .map_err(turbo_train_components::error::Error::Train)?;
        let build_bytes = build_trained.write().map_err(turbo_train_components::error::Error::Train)?;

        // 2. Train check model
        let check_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Check, None);
        let (check_trained, _) =
            trainer::train_events(&check_events, &params.check, &mut TrainingConfiguration::default())
                .map_err(turbo_train_components::error::Error::Train)?;
        let check_bytes = check_trained.write().map_err(turbo_train_components::error::Error::Train)?;

        // 3. Train attach model
        let attach_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Attach, None);
        let (attach_trained, _) =
            trainer::train_events(&attach_events, &params.attach, &mut TrainingConfiguration::default())
                .map_err(turbo_train_components::error::Error::Train)?;
        let attach_bytes = attach_trained.write().map_err(turbo_train_components::error::Error::Train)?;

        // 4. Train POS tagger sub-model
        let pos_samples = collect_pos_samples(samples);
        let pos_factory = PosTaggerFactory::new(language, None, None)?;
        let mut tagger_params = params.tagger.clone();
        if tagger_params.get_int(BEAM_SIZE_KEY).ok().flatten().is_none() {
            tagger_params.put(BEAM_SIZE_KEY, 10);
        }
        let trained_pos = train_pos_model(
            &pos_samples,
            &pos_factory,
            &tagger_params,
            &mut TrainingConfiguration::default(),
            Stamp::at(0),
        )?;
        let tagger_bytes = trained_pos.package.to_bytes();

        // 5. Train chunker sub-model with ParserChunkerFactory
        let chunker_bytes = train_parser_chunker_bytes(language, samples, &params.chunker)?;

        ParserModel::from_serialized_parts(
            language,
            build_bytes,
            check_bytes,
            Some(attach_bytes),
            tagger_bytes,
            chunker_bytes,
            rules,
            ParserType::TreeInsert,
            &[],
            Stamp::at(0),
        )
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, crate::parse::ParseError> {
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| Ok(Some(self.advance_parses(p, q))),
            |p| self.advance_top(p),
            |_| {},
        )
    }
}

/// Event stream for the tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.ParserEventStream`).
pub struct ParserEventStream;

impl ParserEventStream {
    /// Generates training [`Event`]s of type `etype` over `samples`.
    pub fn generate_events(
        samples: &[Parse],
        rules: &dyn HeadRules,
        etype: ParserEventType,
        _dict: Option<&HashSet<String>>,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let build_cg = BuildContextGenerator::new();
        let attach_cg = AttachContextGenerator::new();
        let punct_set = rules.punctuation_tags();
        let check_cg = CheckContextGenerator::new(punct_set.clone());
        let mut chunk_cg = ChunkContextGenerator::new();

        for orig in samples {
            let mut p = orig.clone();
            let _ = Parse::prune_parse(&mut p);
            p.update_heads(rules);

            match etype {
                ParserEventType::Tag => {
                    let pos = ParserPosSample::from_parse(&p);
                    if let Ok(factory) = PosTaggerFactory::new("en", None, None)
                        && let Ok(mut stream) = turbo_train_components::postag::PosEventStream::new(&factory)
                    {
                        let toks: Vec<Field<'_>> =
                            pos.sentence.iter().map(|s| Field::of(s.as_str(), Span::new(0, 0))).collect();
                        let tags: Vec<Field<'_>> =
                            pos.tags.iter().map(|s| Field::of(s.as_str(), Span::new(0, 0))).collect();
                        if let Ok(sample) = PosSample::new(toks, tags, Span::new(0, 0)) {
                            stream.events(&sample, &mut events);
                        }
                    }
                }
                ParserEventType::Chunk => {
                    let cs = ParserChunkSample::from_parse(&p);
                    let words: Vec<&str> = cs.sentence.iter().map(String::as_str).collect();
                    let tags: Vec<&str> = cs.tags.iter().map(String::as_str).collect();
                    let preds: Vec<&str> = cs.preds.iter().map(String::as_str).collect();
                    for i in 0..words.len() {
                        let ctx = chunk_cg.get_context(i, &words, &tags, &preds);
                        events.push(event_from_strings(preds[i], &ctx));
                    }
                }
                ParserEventType::Build | ParserEventType::Check | ParserEventType::Attach => {
                    Self::add_parse_events(&mut events, &mut p, rules, etype, &build_cg, &attach_cg, &check_cg);
                }
            }
        }
        events
    }

    fn add_parse_events(
        events: &mut Vec<Event>,
        p: &mut Parse,
        rules: &dyn HeadRules,
        etype: ParserEventType,
        build_cg: &BuildContextGenerator,
        attach_cg: &AttachContextGenerator,
        check_cg: &CheckContextGenerator,
    ) {
        let punct_set = rules.punctuation_tags();
        let init_paths = p.initial_chunk_paths();
        let mut init_nodes: Vec<Parse> = init_paths.iter().map(|path| p.node_at_path(path).clone()).collect();
        let kept = collapse_punctuation_in_place(&mut init_nodes, punct_set);

        let mut chunk_paths: Vec<Vec<usize>> = Vec::with_capacity(kept.len());
        let mut current_chunks: Vec<Parse> = Vec::with_capacity(kept.len());
        for idx in kept {
            let path = init_paths[idx].clone();
            let mut node = init_nodes[idx].clone();
            node.set_label(Some(COMPLETE.to_owned()));
            let target = p.node_at_path_mut(&path);
            target.set_previous_punctuation(node.previous_punctuation().to_vec());
            target.set_next_punctuation(node.next_punctuation().to_vec());
            target.set_label(Some(COMPLETE.to_owned()));
            chunk_paths.push(path);
            current_chunks.push(node);
        }

        let mut right_frontier: Vec<Vec<usize>> = Vec::new();
        let mut built_nodes: Vec<Vec<usize>> = Vec::new();
        let check_complete = false;

        for ci in 0..chunk_paths.len() {
            let mut cur_gold_path = chunk_paths[ci].clone();
            if cur_gold_path.is_empty() {
                continue;
            }
            let mut parent_path = cur_gold_path[..cur_gold_path.len() - 1].to_vec();
            let mut prev_parent_path = cur_gold_path.clone();
            let mut off = 0usize;

            if !p.node_at_path(&cur_gold_path).is_pos_tag() {
                built_nodes.insert(off, cur_gold_path.clone());
                off += 1;
            }

            // Perform build stages
            while p.node_at_path(&parent_path).kind() != TOP_NODE && p.node_at_path(&parent_path).label().is_none() {
                let parent_type = p.node_at_path(&parent_path).kind().to_owned();
                let prev_type = p.node_at_path(&prev_parent_path).kind().to_owned();
                if prev_type != parent_type {
                    if etype == ParserEventType::Build {
                        let ctx = build_cg.get_context(&current_chunks, ci);
                        events.push(event_from_strings(&parent_type, &ctx));
                    }
                    built_nodes.insert(off, parent_path.clone());
                    off += 1;

                    let cur_cc = current_chunks[ci].clone();
                    let mut new_parent = Parse::new(Arc::clone(cur_cc.text_arc()), cur_cc.span(), &parent_type, 1.0, 0);
                    let _ = new_parent.add(cur_cc.clone(), rules);
                    new_parent.set_previous_punctuation(cur_cc.previous_punctuation().to_vec());
                    new_parent.set_next_punctuation(cur_cc.next_punctuation().to_vec());
                    new_parent.set_label(Some(BUILT.to_owned()));
                    current_chunks[ci] = new_parent;

                    if Self::last_child_in_gold(p, &cur_gold_path, &parent_path, punct_set) {
                        if etype == ParserEventType::Check {
                            let ctx = check_cg.get_context(&current_chunks[ci], &current_chunks, ci, false);
                            events.push(event_from_strings(COMPLETE, &ctx));
                        }
                        current_chunks[ci].set_label(Some(COMPLETE.to_owned()));
                        p.node_at_path_mut(&parent_path).set_label(Some(COMPLETE.to_owned()));
                    } else {
                        if etype == ParserEventType::Check {
                            let ctx = check_cg.get_context(&current_chunks[ci], &current_chunks, ci, false);
                            events.push(event_from_strings(INCOMPLETE, &ctx));
                        }
                        current_chunks[ci].set_label(Some(INCOMPLETE.to_owned()));
                        p.node_at_path_mut(&parent_path).set_label(Some(COMPLETE.to_owned()));
                    }
                    chunk_paths[ci] = parent_path.clone();
                    cur_gold_path = parent_path.clone();
                }
                p.node_at_path_mut(&parent_path).set_label(Some(BUILT.to_owned()));
                prev_parent_path = parent_path.clone();
                if parent_path.is_empty() {
                    break;
                }
                parent_path.pop();
            }

            // Decide to attach
            if etype == ParserEventType::Build {
                let ctx = build_cg.get_context(&current_chunks, ci);
                events.push(event_from_strings(DONE, &ctx));
            }

            if ci > 0 {
                let crf_with_paths = Parser::get_right_frontier_with_paths(&current_chunks[0], punct_set);
                let crf_nodes: Vec<Parse> = crf_with_paths.iter().map(|(n, _)| n.clone()).collect();
                if crf_with_paths.len() != right_frontier.len() {
                    break;
                }

                let parents = Self::get_non_adjoined_parent(p, &chunk_paths[ci], punct_set);
                let mut attach_type: Option<&'static str> = None;
                let mut attach_node_index: Option<usize> = None;

                // Try daughters first
                for (cfi, (cfn, _)) in crf_with_paths.iter().enumerate() {
                    let frontier_path = &right_frontier[cfi];
                    if !check_complete || cfn.label() != Some(COMPLETE) {
                        let non_punct_count = collapse_punctuation(cfn.children(), punct_set).len();
                        if attach_node_index.is_none() && parents.get(frontier_path).copied() == Some(non_punct_count) {
                            attach_type = Some(ATTACH_DAUGHTER);
                            attach_node_index = Some(cfi);
                            if etype == ParserEventType::Attach {
                                let ctx = attach_cg.get_context(&current_chunks, ci, &crf_nodes, cfi);
                                events.push(event_from_strings(ATTACH_DAUGHTER, &ctx));
                            }
                        }
                    }
                    if check_complete && cfn.label() == Some(INCOMPLETE) {
                        break;
                    }
                }

                // Try sisters and generate non-attach events
                for (cfi, (cfn, _)) in crf_with_paths.iter().enumerate() {
                    let frontier_path = &right_frontier[cfi];
                    let frontier_parent =
                        (!frontier_path.is_empty()).then(|| frontier_path[..frontier_path.len() - 1].to_vec());
                    let can_sister = if let Some(fp) = &frontier_parent {
                        parents.contains_key(fp) && p.node_at_path(frontier_path).kind() == p.node_at_path(fp).kind()
                    } else {
                        false
                    };
                    if attach_node_index.is_none() && can_sister {
                        attach_type = Some(ATTACH_SISTER);
                        attach_node_index = Some(cfi);
                        if etype == ParserEventType::Attach {
                            let ctx = attach_cg.get_context(&current_chunks, ci, &crf_nodes, cfi);
                            events.push(event_from_strings(ATTACH_SISTER, &ctx));
                        }
                        let cp = &chunk_paths[ci];
                        if !cp.is_empty() {
                            let cp_parent = &cp[..cp.len() - 1];
                            p.node_at_path_mut(cp_parent).set_label(Some(BUILT.to_owned()));
                        }
                    } else if Some(cfi) == attach_node_index {
                        // Skip over previously attached daughter
                    } else if etype == ParserEventType::Attach {
                        let ctx = attach_cg.get_context(&current_chunks, ci, &crf_nodes, cfi);
                        events.push(event_from_strings(NON_ATTACH, &ctx));
                    }
                    if check_complete && cfn.label() == Some(INCOMPLETE) {
                        break;
                    }
                }

                // Perform attachment on current_chunks[0]
                if let (Some(att_kind), Some(att_idx)) = (attach_type, attach_node_index) {
                    let cfn_path = &crf_with_paths[att_idx].1;
                    if att_kind == ATTACH_DAUGHTER {
                        let daughter = current_chunks[ci].clone();
                        let target = current_chunks[0].node_at_path_mut(cfn_path);
                        let _ = target.add(daughter, rules);
                        let is_last =
                            Self::last_child_in_gold(p, &chunk_paths[ci], &right_frontier[att_idx], punct_set);
                        let updated_target = current_chunks[0].node_at_path(cfn_path).clone();
                        if is_last {
                            if etype == ParserEventType::Check {
                                let ctx = check_cg.get_context(&updated_target, &current_chunks, ci, true);
                                events.push(event_from_strings(COMPLETE, &ctx));
                            }
                            current_chunks[0].node_at_path_mut(cfn_path).set_label(Some(COMPLETE.to_owned()));
                        } else if etype == ParserEventType::Check {
                            let ctx = check_cg.get_context(&updated_target, &current_chunks, ci, true);
                            events.push(event_from_strings(INCOMPLETE, &ctx));
                        }
                    } else if att_kind == ATTACH_SISTER {
                        let frontier_path = right_frontier[att_idx].clone();
                        let frontier_parent = frontier_path[..frontier_path.len() - 1].to_vec();
                        right_frontier[att_idx] = frontier_parent;
                        let sister = current_chunks[ci].clone();

                        let updated_path = if cfn_path.is_empty() {
                            let mut wrapper = Parse::new(
                                Arc::clone(current_chunks[0].text_arc()),
                                Span::new(0, current_chunks[0].text().len() as u32),
                                TOP_NODE,
                                1.0,
                                0,
                            );
                            wrapper.insert(current_chunks[0].clone()).expect("root inside wrapper");
                            wrapper.adjoin_root(sister, rules, 0);
                            current_chunks[0] = wrapper.children()[0].clone();
                            Vec::new()
                        } else {
                            let parent_path = &cfn_path[..cfn_path.len() - 1];
                            let parent = current_chunks[0].node_at_path_mut(parent_path);
                            let _ = parent.adjoin(sister, rules);
                            let last_idx = parent.child_count() - 1;
                            let mut up = parent_path.to_vec();
                            up.push(last_idx);
                            up
                        };

                        let is_last =
                            Self::last_child_in_gold(p, &chunk_paths[ci], &right_frontier[att_idx], punct_set);
                        let new_parent = current_chunks[0].node_at_path(&updated_path).clone();
                        if is_last {
                            if etype == ParserEventType::Check {
                                let ctx = check_cg.get_context(&new_parent, &current_chunks, ci, true);
                                events.push(event_from_strings(COMPLETE, &ctx));
                            }
                            current_chunks[0].node_at_path_mut(&updated_path).set_label(Some(COMPLETE.to_owned()));
                        } else {
                            if etype == ParserEventType::Check {
                                let ctx = check_cg.get_context(&new_parent, &current_chunks, ci, true);
                                events.push(event_from_strings(INCOMPLETE, &ctx));
                            }
                            current_chunks[0].node_at_path_mut(&updated_path).set_label(Some(INCOMPLETE.to_owned()));
                        }
                    }

                    right_frontier.drain(0..att_idx);
                }
            }

            built_nodes.append(&mut right_frontier);
            std::mem::swap(&mut right_frontier, &mut built_nodes);
            built_nodes.clear();
        }
    }

    fn get_non_adjoined_parent(
        p: &Parse,
        node_path: &[usize],
        punct_set: &HashSet<String>,
    ) -> HashMap<Vec<usize>, usize> {
        let mut parents = HashMap::new();
        let mut cur = node_path.to_vec();
        if cur.is_empty() {
            return parents;
        }
        let mut parent = cur[..cur.len() - 1].to_vec();
        let idx = Self::non_punct_index_of(p, &cur, &parent, punct_set);
        parents.insert(parent.clone(), idx);
        while p.node_at_path(&parent).kind() == p.node_at_path(&cur).kind() && !parent.is_empty() {
            cur = parent;
            parent = cur[..cur.len() - 1].to_vec();
            let idx = Self::non_punct_index_of(p, &cur, &parent, punct_set);
            parents.insert(parent.clone(), idx);
        }
        parents
    }

    fn non_punct_index_of(
        p: &Parse,
        child_path: &[usize],
        parent_path: &[usize],
        punct_set: &HashSet<String>,
    ) -> usize {
        let child_idx = child_path[child_path.len() - 1];
        let parent = p.node_at_path(parent_path);
        parent.children()[..child_idx].iter().filter(|k| !punct_set.contains(k.kind())).count()
    }

    fn direct_last_child(p: &Parse, child_path: &[usize], parent_path: &[usize], punct_set: &HashSet<String>) -> bool {
        if child_path.len() != parent_path.len() + 1 || &child_path[..parent_path.len()] != parent_path {
            return false;
        }
        let child_idx = child_path[child_path.len() - 1];
        let parent = p.node_at_path(parent_path);
        parent.children()[child_idx + 1..].iter().all(|k| punct_set.contains(k.kind()))
    }

    fn last_child_in_gold(p: &Parse, child_path: &[usize], parent_path: &[usize], punct_set: &HashSet<String>) -> bool {
        let mut cur = child_path.to_vec();
        let mut lc = Self::direct_last_child(p, &cur, parent_path, punct_set);
        while !lc && !cur.is_empty() {
            let cp = cur[..cur.len() - 1].to_vec();
            if cp != parent_path && p.node_at_path(&cp).kind() == p.node_at_path(&cur).kind() {
                lc = Self::direct_last_child(p, &cp, parent_path, punct_set);
                cur = cp;
            } else {
                break;
            }
        }
        lc
    }
}
