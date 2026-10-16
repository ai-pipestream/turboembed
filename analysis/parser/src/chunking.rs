//! Bottom-up shift-reduce chunking constituency parser, context generators, event stream, and
//! trainer (`opennlp.tools.parser.chunking.*`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_formats::{Field, PosSample};
use turbo_jvm::strict_math;
use turbo_sequence::chunker::DEFAULT_BEAM_SIZE as CHUNKER_DEFAULT_BEAM_SIZE;
use turbo_train::trainer;
use turbo_train::{Event, JavaString, ModelPackage, TrainingConfiguration, TrainingParameters};
use turbo_train_components::package::{FACTORY_KEY, Stamp, add_artifacts};
use turbo_train_components::postag::{BEAM_SIZE_KEY, PosTaggerFactory, train_pos_model};

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, build_dictionary,
    collapse_punctuation_in_place, dictionary_key, parse_bottom_up,
};
use crate::context::{
    ChunkContextGenerator, Cons, check_cons, check_cons_pair, cons_bo_labeled, cons_labeled, cons2, cons3, surround,
};
use crate::head_rules::HeadRules;
use crate::model::{
    HeadRulesArtifact, PARSER_CHUNKER_FACTORY, ParserEventType, ParserModel, ParserModelError, ParserType,
};
use crate::parse::{COMPLETE, CONT, INCOMPLETE, Parse, ParseError, START, TOP_NODE};
use crate::stream::{ParserChunkSample, ParserPosSample};

/// Top-start outcome (`AbstractBottomUpParser.TOP_START = "S-TOP"`).
pub const TOP_START: &str = "S-TOP";

/// Context generator for the shift-reduce build phase (`opennlp.tools.parser.chunking.BuildContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct BuildContextGenerator {
    dict: Option<HashSet<String>>,
}

impl BuildContextGenerator {
    /// Creates a [`BuildContextGenerator`] without a dictionary, so every lexical feature is
    /// used (`new BuildContextGenerator()`).
    pub fn new() -> Self {
        Self { dict: None }
    }

    /// Creates a [`BuildContextGenerator`] whose lexical features are limited to the n-grams of
    /// `dict`, keyed by [`dictionary_key`] (`new BuildContextGenerator(Dictionary)`).
    pub fn with_dictionary(dict: HashSet<String>) -> Self {
        Self { dict: Some(dict) }
    }

    /// Generates build features for `constituents[index]` (`BuildContextGenerator.getContext(Parse[], int)`).
    pub fn get_context(&self, constituents: &[Parse], index: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
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
            let has = |ngram: &[&Parse]| {
                let words: Vec<&str> = ngram.iter().map(|p| p.head_text()).collect();
                dict.contains(&dictionary_key(&words))
            };
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

        let c_2 = Cons::new(cons_labeled(p_2, -2), cons_bo_labeled(p_2, -2), -2, u_2);
        let c_1 = Cons::new(cons_labeled(p_1, -1), cons_bo_labeled(p_1, -1), -1, u_1);
        let c0 = Cons::new(cons_labeled(Some(p0), 0), cons_bo_labeled(Some(p0), 0), 0, u0);
        let c1 = Cons::new(cons_labeled(p1, 1), cons_bo_labeled(p1, 1), 1, u1);
        let c2 = Cons::new(cons_labeled(p2, 2), cons_bo_labeled(p2, 2), 2, u2);

        features.push("default".to_owned());
        for c in [&c0, &c_2, &c_1, &c1, &c2] {
            if c.unigram {
                features.push(c.cons.clone());
            }
            features.push(c.cons_bo.clone());
        }

        cons2(&mut features, &c0, &c1, punct1s, b01);
        cons2(&mut features, &c_1, &c0, punct_1s, b_10);
        cons3(&mut features, &c0, &c1, &c2, punct1s, punct2s, t012, b01, b12);
        cons3(&mut features, &c_2, &c_1, &c0, punct_2s, punct_1s, t_2_10, b_2_1, b_10);
        cons3(&mut features, &c_1, &c0, &c1, punct_1s, punct1s, t_101, b_10, b01);

        // Walks back to the closest start; a node before `index` always has a label here, where
        // OpenNLP would throw `NullPointerException` on one without.
        let starts_constituent = |p: &Parse| p.label().is_some_and(|l| l.starts_with(START));
        let matches_back = |open: &str, feature: &str, features: &mut Vec<String>| {
            for p in constituents[..index].iter().rev() {
                if p.kind() == open {
                    features.push(feature.to_owned());
                    break;
                }
                if starts_constituent(p) {
                    break;
                }
            }
        };
        match p0.kind() {
            "-RRB-" => matches_back("-LRB-", "bracketsmatch", &mut features),
            "-RCB-" => matches_back("-LCB-", "bracketsmatch", &mut features),
            "''" => matches_back("``", "quotesmatch", &mut features),
            "'" => matches_back("`", "quotesmatch", &mut features),
            "," => matches_back(",", "iscomma", &mut features),
            "." if index + 1 == ps => {
                if let Some(pi) = constituents[..index].iter().rposition(starts_constituent)
                    && pi == 0
                {
                    features.push("endofsentence".to_owned());
                }
            }
            _ => {}
        }
        features
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
        let ps = constituents.len();
        let mut features = Vec::with_capacity(100);

        features.push("default".to_owned());
        features.push(format!("fl={}", constituents[0].label().unwrap_or("null")));
        let pstart = &constituents[start];
        let pend = &constituents[end];
        check_cons(pstart, "begin", kind, &mut features);
        check_cons(pend, "last", kind, &mut features);
        let mut production = format!("p={kind}->");
        let mut punct_production = format!("pp={kind}->");
        for p in &constituents[start..end] {
            check_cons_pair(p, pend, kind, &mut features);
            production.push_str(p.kind());
            production.push(',');
            punct_production.push_str(p.kind());
            punct_production.push(',');
            for punct in p.next_punctuation() {
                punct_production.push_str(punct.kind());
                punct_production.push(',');
            }
        }
        production.push_str(pend.kind());
        punct_production.push_str(pend.kind());
        features.push(production);
        features.push(punct_production);

        let p1s = Some(constituents[end].next_punctuation());
        let p_1s = Some(constituents[start].previous_punctuation());
        let p_2 = start.checked_sub(2).map(|j| &constituents[j]);
        let p_1 = start.checked_sub(1).map(|j| &constituents[j]);
        let p_2s = p_1.map(Parse::previous_punctuation);
        let p1 = (end + 1 < ps).then(|| &constituents[end + 1]);
        let p2s = p1.map(Parse::next_punctuation);
        let p2 = (end + 2 < ps).then(|| &constituents[end + 2]);

        surround(p_1, -1, kind, p_1s, &mut features);
        surround(p_2, -2, kind, p_2s, &mut features);
        surround(p1, 1, kind, p1s, &mut features);
        surround(p2, 2, kind, p2s, &mut features);

        features
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

    fn advance_top(&self, p: &mut Parse) {
        let build_model = self.model.build_model();
        let check_model = self.model.check_model();
        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut cprobs = vec![0.0; check_model.num_outcomes()];

        let b_ctx = self.build_cg.get_context(p.children(), 0);
        let b_refs: Vec<&str> = b_ctx.iter().map(String::as_str).collect();
        build_model.eval(&b_refs, &mut bprobs);
        if let Some(tsi) = self.top_start_index {
            p.add_prob(strict_math::log(bprobs[tsi]));
        }

        let c_ctx = self.check_cg.get_context(p.children(), TOP_NODE, 0, 0);
        let c_refs: Vec<&str> = c_ctx.iter().map(String::as_str).collect();
        check_model.eval(&c_refs, &mut cprobs);
        p.add_prob(strict_math::log(cprobs[self.complete_index]));
        p.set_kind(TOP_NODE);
    }

    /// `chunking.Parser.advanceParses`. `None` is OpenNLP's `null`, for a parse without
    /// non-punctuation children. Where every child is labeled already OpenNLP indexes past the
    /// children and throws; this returns no derivations instead.
    fn advance_parses(&self, p: &mut Parse, prob_mass: f64) -> Option<Vec<Parse>> {
        let q = 1.0 - prob_mass;
        let punct_set = self.model.head_rules().punctuation_tags();
        let non_punct_indices = collapse_punctuation_in_place(p.children_mut(), punct_set);
        let children: Vec<Parse> = non_punct_indices.iter().map(|&idx| p.children()[idx].clone()).collect();
        let num_nodes = children.len();
        if num_nodes == 0 {
            return None;
        }

        let mut last_start_node: Option<Parse> = None;
        let mut last_start_index: Option<usize> = None;
        let mut last_start_type: Option<String> = None;
        let mut advance_node_index = num_nodes;

        for (pi, np) in children.iter().enumerate() {
            if let Some(lbl) = np.label() {
                if let Some(st) = self.start_type_map.get(lbl) {
                    last_start_type = Some(st.clone());
                    last_start_node = Some(np.clone());
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
        let advance_node = children[advance_node_index].clone();
        let original_advance_index = non_punct_indices[advance_node_index];

        let build_model = self.model.build_model();
        let check_model = self.model.check_model();
        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut cprobs = vec![0.0; check_model.num_outcomes()];

        let b_ctx = self.build_cg.get_context(&children, advance_node_index);
        let b_refs: Vec<&str> = b_ctx.iter().map(String::as_str).collect();
        build_model.eval(&b_refs, &mut bprobs);

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
                last_start_node = Some(advance_node.clone());
                last_start_type = Some(st.clone());
            } else if let Some(ct) = self.cont_type_map.get(tag)
                && (last_start_node.is_none() || last_start_type.as_deref() != Some(ct.as_str()))
            {
                continue;
            }

            let mut new_parse1 = p.clone();
            new_parse1.set_child(original_advance_index, tag);
            new_parse1.add_prob(strict_math::log(bprob));

            if let (Some(ls_idx), Some(ls_type), Some(ls_node)) =
                (last_start_index, last_start_type.as_deref(), last_start_node.as_ref())
            {
                let np1_kept = collapse_punctuation_in_place(new_parse1.children_mut(), punct_set);
                let np1_children: Vec<Parse> = np1_kept.iter().map(|&idx| new_parse1.children()[idx].clone()).collect();
                let c_ctx = self.check_cg.get_context(&np1_children, ls_type, ls_idx, advance_node_index);
                let c_refs: Vec<&str> = c_ctx.iter().map(String::as_str).collect();
                check_model.eval(&c_refs, &mut cprobs);

                if cprobs[self.complete_index] > q {
                    let mut new_parse2 = new_parse1.clone();
                    new_parse2.add_prob(strict_math::log(cprobs[self.complete_index]));

                    let len = advance_node_index - ls_idx + 1;
                    let mut cons = Vec::with_capacity(len);
                    let mut flat = ls_node.is_pos_tag();
                    cons.push(ls_node.clone());
                    for ci in 1..(len - 1) {
                        let mid = children[ci + ls_idx].clone();
                        flat &= mid.is_pos_tag();
                        cons.push(mid);
                    }
                    if len > 1 {
                        flat &= advance_node.is_pos_tag();
                        cons.push(advance_node.clone());
                    }

                    if !flat {
                        let head = self.model.head_rules().get_head(&cons, ls_type);
                        let node_prob = cprobs.get(1).copied().unwrap_or(cprobs[self.complete_index]);
                        let span = if ls_idx == 0 && advance_node_index == num_nodes - 1 {
                            p.span()
                        } else {
                            Span::new(ls_node.span().start, advance_node.span().end)
                        };
                        new_parse2
                            .insert(Parse::with_head(Arc::clone(p.text_arc()), span, ls_type, node_prob, head))
                            .expect("reduced constituent span inside root");
                        new_parses_list.push(new_parse2);
                    }
                }
                if cprobs[self.incomplete_index] > q && advance_node_index != num_nodes - 1 {
                    new_parse1.add_prob(strict_math::log(cprobs[self.incomplete_index]));
                    new_parses_list.push(new_parse1);
                }
            } else {
                // When tag is `O` and no prior start exists, OpenNLP calls checkContextGenerator
                // with `lastStartIndex = -1`, which would throw ArrayIndexOutOfBoundsException in
                // Java unless a start outcome was already seen or tag is in startTypeMap.
            }
        }

        Some(new_parses_list)
    }

    /// Trains a `CHUNKING` [`ParserModel`] from `samples` (`chunking.Parser.train`).
    pub fn train(
        language: &str,
        samples: &[Parse],
        rules: HeadRulesArtifact,
        params: &ParserTrainingParameters,
    ) -> Result<ParserModel, ParserModelError> {
        let dict = build_dictionary(samples, &rules, params.dict_cutoff);

        let mut report = TrainingReports::default();

        // 1. Train build model
        let build_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Build, Some(&dict))?;
        let build_bytes = report.train("build", &build_events, &params.build)?;

        // 2. Train POS tagger sub-model
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

        // 3. Train chunker sub-model with ParserChunkerFactory
        let chunker_bytes = train_parser_chunker_bytes(language, samples, &params.chunker)?;

        // 4. Train check model
        let check_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Check, None)?;
        let check_bytes = report.train("check", &check_events, &params.check)?;

        ParserModel::from_serialized_parts(
            language,
            build_bytes,
            check_bytes,
            None,
            tagger_bytes,
            chunker_bytes,
            rules,
            ParserType::Chunking,
            &report.manifest_entries(),
            Stamp::at(0),
        )
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| Ok(self.advance_parses(p, q)),
            |p| self.advance_top(p),
            |_| {},
        )
    }
}

/// Event stream for the shift-reduce chunking parser (`opennlp.tools.parser.chunking.ParserEventStream`).
pub struct ParserEventStream;

impl ParserEventStream {
    /// Generates training [`Event`]s of type `etype` over `samples` (`AbstractParserEventStream`).
    ///
    /// A sample `Parse::prune_parse` refuses is an error, where OpenNLP's stream throws.
    pub fn generate_events(
        samples: &[Parse],
        rules: &dyn HeadRules,
        etype: ParserEventType,
        dict: Option<&HashSet<String>>,
    ) -> Result<Vec<Event>, ParseError> {
        let mut events = Vec::new();
        let bcg = BuildContextGenerator::with_dictionary(dict.cloned().unwrap_or_default());
        let bcg_no_dict = BuildContextGenerator::new();
        let build_cg = if dict.is_some() { &bcg } else { &bcg_no_dict };
        let check_cg = CheckContextGenerator::new();
        let mut chunk_cg = ChunkContextGenerator::new();
        let punct_set = rules.punctuation_tags();

        for orig in samples {
            let mut p = orig.clone();
            Parse::prune_parse(&mut p)?;
            if rules_fix_possessives() {
                Parse::fix_possessives(&mut p);
            }
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
                    Self::add_parse_events(&mut events, &mut p, punct_set, etype, build_cg, &check_cg);
                }
            }
        }
        Ok(events)
    }

    fn add_parse_events(
        events: &mut Vec<Event>,
        p: &mut Parse,
        punct_set: &HashSet<String>,
        etype: ParserEventType,
        bcg: &BuildContextGenerator,
        kcg: &CheckContextGenerator,
    ) {
        let init_paths = p.initial_chunk_paths();
        let mut init_nodes: Vec<Parse> = init_paths.iter().map(|path| p.node_at_path(path).clone()).collect();
        let kept = collapse_punctuation_in_place(&mut init_nodes, punct_set);

        let mut chunks: Vec<Parse> = Vec::with_capacity(kept.len());
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(kept.len());
        for idx in kept {
            let path = init_paths[idx].clone();
            let node = init_nodes[idx].clone();
            let target = p.node_at_path_mut(&path);
            target.set_previous_punctuation(node.previous_punctuation().to_vec());
            target.set_next_punctuation(node.next_punctuation().to_vec());
            chunks.push(node);
            paths.push(path);
        }

        let mut ci = 0usize;
        while ci < chunks.len() {
            let path = paths[ci].clone();
            if !path.is_empty() {
                let parent_path = &path[..path.len() - 1];
                let child_idx = path[path.len() - 1];
                let parent_node = p.node_at_path(parent_path);
                let parent_type = parent_node.kind().to_owned();

                let is_first = parent_node.children()[..child_idx].iter().all(|k| punct_set.contains(k.kind()));
                let is_last = parent_node.children()[child_idx + 1..].iter().all(|k| punct_set.contains(k.kind()));

                let outcome = if is_first { format!("{START}{parent_type}") } else { format!("{CONT}{parent_type}") };
                chunks[ci].set_label(Some(outcome.clone()));
                p.node_at_path_mut(&path).set_label(Some(outcome.clone()));

                if etype == ParserEventType::Build {
                    let ctx = bcg.get_context(&chunks, ci);
                    events.push(event_from_strings(&outcome, &ctx));
                }

                let mut start = ci;
                while start > 0
                    && !paths[start - 1].is_empty()
                    && &paths[start - 1][..paths[start - 1].len() - 1] == parent_path
                {
                    start -= 1;
                }

                if is_last {
                    if etype == ParserEventType::Check {
                        let ctx = kcg.get_context(&chunks, &parent_type, start, ci);
                        events.push(event_from_strings(COMPLETE, &ctx));
                    }
                    let reduce_start = start;
                    let reduce_end = ci;
                    if parent_type != TOP_NODE {
                        let prev_p = chunks[reduce_start].previous_punctuation().to_vec();
                        let next_p = chunks[reduce_end].next_punctuation().to_vec();
                        let parent_mut = p.node_at_path_mut(parent_path);
                        parent_mut.set_previous_punctuation(prev_p);
                        parent_mut.set_next_punctuation(next_p);
                        let parent_clone = parent_mut.clone();
                        let parent_path_owned = parent_path.to_vec();
                        chunks.splice(reduce_start..=reduce_end, std::iter::once(parent_clone));
                        paths.splice(reduce_start..=reduce_end, std::iter::once(parent_path_owned));
                    } else {
                        chunks.clear();
                        paths.clear();
                    }
                    // OpenNLP sets `ci = reduceStart - 1` and increments, so the reduced node is
                    // visited next, including when the reduction starts at index 0.
                    ci = reduce_start;
                    continue;
                } else if etype == ParserEventType::Check {
                    let ctx = kcg.get_context(&chunks, &parent_type, start, ci);
                    events.push(event_from_strings(INCOMPLETE, &ctx));
                }
            }
            ci += 1;
        }
    }
}

fn rules_fix_possessives() -> bool {
    // OpenNLP AbstractParserEventStream sets `fixPossesives = false`.
    false
}

pub(crate) fn event_from_strings(outcome: &str, ctx: &[String]) -> Event {
    let preds: Vec<JavaString> = ctx.iter().map(|s| JavaString::new(s.as_str())).collect();
    Event::new(outcome, preds)
}

/// The manifest entries a parser's `train` collects from the reports of the maxent models it
/// trains (`chunking.Parser.mergeReportIntoManifest`): each report entry under its model's name and
/// a `.`, as `build.Training-Eventhash`.
#[derive(Debug, Default)]
pub(crate) struct TrainingReports(Vec<(String, String)>);

impl TrainingReports {
    /// Trains the maxent model `namespace` names on `events`, keeps its report, and returns the
    /// model's bytes.
    pub(crate) fn train(
        &mut self,
        namespace: &str,
        events: &[Event],
        params: &TrainingParameters,
    ) -> Result<Vec<u8>, ParserModelError> {
        let (model, report) = trainer::train_events(events, params, &mut TrainingConfiguration::default())
            .map_err(turbo_train_components::error::Error::Train)?;
        for (key, value) in report.manifest_entries() {
            self.0.push((format!("{namespace}.{key}"), value));
        }
        Ok(model.write().map_err(turbo_train_components::error::Error::Train)?)
    }

    /// The entries, for [`ParserModel::from_serialized_parts`].
    pub(crate) fn manifest_entries(&self) -> Vec<(&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }
}

pub(crate) fn collect_pos_samples(samples: &[Parse]) -> Vec<PosSample<'static>> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let pos = ParserPosSample::from_parse(p);
        let toks: Vec<Field<'static>> = pos.sentence.into_iter().map(|s| Field::made(s, Span::new(0, 0))).collect();
        let tags: Vec<Field<'static>> = pos.tags.into_iter().map(|s| Field::made(s, Span::new(0, 0))).collect();
        if let Ok(sample) = PosSample::new(toks, tags, Span::new(0, 0)) {
            out.push(sample);
        }
    }
    out
}

/// The chunker training events (`ChunkerEventStream` over `ChunkSampleStream`), one per token.
fn parser_chunker_events(samples: &[Parse]) -> Vec<Event> {
    let mut cg = ChunkContextGenerator::with_cache(CHUNKER_DEFAULT_BEAM_SIZE);
    let mut events = Vec::new();
    for p in samples {
        let cs = ParserChunkSample::from_parse(p);
        let words: Vec<&str> = cs.sentence.iter().map(String::as_str).collect();
        let tags: Vec<&str> = cs.tags.iter().map(String::as_str).collect();
        let preds: Vec<&str> = cs.preds.iter().map(String::as_str).collect();
        for i in 0..words.len() {
            let ctx = cg.get_context(i, &words, &tags, &preds);
            events.push(event_from_strings(preds[i], &ctx));
        }
    }
    events
}

pub(crate) fn train_parser_chunker_bytes(
    language: &str,
    samples: &[Parse],
    params: &TrainingParameters,
) -> Result<Vec<u8>, ParserModelError> {
    let events = parser_chunker_events(samples);
    let beam = params.get_int(BEAM_SIZE_KEY).ok().flatten().unwrap_or(CHUNKER_DEFAULT_BEAM_SIZE as i32);
    let (model, report) = trainer::train_events(&events, params, &mut TrainingConfiguration::default())
        .map_err(turbo_train_components::error::Error::Train)?;
    let model_bytes = model.write().map_err(turbo_train_components::error::Error::Train)?;

    let mut manifest = Stamp::at(0).manifest("ChunkerME", language);
    manifest.set_training(&report);
    manifest.set(FACTORY_KEY, PARSER_CHUNKER_FACTORY);
    manifest.set(BEAM_SIZE_KEY, &beam.to_string());
    let mut package = ModelPackage::new(manifest);
    add_artifacts(&mut package, vec![(turbo_sequence::chunker::MODEL_ENTRY_NAME, model_bytes)])?;
    Ok(package.to_bytes())
}

/// Training parameters for the sub-models of a constituency parser (`chunking::Parser::train`
/// and `treeinsert::Parser::train`).
#[derive(Clone, Debug)]
pub struct ParserTrainingParameters {
    /// Dictionary n-gram frequency cutoff (`dict.Cutoff`, default `5`).
    pub dict_cutoff: usize,
    /// Parameters for `build.model`.
    pub build: TrainingParameters,
    /// Parameters for `check.model`.
    pub check: TrainingParameters,
    /// Parameters for `attach.model` (`TREEINSERT` only).
    pub attach: TrainingParameters,
    /// Parameters for `parsertager.postagger`.
    pub tagger: TrainingParameters,
    /// Parameters for `parserchunker.chunker`.
    pub chunker: TrainingParameters,
}

impl Default for ParserTrainingParameters {
    fn default() -> Self {
        Self::with_iterations_and_cutoff(100, 5)
    }
}

impl ParserTrainingParameters {
    /// Creates [`ParserTrainingParameters`] where every sub-model uses `iterations` and `cutoff`.
    pub fn with_iterations_and_cutoff(iterations: i32, cutoff: i32) -> Self {
        let make = || {
            let mut p = TrainingParameters::new();
            p.put(turbo_train::Key::ITERATIONS, iterations);
            p.put(turbo_train::Key::CUTOFF, cutoff);
            p
        };
        let mut tagger = make();
        tagger.put(BEAM_SIZE_KEY, 10);
        Self {
            dict_cutoff: cutoff.max(0) as usize,
            build: make(),
            check: make(),
            attach: make(),
            tagger,
            chunker: make(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_events_use_each_sample_s_own_words() {
        // Same tags and outcomes, so the cache keys match; only the words differ.
        let samples: Vec<Parse> = ["(TOP (NP (DT The) (NN dog)))", "(TOP (NP (DT A) (NN cat)))"]
            .iter()
            .map(|s| Parse::parse(s).unwrap())
            .collect();
        let events = parser_chunker_events(&samples);
        assert_eq!(events.len(), 4);
        let has = |e: &Event, f: &str| e.context().iter().any(|c| c == f);
        assert!(has(&events[0], "0=The|DT"));
        assert!(has(&events[2], "0=A|DT"), "{:?}", events[2].context());
        assert!(has(&events[3], "0=cat|NN"), "{:?}", events[3].context());
    }
}
