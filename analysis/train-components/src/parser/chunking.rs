//! The chunking parser's training side: the events of `chunking.ParserEventStream` and
//! `chunking.Parser.train`.

use std::collections::HashSet;

use turbo_parser::bottom_up::{build_dictionary, collapse_punctuation_in_place};
use turbo_parser::chunking::{BuildContextGenerator, CheckContextGenerator};
use turbo_parser::model::{HeadRulesArtifact, ParserEventType, ParserModel, ParserType};
use turbo_parser::{
    COMPLETE, CONT, ChunkContextGenerator, HeadRules, INCOMPLETE, Parse, ParseError, ParserChunkSample, START, TOP_NODE,
};
use turbo_train::Event;

use super::stream::{add_tag_events, event_from_strings};
use super::{ParserTrainingParameters, TrainingReports, model, train_parser_chunker_bytes, train_parser_tagger_bytes};
use crate::Stamp;
use crate::postag::BEAM_SIZE_KEY;

/// Trains a `CHUNKING` [`ParserModel`] from `samples` (`chunking.Parser.train`).
pub fn train(
    language: &str,
    samples: &[Parse],
    rules: HeadRulesArtifact,
    params: &ParserTrainingParameters,
) -> crate::Result<ParserModel> {
    let dict = build_dictionary(samples, &rules, params.dict_cutoff);

    let mut report = TrainingReports::default();

    // 1. Train build model
    let build_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Build, Some(&dict))?;
    let build_bytes = report.train("build", &build_events, &params.build)?;

    // 2. Train POS tagger sub-model
    let mut tagger_params = params.tagger.clone();
    if tagger_params.get_int(BEAM_SIZE_KEY).ok().flatten().is_none() {
        tagger_params.put(BEAM_SIZE_KEY, 10);
    }
    let tagger_bytes = train_parser_tagger_bytes(language, samples, &tagger_params)?;

    // 3. Train chunker sub-model with ParserChunkerFactory
    let chunker_bytes = train_parser_chunker_bytes(language, samples, &params.chunker)?;

    // 4. Train check model
    let check_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Check, None)?;
    let check_bytes = report.train("check", &check_events, &params.check)?;

    model::from_serialized_parts(
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
                ParserEventType::Tag => add_tag_events(&p, &mut events),
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
