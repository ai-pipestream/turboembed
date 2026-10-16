//! The tree-insertion parser's training side: the events of `treeinsert.ParserEventStream` and
//! `treeinsert.Parser.train`.
//!
//! The events follow `opennlp.tools.parser.treeinsert` exactly, as the parser in
//! `turbo_parser::treeinsert` does; that module says how both repeat the changes OpenNLP's
//! context generators make to the punctuation sets of the nodes they look at.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_parser::bottom_up::{build_dictionary, collapse_punctuation_in_place};
use turbo_parser::model::{HeadRulesArtifact, ParserEventType, ParserModel, ParserType};
use turbo_parser::treeinsert::{
    ATTACH_DAUGHTER, ATTACH_SISTER, AttachContextGenerator, BUILT, BuildContextGenerator, CheckContextGenerator, DONE,
    NON_ATTACH, TreeInsertError, collapse_children, right_frontier_paths,
};
use turbo_parser::{COMPLETE, ChunkContextGenerator, HeadRules, INCOMPLETE, Parse, ParserChunkSample, TOP_NODE};
use turbo_train::Event;

use super::stream::{add_tag_events, event_from_strings};
use super::{ParserTrainingParameters, TrainingReports, model, train_parser_chunker_bytes, train_parser_tagger_bytes};
use crate::Stamp;

/// Trains a `TREEINSERT` [`ParserModel`] from `samples` (`treeinsert.Parser.train`).
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

    // 2. Train check model
    let check_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Check, None)?;
    let check_bytes = report.train("check", &check_events, &params.check)?;

    // 3. Train attach model
    let attach_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Attach, None)?;
    let attach_bytes = report.train("attach", &attach_events, &params.attach)?;

    // 4. Train POS tagger sub-model. Unlike the chunking parser, OpenNLP's tree-insertion
    // `train` leaves the tagger's beam size alone, so it is the tagger's default of 3 unless the
    // parameters give one.
    let tagger_bytes = train_parser_tagger_bytes(language, samples, &params.tagger)?;

    // 5. Train chunker sub-model with ParserChunkerFactory
    let chunker_bytes = train_parser_chunker_bytes(language, samples, &params.chunker)?;

    model::from_serialized_parts(
        language,
        build_bytes,
        check_bytes,
        Some(attach_bytes),
        tagger_bytes,
        chunker_bytes,
        rules,
        ParserType::TreeInsert,
        &report.manifest_entries(),
        Stamp::at(0),
    )
}

/// The path of the parent of the node at `path`, or `None` for the root.
fn path_parent(path: &[usize]) -> Option<&[usize]> {
    path.split_last().map(|(_, parent)| parent)
}

/// Event stream for the tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.ParserEventStream`).
pub struct ParserEventStream;

/// The context generators of one event stream.
struct Generators {
    build: BuildContextGenerator,
    attach: AttachContextGenerator,
    check: CheckContextGenerator,
}

impl ParserEventStream {
    /// Generates training [`Event`]s of type `etype` over `samples` (`AbstractParserEventStream`
    /// with `treeinsert.ParserEventStream.addParseEvents`). The dictionary is not used by the
    /// tree-insertion build features.
    pub fn generate_events(
        samples: &[Parse],
        rules: &dyn HeadRules,
        etype: ParserEventType,
        _dict: Option<&HashSet<String>>,
    ) -> Result<Vec<Event>, TreeInsertError> {
        let mut events = Vec::new();
        let punct_set = rules.punctuation_tags();
        let cgs = Generators {
            build: BuildContextGenerator::new(),
            attach: AttachContextGenerator::new(punct_set.clone()),
            check: CheckContextGenerator::new(punct_set.clone()),
        };
        let mut chunk_cg = ChunkContextGenerator::new();

        for orig in samples {
            let mut p = orig.clone();
            Parse::prune_parse(&mut p)?;
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
                    Self::add_parse_events(&mut events, &mut p, rules, etype, &cgs)?;
                }
            }
        }
        Ok(events)
    }

    /// `treeinsert.ParserEventStream.addParseEvents` over the gold parse `gold`.
    ///
    /// Gold nodes are named by their paths in `gold`, whose shape never changes here. The parse
    /// built so far is `current[0]`, with the chunks still to attach after it; its nodes are named
    /// by their paths in `current[0]`.
    fn add_parse_events(
        events: &mut Vec<Event>,
        gold: &mut Parse,
        rules: &dyn HeadRules,
        etype: ParserEventType,
        cgs: &Generators,
    ) -> Result<(), TreeInsertError> {
        let punct_set = rules.punctuation_tags();
        let init_paths = gold.initial_chunk_paths();
        let mut init_nodes: Vec<Parse> = init_paths.iter().map(|path| gold.node_at_path(path).clone()).collect();
        let kept = collapse_punctuation_in_place(&mut init_nodes, punct_set);

        let mut chunks: Vec<Vec<usize>> = Vec::with_capacity(kept.len());
        let mut current: Vec<Parse> = Vec::with_capacity(kept.len());
        for idx in kept {
            let mut node = init_nodes[idx].clone();
            node.set_label(Some(COMPLETE.to_owned()));
            gold.node_at_path_mut(&init_paths[idx]).set_label(Some(COMPLETE.to_owned()));
            chunks.push(init_paths[idx].clone());
            current.push(node);
        }

        let kind_at = |gold: &Parse, path: &[usize]| gold.node_at_path(path).kind().to_owned();
        let mut right_frontier: Vec<Vec<usize>> = Vec::new();
        let mut built_nodes: Vec<Vec<usize>> = Vec::new();

        for ci in 0..chunks.len() {
            let mut parent = path_parent(&chunks[ci]).ok_or(TreeInsertError::MissingParent)?.to_vec();
            let mut prev_parent = chunks[ci].clone();
            let mut off = 0;
            if !gold.node_at_path(&chunks[ci]).is_pos_tag() {
                built_nodes.insert(off, chunks[ci].clone());
                off += 1;
            }

            // Build stages
            loop {
                let parent_node = gold.node_at_path(&parent);
                if parent_node.kind() == TOP_NODE || parent_node.label().is_some() {
                    break;
                }
                let parent_type = parent_node.kind().to_owned();
                if kind_at(gold, &prev_parent) != parent_type {
                    if etype == ParserEventType::Build {
                        let ctx = cgs.build.get_context(&current.iter().collect::<Vec<_>>(), ci);
                        events.push(event_from_strings(&parent_type, &ctx));
                    }
                    built_nodes.insert(off, parent.clone());
                    off += 1;

                    let cur = &current[ci];
                    let (prev_punct, next_punct) =
                        (cur.previous_punctuation().to_vec(), cur.next_punctuation().to_vec());
                    let mut new_parent = Parse::new(Arc::clone(cur.text_arc()), cur.span(), &parent_type, 1.0, 0);
                    new_parent.add(cur.clone(), rules)?;
                    new_parent.set_previous_punctuation(prev_punct);
                    new_parent.set_next_punctuation(next_punct);
                    new_parent.set_label(Some(BUILT.to_owned()));
                    current[ci] = new_parent;

                    let complete = Self::last_child(gold, &chunks[ci], &parent, punct_set);
                    if etype == ParserEventType::Check {
                        let refs: Vec<&Parse> = current.iter().collect();
                        let ctx = cgs.check.get_context(&current[ci], &refs, ci, false)?;
                        events.push(event_from_strings(if complete { COMPLETE } else { INCOMPLETE }, &ctx));
                        collapse_children(&mut current[ci], punct_set);
                    }
                    current[ci].set_label(Some(if complete { COMPLETE } else { INCOMPLETE }.to_owned()));
                    gold.node_at_path_mut(&parent).set_label(Some(COMPLETE.to_owned()));
                    chunks[ci] = parent.clone();
                }
                gold.node_at_path_mut(&parent).set_label(Some(BUILT.to_owned()));
                prev_parent = parent.clone();
                parent = path_parent(&parent).ok_or(TreeInsertError::MissingParent)?.to_vec();
            }

            if etype == ParserEventType::Build {
                let ctx = cgs.build.get_context(&current.iter().collect::<Vec<_>>(), ci);
                events.push(event_from_strings(DONE, &ctx));
            }

            if ci > 0 {
                Self::attach(events, gold, &mut current, &chunks, &mut right_frontier, ci, rules, etype, cgs)?;
            }

            built_nodes.append(&mut right_frontier);
            std::mem::swap(&mut right_frontier, &mut built_nodes);
        }
        Ok(())
    }

    /// The attachment part of `addParseEvents` for chunk `ci > 0`.
    #[allow(clippy::too_many_arguments)]
    fn attach(
        events: &mut Vec<Event>,
        gold: &mut Parse,
        current: &mut [Parse],
        chunks: &[Vec<usize>],
        right_frontier: &mut Vec<Vec<usize>>,
        ci: usize,
        rules: &dyn HeadRules,
        etype: ParserEventType,
        cgs: &Generators,
    ) -> Result<(), TreeInsertError> {
        let punct_set = rules.punctuation_tags();
        let crf = right_frontier_paths(&current[0], punct_set);
        if crf.len() != right_frontier.len() {
            return Err(TreeInsertError::FrontierMismatch { current: crf.len(), gold: right_frontier.len() });
        }
        let parents = Self::non_adjoined_parents(gold, &chunks[ci], punct_set)?;
        let mut attach: Option<(&'static str, usize)> = None;

        // Try daughters first.
        let mut collapsed = Vec::new();
        {
            let refs: Vec<&Parse> = current.iter().collect();
            let crf_nodes: Vec<&Parse> = crf.iter().map(|path| current[0].node_at_path(path)).collect();
            for (cfi, cfn) in crf_nodes.iter().enumerate() {
                let i = parents.get(&right_frontier[cfi]).copied();
                let non_punct = cfn.children().iter().filter(|k| !punct_set.contains(k.kind())).count();
                if attach.is_none() && i == Some(non_punct) {
                    attach = Some((ATTACH_DAUGHTER, cfi));
                    if etype == ParserEventType::Attach {
                        let ctx = cgs.attach.get_context(&refs, ci, &crf_nodes, cfi);
                        events.push(event_from_strings(ATTACH_DAUGHTER, &ctx));
                        collapsed.push(cfi);
                    }
                }
            }
        }
        for &cfi in &collapsed {
            collapse_children(current[0].node_at_path_mut(&crf[cfi]), punct_set);
        }

        // Try sisters, and generate non-attach events.
        collapsed.clear();
        {
            let refs: Vec<&Parse> = current.iter().collect();
            let crf_nodes: Vec<&Parse> = crf.iter().map(|path| current[0].node_at_path(path)).collect();
            for (cfi, frontier) in right_frontier.iter().enumerate() {
                let sister_site = path_parent(frontier).is_some_and(|fp| {
                    parents.contains_key(fp) && gold.node_at_path(frontier).kind() == gold.node_at_path(fp).kind()
                });
                let outcome = if attach.is_none() && sister_site {
                    attach = Some((ATTACH_SISTER, cfi));
                    Some(ATTACH_SISTER)
                } else if attach.is_some_and(|(_, idx)| idx == cfi) {
                    None
                } else {
                    Some(NON_ATTACH)
                };
                if let Some(outcome) = outcome
                    && etype == ParserEventType::Attach
                {
                    let ctx = cgs.attach.get_context(&refs, ci, &crf_nodes, cfi);
                    events.push(event_from_strings(outcome, &ctx));
                    collapsed.push(cfi);
                }
            }
        }
        for &cfi in &collapsed {
            collapse_children(current[0].node_at_path_mut(&crf[cfi]), punct_set);
        }
        if attach.is_some_and(|(kind, _)| kind == ATTACH_SISTER) {
            let chunk_parent = path_parent(&chunks[ci]).ok_or(TreeInsertError::MissingParent)?;
            gold.node_at_path_mut(chunk_parent).set_label(Some(BUILT.to_owned()));
        }

        let Some((kind, index)) = attach else {
            return Err(TreeInsertError::NoAttachment { span: gold.node_at_path(&chunks[ci]).span() });
        };
        let attachment = current[ci].clone();
        let updated = if kind == ATTACH_DAUGHTER {
            current[0].node_at_path_mut(&crf[index]).add(attachment, rules)?;
            crf[index].clone()
        } else {
            let frontier_parent = path_parent(&right_frontier[index]).ok_or(TreeInsertError::MissingParent)?.to_vec();
            right_frontier[index] = frontier_parent;
            match path_parent(&crf[index]) {
                None => {
                    // The parse built so far sits under a `TOP` node made for the first chunk.
                    let root = std::mem::replace(
                        &mut current[0],
                        Parse::new(Arc::clone(attachment.text_arc()), Span::new(0, 0), "", 0.0, 0),
                    );
                    let text = Arc::clone(root.text_arc());
                    let mut top = Parse::new(Arc::clone(&text), Span::new(0, text.len() as u32), TOP_NODE, 1.0, 0);
                    top.insert(root)?;
                    top.adjoin(attachment, rules)?;
                    current[0] = top.children()[0].clone();
                    Vec::new()
                }
                Some(site) => {
                    let site = site.to_vec();
                    let node = current[0].node_at_path_mut(&site);
                    node.adjoin(attachment, rules)?;
                    let mut path = site;
                    path.push(node.child_count() - 1);
                    path
                }
            }
        };

        let complete = Self::last_child(gold, &chunks[ci], &right_frontier[index], punct_set);
        if etype == ParserEventType::Check {
            let refs: Vec<&Parse> = current.iter().collect();
            let ctx = cgs.check.get_context(current[0].node_at_path(&updated), &refs, ci, true)?;
            events.push(event_from_strings(if complete { COMPLETE } else { INCOMPLETE }, &ctx));
            collapse_children(current[0].node_at_path_mut(&updated), punct_set);
        }
        let node = current[0].node_at_path_mut(&updated);
        if complete {
            node.set_label(Some(COMPLETE.to_owned()));
        } else if kind == ATTACH_SISTER {
            node.set_label(Some(INCOMPLETE.to_owned()));
        }
        right_frontier.drain(..index);
        Ok(())
    }

    /// `treeinsert.ParserEventStream.getNonAdjoinedParent`: the parents of `node` up through the
    /// ancestors of its own type, each with the position among its children that are not
    /// punctuation of the child on the way up.
    fn non_adjoined_parents(
        gold: &Parse,
        node: &[usize],
        punct_set: &HashSet<String>,
    ) -> Result<HashMap<Vec<usize>, usize>, TreeInsertError> {
        let mut parents = HashMap::new();
        let mut node = node.to_vec();
        let mut parent = path_parent(&node).ok_or(TreeInsertError::MissingParent)?.to_vec();
        parents.insert(parent.clone(), Self::index_of(gold, &node, punct_set));
        while gold.node_at_path(&parent).kind() == gold.node_at_path(&node).kind() {
            node = parent;
            parent = path_parent(&node).ok_or(TreeInsertError::MissingParent)?.to_vec();
            parents.insert(parent.clone(), Self::index_of(gold, &node, punct_set));
        }
        Ok(parents)
    }

    /// Position of the child at `child` among the children of its parent that are not punctuation.
    fn index_of(gold: &Parse, child: &[usize], punct_set: &HashSet<String>) -> usize {
        let (&idx, parent) = child.split_last().expect("child below the root");
        gold.node_at_path(parent).children()[..idx].iter().filter(|k| !punct_set.contains(k.kind())).count()
    }

    /// `AbstractParserEventStream.lastChild`: `child` is the last child of `parent` that is not
    /// punctuation.
    fn direct_last_child(gold: &Parse, child: &[usize], parent: &[usize], punct_set: &HashSet<String>) -> bool {
        let Some((&idx, child_parent)) = child.split_last() else {
            return false;
        };
        child_parent == parent
            && gold.node_at_path(parent).children()[idx + 1..].iter().all(|k| punct_set.contains(k.kind()))
    }

    /// `treeinsert.ParserEventStream.lastChild`: `child`, or an ancestor of its own type below
    /// `parent`, is the last child of `parent` that is not punctuation.
    fn last_child(gold: &Parse, child: &[usize], parent: &[usize], punct_set: &HashSet<String>) -> bool {
        let mut child = child;
        let mut lc = Self::direct_last_child(gold, child, parent, punct_set);
        while !lc {
            let Some(cp) = path_parent(child) else {
                break;
            };
            if cp != parent && gold.node_at_path(cp).kind() == gold.node_at_path(child).kind() {
                lc = Self::direct_last_child(gold, cp, parent, punct_set);
                child = cp;
            } else {
                break;
            }
        }
        lc
    }
}
