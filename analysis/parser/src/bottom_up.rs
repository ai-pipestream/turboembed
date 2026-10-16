//! Shared bottom-up constituency parser driver, punctuation collapsing, dictionary builder,
//! and parser factory (`opennlp.tools.parser.Parser`, `opennlp.tools.parser.AbstractBottomUpParser`,
//! and `opennlp.tools.parser.ParserFactory`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_jvm::strict_math;
use turbo_postag::PosModel;
use turbo_sequence::BeamSearch;

use crate::head_rules::HeadRules;
use crate::model::{ParserChunkerModel, ParserModel, ParserType};
use crate::parse::{CONT, OTHER, Parse, START, TOP_NODE, insert_by_prob};

/// Default beam width `M` (`AbstractBottomUpParser.defaultBeamSize`).
pub const DEFAULT_BEAM_SIZE: usize = 20;

/// Default probability mass threshold `Q` (`AbstractBottomUpParser.defaultAdvancePercentage`).
pub const DEFAULT_ADVANCE_PERCENTAGE: f64 = 0.95;

/// Floor log probability (`BeamSearch.ZERO_LOG`).
pub const ZERO_LOG: f64 = -100000.0;

/// Interface for full syntactic constituency parsers (`opennlp.tools.parser.Parser`).
pub trait Parser {
    /// Returns up to `num_parses` highest-scoring constituency parses for `tokens`
    /// (`Parser.parse(Parse, int)`).
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Vec<Parse>;

    /// Returns the single best constituency parse for `tokens` (`Parser.parse(Parse)`).
    fn parse(&self, tokens: &Parse) -> Option<Parse> {
        self.parse_n(tokens, 1).into_iter().next()
    }
}

/// Concrete parser instance created by [`ParserFactory`].
#[derive(Debug)]
pub enum ParserEngine {
    /// Shift-reduce chunking parser (`opennlp.tools.parser.chunking.Parser`).
    Chunking(crate::chunking::Parser),
    /// Tree-insertion parser (`opennlp.tools.parser.treeinsert.Parser`).
    TreeInsert(crate::treeinsert::Parser),
}

impl Parser for ParserEngine {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Vec<Parse> {
        match self {
            Self::Chunking(p) => p.parse_n(tokens, num_parses),
            Self::TreeInsert(p) => p.parse_n(tokens, num_parses),
        }
    }

    fn parse(&self, tokens: &Parse) -> Option<Parse> {
        match self {
            Self::Chunking(p) => p.parse(tokens),
            Self::TreeInsert(p) => p.parse(tokens),
        }
    }
}

/// Factory for instantiating a constituency parser from a [`ParserModel`]
/// (`opennlp.tools.parser.ParserFactory`).
pub struct ParserFactory;

impl ParserFactory {
    /// Creates a [`ParserEngine`] with default beam size (20) and advance percentage (0.95)
    /// (`ParserFactory.create(ParserModel)`).
    pub fn create(model: impl Into<Arc<ParserModel>>) -> ParserEngine {
        Self::create_with_params(model, DEFAULT_BEAM_SIZE, DEFAULT_ADVANCE_PERCENTAGE)
    }

    /// Creates a [`ParserEngine`] with explicit `beam_size` and `advance_percentage`
    /// (`ParserFactory.create(ParserModel, int, double)`).
    pub fn create_with_params(
        model: impl Into<Arc<ParserModel>>,
        beam_size: usize,
        advance_percentage: f64,
    ) -> ParserEngine {
        let model = model.into();
        match model.parser_type() {
            ParserType::Chunking => {
                ParserEngine::Chunking(crate::chunking::Parser::with_params(model, beam_size, advance_percentage))
            }
            ParserType::TreeInsert => {
                ParserEngine::TreeInsert(crate::treeinsert::Parser::with_params(model, beam_size, advance_percentage))
            }
        }
    }
}

/// Collapses punctuation nodes in `chunks` in place (`AbstractBottomUpParser.collapsePunctuation`),
/// attaching each punctuation node to the preceding non-punctuation node's `next_punctuation` set
/// and the following non-punctuation node's `previous_punctuation` set, and returning the indices
/// in `chunks` of the non-punctuation constituents.
pub fn collapse_punctuation_in_place(chunks: &mut [Parse], punct_set: &HashSet<String>) -> Vec<usize> {
    let cn = chunks.len();
    let mut non_punct_indices = Vec::with_capacity(cn);
    let mut last_non_punct: Option<usize> = None;
    for ci in 0..cn {
        if punct_set.contains(chunks[ci].kind()) {
            let punct_clone = chunks[ci].clone();
            if let Some(last) = last_non_punct {
                chunks[last].add_next_punctuation(punct_clone.clone());
            }
            if let Some(next) = ((ci + 1)..cn).find(|&ni| !punct_set.contains(chunks[ni].kind())) {
                chunks[next].add_previous_punctuation(punct_clone);
            }
        } else {
            non_punct_indices.push(ci);
            last_non_punct = Some(ci);
        }
    }
    non_punct_indices
}

/// Returns a new vector of non-punctuation [`Parse`] nodes with punctuation collapsed onto
/// adjacent non-punctuation nodes (`AbstractBottomUpParser.collapsePunctuation`).
pub fn collapse_punctuation(chunks: &[Parse], punct_set: &HashSet<String>) -> Vec<Parse> {
    let mut cloned = chunks.to_vec();
    let kept = collapse_punctuation_in_place(&mut cloned, punct_set);
    if kept.len() == cloned.len() {
        return cloned;
    }
    let mut out = Vec::with_capacity(kept.len());
    let mut slots: Vec<Option<Parse>> = cloned.into_iter().map(Some).collect();
    for idx in kept {
        out.push(slots[idx].take().expect("each index once"));
    }
    out
}

/// Advances POS tag assignments for `p` using `tagger` (`AbstractBottomUpParser.advanceTags`).
pub fn advance_tags(p: &Parse, tagger: &PosModel, _beam_size: usize) -> Vec<Parse> {
    let children = p.children();
    let words: Vec<&str> = children.iter().map(Parse::covered_text).collect();
    let mut context = tagger.new_context().expect("validated PosModel descriptor");
    let validator = tagger.validator();
    let search = BeamSearch::new(tagger.beam_size(), tagger.maxent()).expect("validated PosModel outcomes");
    let sequences = search.best_sequences(tagger.beam_size(), &words, ZERO_LOG, &mut context, &validator);

    let mut out = Vec::with_capacity(sequences.len());
    for seq in &sequences {
        let tags = seq.outcomes();
        let probs = seq.probs();
        let mut new_parse = p.clone();
        for (j, (&tag, &prob)) in tags.iter().zip(probs.iter()).enumerate() {
            let word = &children[j];
            new_parse
                .insert(Parse::new(Arc::clone(word.text_arc()), word.span(), tag, prob, j as i32))
                .expect("word span lies inside sentence");
            new_parse.add_prob(strict_math::log(prob));
        }
        out.push(new_parse);
    }
    out
}

/// Advances basal chunk assignments for `p` (`AbstractBottomUpParser.advanceChunks`).
pub fn advance_chunks(
    p: &Parse,
    min_chunk_score: f64,
    chunker: &ParserChunkerModel,
    head_rules: &dyn HeadRules,
) -> Vec<Parse> {
    let children = p.children();
    let mut words = Vec::with_capacity(children.len());
    let mut ptags = Vec::with_capacity(children.len());
    for child in children {
        words.push(child.covered_text());
        ptags.push(child.kind());
    }
    let cs = chunker.top_k_sequences(&words, &ptags, min_chunk_score - p.prob());
    let mut new_parses = Vec::with_capacity(cs.len());
    for seq in &cs {
        let tags = seq.outcomes();
        let probs = seq.probs();
        let mut new_parse = p.clone();
        let mut start = 0usize;
        let mut end = 0usize;
        let mut ctype: Option<&str> = None;

        for j in 0..=tags.len() {
            if j != tags.len() {
                new_parse.add_prob(strict_math::log(probs[j]));
            }
            if j != tags.len() && tags[j].starts_with(CONT) {
                end = j;
            } else {
                if let Some(kind) = ctype {
                    let p1 = &p.children()[start];
                    let p2 = &p.children()[end];
                    let cons = &p.children()[start..=end];
                    let head = head_rules.get_head(cons, kind);
                    let mut chunk = Parse::with_head(
                        Arc::clone(p.text_arc()),
                        Span::new(p1.span().start, p2.span().end),
                        kind,
                        1.0,
                        head,
                    );
                    chunk.set_is_chunk(true);
                    new_parse.insert(chunk).expect("chunk span lies inside sentence");
                }
                if j != tags.len() {
                    if let Some(rest) = tags[j].strip_prefix(START) {
                        ctype = Some(rest);
                        start = j;
                        end = j;
                    } else {
                        ctype = None;
                    }
                }
            }
        }
        new_parses.push(new_parse);
    }
    new_parses
}

/// Executes the shared bottom-up beam search loop (`AbstractBottomUpParser.parse(Parse, int)`).
#[allow(clippy::too_many_arguments)]
pub fn parse_bottom_up<FAdvance, FTop>(
    tokens: &Parse,
    num_parses: usize,
    m_beam: usize,
    k_beam: usize,
    q_mass: f64,
    tagger: &PosModel,
    chunker: &ParserChunkerModel,
    head_rules: &dyn HeadRules,
    mut advance_parses: FAdvance,
    mut advance_top: FTop,
) -> Vec<Parse>
where
    FAdvance: FnMut(&mut Parse, f64) -> Vec<Parse>,
    FTop: FnMut(&mut Parse),
{
    if tokens.child_count() == 0 {
        return Vec::new();
    }
    let mut old_derivations: Vec<Parse> = Vec::new();
    let mut new_derivations: Vec<Parse> = Vec::new();
    let mut complete_parses: Vec<Parse> = Vec::new();
    let max_stage_length = 2 * tokens.child_count() + 3;

    let mut min_chunk_score = ZERO_LOG;
    let tag_parses = advance_tags(tokens, tagger, m_beam);
    for tp in &tag_parses {
        let chunk_parses = advance_chunks(tp, min_chunk_score, chunker, head_rules);
        for cp in chunk_parses {
            insert_by_prob(&mut old_derivations, cp);
        }
        if old_derivations.len() > k_beam {
            min_chunk_score = old_derivations[k_beam - 1].prob();
        }
    }

    let mut derivation_stage = 0usize;
    let mut min_complete_mass = ZERO_LOG;
    let mut parse_over_mass = true;

    while !old_derivations.is_empty()
        && (complete_parses.len() < m_beam || parse_over_mass)
        && derivation_stage < max_stage_length
    {
        new_derivations.clear();
        let mut derivation_rank = 0usize;

        for old_p in &old_derivations {
            if derivation_rank >= k_beam {
                break;
            }
            let mut p = old_p.clone();
            if p.prob() < min_complete_mass && complete_parses.len() >= num_parses {
                break;
            }
            derivation_rank += 1;
            let next = advance_parses(&mut p, q_mass);
            if !next.is_empty() {
                for mut np in next {
                    if np.complete() {
                        advance_top(&mut np);
                        insert_by_prob(&mut complete_parses, np);
                        min_complete_mass = complete_parses[complete_parses.len() - 1].prob();
                    } else {
                        insert_by_prob(&mut new_derivations, np);
                    }
                }
            } else if complete_parses.is_empty() && derivation_rank == 1 && new_derivations.is_empty() {
                guess_parse(p, head_rules, &mut complete_parses);
                break;
            }
        }
        derivation_stage += 1;
        std::mem::swap(&mut old_derivations, &mut new_derivations);
        if complete_parses.len() >= num_parses
            && !old_derivations.is_empty()
            && old_derivations[0].prob() < min_complete_mass
        {
            parse_over_mass = false;
        }
    }

    if complete_parses.is_empty() {
        let mut gp = advance_tags(tokens, tagger, 1).into_iter().next().unwrap_or_else(|| tokens.clone());
        let child_spans: Vec<Span> = gp.children().iter().map(Parse::span).collect();
        for sp in child_spans {
            gp.insert(Parse::new(Arc::clone(gp.text_arc()), sp, OTHER, 0.0, 0)).expect("child span inside root");
        }
        vec![gp]
    } else {
        complete_parses.truncate(num_parses);
        complete_parses
    }
}

fn guess_parse(mut p: Parse, head_rules: &dyn HeadRules, complete_parses: &mut Vec<Parse>) {
    p.set_kind(TOP_NODE);
    let first = &p.children()[0];
    let last = &p.children()[p.child_count() - 1];
    let head = head_rules.get_head(p.children(), TOP_NODE);
    let top_span = Span::new(first.span().start, last.span().end);
    p.insert(Parse::with_head(Arc::clone(p.text_arc()), top_span, TOP_NODE, 0.0, head))
        .expect("children lie inside root");
    insert_by_prob(complete_parses, p);
}

/// Builds the n-gram dictionary (`Set<String>` of comma-joined unigrams, bigrams, and trigrams)
/// from training parses (`AbstractBottomUpParser.buildDictionary`).
pub fn build_dictionary(parses: &[Parse], rules: &dyn HeadRules, cutoff: usize) -> HashSet<String> {
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut order: Vec<Vec<String>> = Vec::new();
    let add_ngram = |ngram: Vec<String>, counts: &mut HashMap<Vec<String>, usize>, order: &mut Vec<Vec<String>>| {
        if let Some(c) = counts.get_mut(&ngram) {
            *c += 1;
        } else {
            order.push(ngram.clone());
            counts.insert(ngram, 1);
        }
    };
    let add_window = |words: &[String],
                      min_len: usize,
                      max_len: usize,
                      counts: &mut HashMap<Vec<String>, usize>,
                      order: &mut Vec<Vec<String>>| {
        for len in min_len..=max_len {
            if words.len() >= len {
                for start in 0..=(words.len() - len) {
                    add_ngram(words[start..start + len].to_vec(), counts, order);
                }
            }
        }
    };

    let punct_set = rules.punctuation_tags();
    for orig in parses {
        let mut p = orig.clone();
        p.update_heads(rules);
        let words: Vec<String> = p.tag_nodes().into_iter().map(|t| t.covered_text().to_owned()).collect();
        add_window(&words, 1, 1, &mut counts, &mut order);

        let init_paths = p.initial_chunk_paths();
        let mut chunk_paths: Vec<Vec<usize>> =
            init_paths.into_iter().filter(|path| !punct_set.contains(p.node_at_path(path).kind())).collect();
        let cwords: Vec<String> = chunk_paths.iter().map(|path| p.node_at_path(path).head_text().to_owned()).collect();
        add_window(&cwords, 2, 3, &mut counts, &mut order);

        let mut ci = 0usize;
        while ci < chunk_paths.len() {
            let path = &chunk_paths[ci];
            if !path.is_empty() {
                let parent_path = &path[..path.len() - 1];
                let child_idx = path[path.len() - 1];
                let parent = p.node_at_path(parent_path);
                let is_last = parent.children()[child_idx + 1..].iter().all(|k| punct_set.contains(k.kind()));
                if is_last {
                    let parent_path_owned = parent_path.to_vec();
                    let mut reduce_start = ci;
                    while reduce_start > 0
                        && !chunk_paths[reduce_start - 1].is_empty()
                        && chunk_paths[reduce_start - 1][..chunk_paths[reduce_start - 1].len() - 1]
                            == parent_path_owned[..]
                    {
                        reduce_start -= 1;
                    }
                    if parent.kind() != TOP_NODE {
                        chunk_paths.splice(reduce_start..=ci, std::iter::once(parent_path_owned));
                    } else {
                        chunk_paths.clear();
                    }
                    ci = reduce_start;
                    if !chunk_paths.is_empty() {
                        let mut window = Vec::with_capacity(5);
                        if let Some(j) = ci.checked_sub(2) {
                            window.push(p.node_at_path(&chunk_paths[j]).head_text().to_owned());
                        }
                        if let Some(j) = ci.checked_sub(1) {
                            window.push(p.node_at_path(&chunk_paths[j]).head_text().to_owned());
                        }
                        window.push(p.node_at_path(&chunk_paths[ci]).head_text().to_owned());
                        if ci + 1 < chunk_paths.len() {
                            window.push(p.node_at_path(&chunk_paths[ci + 1]).head_text().to_owned());
                        }
                        if ci + 2 < chunk_paths.len() {
                            window.push(p.node_at_path(&chunk_paths[ci + 2]).head_text().to_owned());
                        }
                        if window.len() >= 3 {
                            add_window(&window, 2, 3, &mut counts, &mut order);
                        } else if window.len() == 2 {
                            add_window(&window, 2, 2, &mut counts, &mut order);
                        }
                    }
                    if reduce_start == 0 {
                        continue;
                    }
                    ci = reduce_start - 1;
                }
            }
            ci += 1;
        }
    }

    let mut dict = HashSet::new();
    for ngram in order {
        if counts.get(&ngram).copied().unwrap_or(0) >= cutoff {
            dict.insert(ngram.join(","));
        }
    }
    dict
}
