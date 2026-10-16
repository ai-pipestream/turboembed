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
use crate::parse::{CONT, Parse, ParseError, START, TOP_NODE, insert_by_prob};

/// Default beam width `M` (`AbstractBottomUpParser.defaultBeamSize`).
pub const DEFAULT_BEAM_SIZE: usize = 20;

/// Default probability mass threshold `Q` (`AbstractBottomUpParser.defaultAdvancePercentage`).
pub const DEFAULT_ADVANCE_PERCENTAGE: f64 = 0.95;

/// Floor log probability (`BeamSearch.ZERO_LOG`).
pub const ZERO_LOG: f64 = -100000.0;

/// Interface for full syntactic constituency parsers (`opennlp.tools.parser.Parser`).
pub trait Parser {
    /// Returns up to `num_parses` highest-scoring constituency parses for `tokens`, best first
    /// (`Parser.parse(Parse, int)`). With no complete parse, it is the one incomplete guess, or
    /// nothing where OpenNLP returns `{null}`. An error is where OpenNLP throws.
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError>;

    /// Returns the best constituency parse for `tokens` (`Parser.parse(Parse)`).
    fn parse(&self, tokens: &Parse) -> Result<Option<Parse>, ParseError> {
        Ok(self.parse_n(tokens, 1)?.into_iter().next())
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
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        match self {
            Self::Chunking(p) => p.parse_n(tokens, num_parses),
            Self::TreeInsert(p) => p.parse_n(tokens, num_parses),
        }
    }

    fn parse(&self, tokens: &Parse) -> Result<Option<Parse>, ParseError> {
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
pub fn advance_tags(p: &Parse, tagger: &PosModel) -> Vec<Parse> {
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

/// Executes the shared bottom-up derivation search (`AbstractBottomUpParser.parse(Parse, int)`).
///
/// Stage 0 tags `tokens`, stage 1 chunks each tagging and calls `after_chunks` on the result
/// (tree insertion marks the chunks complete there), and every later stage calls
/// `advance_parses`. A derivation that comes back complete goes through `advance_top` into the
/// complete parses. `advance_parses` returning `None` is OpenNLP's `null`: the derivation itself
/// is passed to `advance_top` and taken as complete. `Some` of an empty list drops it.
///
/// The search runs while there are derivations, fewer than `m_beam` complete parses or a best
/// derivation below the lowest complete probability, and fewer than `2 * tokens + 3` stages;
/// each stage advances at most `k_beam` derivations. Derivations and complete parses are kept as
/// OpenNLP's `TreeSet`s keep them: by descending probability, one parse per probability. With no
/// complete parse, the result is the best derivation of stage 2 (`guess`), or nothing when the
/// search never reached stage 2.
#[allow(clippy::too_many_arguments)]
pub fn parse_bottom_up<FAdvance, FTop, FChunks>(
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
    mut after_chunks: FChunks,
) -> Result<Vec<Parse>, ParseError>
where
    FAdvance: FnMut(&mut Parse, f64) -> Result<Option<Vec<Parse>>, ParseError>,
    FTop: FnMut(&mut Parse),
    FChunks: FnMut(&mut [Parse]),
{
    if tokens.child_count() == 0 {
        // `Parser.parse(Parse)` returns a parse without tokens as it is.
        return Ok(vec![tokens.clone()]);
    }
    let mut odh: Vec<Parse> = vec![tokens.clone()];
    let mut complete_parses: Vec<Parse> = Vec::new();
    let mut guess: Option<Parse> = None;
    let mut min_complete = 2.0;
    let mut best_complete = ZERO_LOG;
    let max_derivation_length = 2 * tokens.child_count() + 3;
    let mut derivation_stage = 0usize;

    while !odh.is_empty()
        && (complete_parses.len() < m_beam || odh[0].prob() < min_complete)
        && derivation_stage < max_derivation_length
    {
        let mut ndh: Vec<Parse> = Vec::new();
        for tp in odh.iter().take(k_beam) {
            if guess.is_none() && derivation_stage == 2 {
                guess = Some(tp.clone());
            }
            let nd = match derivation_stage {
                0 => Some(advance_tags(tp, tagger)),
                1 => {
                    let min_chunk_score = if ndh.len() < k_beam { best_complete } else { ndh[ndh.len() - 1].prob() };
                    let mut chunked = advance_chunks(tp, min_chunk_score, chunker, head_rules);
                    after_chunks(&mut chunked);
                    Some(chunked)
                }
                _ => {
                    let mut p = tp.clone();
                    match advance_parses(&mut p, q_mass)? {
                        Some(nd) => Some(nd),
                        None => {
                            advance_top(&mut p);
                            insert_by_prob(&mut complete_parses, p);
                            None
                        }
                    }
                }
            };
            for mut parse in nd.into_iter().flatten() {
                if parse.complete() {
                    advance_top(&mut parse);
                    if parse.prob() > best_complete {
                        best_complete = parse.prob();
                    }
                    if parse.prob() < min_complete {
                        min_complete = parse.prob();
                    }
                    insert_by_prob(&mut complete_parses, parse);
                } else {
                    insert_by_prob(&mut ndh, parse);
                }
            }
        }
        derivation_stage += 1;
        odh = ndh;
    }

    if complete_parses.is_empty() {
        Ok(guess.into_iter().collect())
    } else {
        complete_parses.truncate(num_parses);
        Ok(complete_parses)
    }
}

/// Builds the n-gram dictionary (the unigrams, bigrams and trigrams kept by the cutoff, each
/// keyed by [`dictionary_key`]) from training parses (`AbstractBottomUpParser.buildDictionary`).
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
            dict.insert(dictionary_key(&ngram));
        }
    }
    dict
}

/// The key of an n-gram in a [`build_dictionary`] set: each token's byte length, `:` and the
/// token. OpenNLP's `Dictionary` holds `StringList`s, so `1,000` and the bigram `1` `000` are
/// different entries; a plain join would make them one.
pub fn dictionary_key<S: AsRef<str>>(ngram: &[S]) -> String {
    let mut key = String::with_capacity(ngram.iter().map(|t| t.as_ref().len() + 4).sum());
    for token in ngram {
        let token = token.as_ref();
        key.push_str(&token.len().to_string());
        key.push(':');
        key.push_str(token);
    }
    key
}
