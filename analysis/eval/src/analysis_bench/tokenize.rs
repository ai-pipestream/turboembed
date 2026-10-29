//! Our tokenizer tiers for the analysis benchmark (#1645): the words
//! `turbo-tokenize` makes of each gold document's joined `# text`,
//! scored against the gold words by the same form alignment the
//! CoNLL-U predictions get. A tier writes no sentences and no fields,
//! so only its `tokens` score is real.

use std::time::Instant;

use turbo_analysis::Span;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::uax29::word_boundaries;
use turbo_tokenize::whitespace::whitespace_spans;
use turbo_unicode::whitespace::is_whitespace;

use super::align::{align_forms, flatten, joined_documents};
use super::{ConlluSentence, ConlluToken, MetricScore, SystemScores};
use crate::Result;
use crate::model_gate::{Counts, Measure};

/// A word-segmentation tier that needs no model: UAX #29, or the
/// whitespace baseline a word segmenter is judged against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerTier {
    /// UAX #29 word segments, whitespace-only segments dropped, so
    /// punctuation segments stand as words, as UD counts them.
    Uax29,
    /// Runs without a Unicode `White_Space` character.
    Whitespace,
}

impl TokenizerTier {
    /// The tier `--tokenize <name>` names.
    pub fn parse(name: &str) -> Result<TokenizerTier> {
        match name {
            "uax29" => Ok(TokenizerTier::Uax29),
            "whitespace" => Ok(TokenizerTier::Whitespace),
            other => Err(format!("--tokenize {other}: uax29 or whitespace")),
        }
    }

    /// The record's system name for the tier.
    pub fn system_name(self) -> &'static str {
        match self {
            TokenizerTier::Uax29 => "turbo-uax29",
            TokenizerTier::Whitespace => "turbo-whitespace",
        }
    }

    /// The tier's token forms of `text`, in order.
    pub fn forms(self, text: &str) -> Vec<String> {
        match self {
            TokenizerTier::Uax29 => {
                let mut bounds = Vec::new();
                word_boundaries(text, &mut bounds);
                bounds
                    .windows(2)
                    .map(|w| &text[w[0] as usize..w[1] as usize])
                    .filter(|segment| !segment.chars().all(|c| is_whitespace(c as u32)))
                    .map(str::to_owned)
                    .collect()
            }
            TokenizerTier::Whitespace => {
                let mut spans = Vec::new();
                whitespace_spans(text, TokenizerOptions::default(), &mut spans);
                spans.iter().map(|s: &Span| text[s.start as usize..s.end as usize].to_owned()).collect()
            }
        }
    }
}

/// One gold document for a tokenizer tier: the text the tier sees and
/// the gold words over it, by sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenDocument {
    pub text: String,
    pub sentences: Vec<Vec<String>>,
}

/// Scores a tokenizer tier over each document's joined `# text` lines:
/// the predicted tokens are one stream per document, form-aligned to
/// the gold words, so `tokens` is word-segmentation F1 with gold
/// denominators, as in [`super::score_conllu`]. Bootstrap units are
/// gold documents.
pub fn score_tokenizer(
    system: &str,
    gold: &[ConlluSentence],
    tier: TokenizerTier,
    resamples: u32,
) -> Result<SystemScores> {
    let joined = joined_documents(gold)?;
    let mut documents: Vec<TokenDocument> = Vec::with_capacity(joined.len());
    for (text, _) in joined {
        documents.push(TokenDocument { text, sentences: Vec::new() });
    }
    // Group the gold forms exactly as joined_documents grouped texts.
    let mut at = 0usize;
    for (i, sentence) in gold.iter().enumerate() {
        if sentence.new_doc && i > 0 {
            at += 1;
        }
        documents[at].sentences.push(sentence.tokens.iter().map(|t| t.form.clone()).collect());
    }
    score_tokenizer_documents(system, &documents, tier, resamples, gold.len())
}

/// Scores a tokenizer tier over gold documents of any source corpus:
/// `sentences` of each is only its gold count, for the throughput
/// field. Gold words are the concatenated sentence forms.
pub fn score_tokenizer_documents(
    system: &str,
    documents: &[TokenDocument],
    tier: TokenizerTier,
    resamples: u32,
    sentences: usize,
) -> Result<SystemScores> {
    fn sentence(forms: Vec<String>, new_doc: bool) -> ConlluSentence {
        ConlluSentence {
            text: None,
            tokens: forms
                .into_iter()
                .map(|form| ConlluToken {
                    form,
                    lemma: "_".into(),
                    upos: "_".into(),
                    xpos: "_".into(),
                    feats: "_".into(),
                    head: None,
                    deprel: "_".into(),
                })
                .collect(),
            new_doc,
        }
    }
    let started = Instant::now();
    let mut predicted: Vec<ConlluSentence> = Vec::with_capacity(documents.len());
    for document in documents {
        predicted.push(sentence(tier.forms(&document.text), true));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let mut gold_sentences: Vec<ConlluSentence> = Vec::new();
    for document in documents {
        for (i, forms) in document.sentences.iter().enumerate() {
            gold_sentences.push(sentence(forms.clone(), i == 0));
        }
    }
    let gold = &gold_sentences[..];

    let gold_all = flatten(gold);
    let (gold_flat, gold_docs, gold_sents) = (&gold_all.tokens, &gold_all.documents, &gold_all.sentences);
    let pred_all = flatten(&predicted);
    let pred_flat = &pred_all.tokens;
    let pairs = align_forms(gold_flat, pred_flat);
    let doc_of_gold = |g: usize| gold_docs.partition_point(|&(_, end)| end <= g);
    // Predicted tokens join the gold document of the nearest preceding
    // aligned pair, as in score_conllu.
    let mut gold_aligned = vec![false; gold_flat.len()];
    let mut units: Vec<Counts> = vec![Counts::default(); gold_docs.len()];
    for (doc, &(start, end)) in gold_docs.iter().enumerate() {
        units[doc].target = (end - start) as u64;
    }
    let mut current = 0usize;
    let mut pair_at = vec![usize::MAX; pred_flat.len()];
    for &(g, p) in &pairs {
        pair_at[p] = g;
    }
    for &g in &pair_at {
        if g != usize::MAX {
            current = doc_of_gold(g);
        }
        units[current].selected += 1;
    }
    for &(g, _) in &pairs {
        gold_aligned[g] = true;
        units[doc_of_gold(g)].true_positives += 1;
    }
    let mismatch = gold_sents.iter().filter(|&&(s, e)| (s..e).any(|g| !gold_aligned[g])).count();
    let pooled = units.iter().fold(Counts::default(), |mut total, u| {
        total.true_positives += u.true_positives;
        total.selected += u.selected;
        total.target += u.target;
        total
    });
    Ok(SystemScores {
        system: system.to_string(),
        sentences: None,
        tokens: Some(MetricScore::of(pooled, Measure::FMeasure, Some(&units), resamples)),
        upos: None,
        xpos: None,
        feats: None,
        lemma: None,
        uas: None,
        las: None,
        entities: None,
        language: None,
        not_run: vec!["sentences/fields: a tokenizer tier writes no sentence or field predictions".into()],
        token_mismatch_sentences: mismatch,
        throughput_sentences_per_second: (elapsed > 0.0).then_some(sentences as f64 / elapsed),
    })
}
