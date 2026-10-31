//! A UD treebank read as surface tokens and the syntactic words each
//! one splits into, for the token split head (`HEAD_TOKEN_SPLIT`,
//! #1698): [`crate::ud::parse`] drops a multiword token's surface line,
//! which is the very thing this head reads.
//!
//! A token's gold label is `_` when it is one word of the same form, as
//! the core's head keeps it whole ([`turbo::manifest::KEEP_TOKEN`]);
//! any other token's is one edit tree a word
//! ([`turbo::head::edit_tree::EditTree::mine`]), joined by single
//! spaces, as `turbo::head::split::parse` reads. The head's labels are
//! the ones that come up often enough in training, mined as the
//! lemmatizer mines its trees (`turbo_trainer::mine_trees`): a label
//! whose trees do not give the token's own words back is no label of
//! the head's, and a training or dev sentence holding one the head
//! lacks is left out by [`crate::run::examples`], as everywhere. The
//! test split keeps every token's gold label, the head's or not, so a
//! split it never learned counts against it.
//!
//! The score is a token's label being right, as the trainer's
//! `score_split` counts it on dev; the words a decoded label gives are
//! the core's (`TokenSplitter::words`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use turbo::head::TokenSplitter;
use turbo::head::edit_tree::EditTree;
use turbo::head::split::{MAX_SPLIT_WORDS, parse as parse_split};
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, KEEP_TOKEN, MAX_LABELS, Scheme, WordPooling};
use turbo_tagger::WordEncoder;

use crate::corpus::{CorpusRecord, Counts, FileRecord, Sentence, Task, check_pin, read_file};
use crate::run::PREDICT_CHUNK;
use crate::ud::Treebank;
use crate::{Error, Result};

/// The longest tree label a head carries (manifest rule), as the
/// trainer's `mine_trees` keeps.
const MAX_LABEL: usize = 128;

/// One sentence's surface tokens and each one's syntactic words:
/// `words[i]` splits `tokens[i]`, and holds the token itself when the
/// token is one word.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenSentence {
    pub tokens: Vec<String>,
    pub words: Vec<Vec<String>>,
}

/// The sentences of one CoNLL-U file as tokens: a `2-3` line opens a
/// token the word lines 2 and 3 then fill, an empty node (`8.1`) is
/// left out as everywhere, and every other word line is a token of one
/// word.
///
/// # Errors
///
/// Text that is not UTF-8, a row with no form column, a word whose ID
/// is not the next in order from 1, a token range that does not start
/// where the next word is due, does not span two or more words or
/// opens inside another, or a sentence that ends inside a range,
/// naming the file and line.
pub fn parse_tokens(file: &str, bytes: &[u8]) -> Result<Vec<TokenSentence>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: not UTF-8: {e}")))?;
    let mut out = Vec::new();
    let mut s = TokenSentence::default();
    // The word ID due next, and the last word ID of the multiword
    // token still collecting its words, if one is.
    let (mut next, mut open) = (1usize, None::<usize>);
    for (i, line) in text.lines().chain([""]).enumerate() {
        let at = i + 1;
        if line.starts_with('#') {
            continue;
        }
        if line.is_empty() {
            if open.is_some() {
                return Err(Error(format!("{file}:{at}: the sentence ends inside a multiword token")));
            }
            if !s.tokens.is_empty() {
                out.push(std::mem::take(&mut s));
            }
            next = 1;
            continue;
        }
        let mut fields = line.split('\t');
        let (Some(id), Some(form)) = (fields.next(), fields.next()) else {
            return Err(Error(format!("{file}:{at}: no form column")));
        };
        if id.contains('.') {
            continue;
        }
        if let Some((a, b)) = id.split_once('-') {
            let range = a.parse::<usize>().ok().zip(b.parse::<usize>().ok()).filter(|&(a, b)| b > a);
            let Some((a, b)) = range else {
                return Err(Error(format!("{file}:{at}: token ID {id:?} is not a range over two or more words")));
            };
            if open.is_some() {
                return Err(Error(format!("{file}:{at}: token ID {id:?} opens inside a multiword token")));
            }
            if a != next {
                return Err(Error(format!("{file}:{at}: token ID {id:?} where a token at {next} is due")));
            }
            s.tokens.push(form.to_owned());
            s.words.push(Vec::new());
            open = Some(b);
            continue;
        }
        if id.parse::<usize>().ok() != Some(next) {
            return Err(Error(format!("{file}:{at}: word ID {id:?} where {next} is due")));
        }
        match open {
            Some(_) => s.words.last_mut().expect("a range opens a token").push(form.to_owned()),
            None => {
                s.tokens.push(form.to_owned());
                s.words.push(vec![form.to_owned()]);
            }
        }
        next += 1;
        if open == Some(next - 1) {
            open = None;
        }
    }
    Ok(out)
}

/// A treebank's three splits as tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenSplits {
    pub train: Vec<TokenSentence>,
    pub dev: Vec<TokenSentence>,
    pub test: Vec<TokenSentence>,
}

/// Reads `bank`'s train, dev and test files under `dir` as tokens, as
/// [`crate::ud::read_treebank`] reads them as words: files that do not
/// match their pins are refused unless `unrecorded`, and the returned
/// flag says whether every file matched.
///
/// # Errors
///
/// A file that cannot be read or parsed, or a pin mismatch.
pub fn read_tokens(bank: &Treebank, dir: &Path, unrecorded: bool) -> Result<(TokenSplits, Vec<FileRecord>, bool)> {
    let (mut files, mut parts, mut mismatches) = (Vec::new(), Vec::new(), Vec::new());
    for pin in [bank.train, bank.dev, bank.test] {
        let (record, bytes) = read_file(dir, pin.path)?;
        if let Err(m) = check_pin(&record, &pin) {
            mismatches.push(m);
        }
        parts.push(parse_tokens(pin.path, &bytes)?);
        files.push(record);
    }
    if !mismatches.is_empty() && !unrecorded {
        return Err(Error(format!(
            "{} is not the recorded release (pass --unrecorded-corpus to train on it anyway): {}",
            bank.name,
            mismatches.join("; ")
        )));
    }
    let test = parts.pop().unwrap_or_default();
    let dev = parts.pop().unwrap_or_default();
    let train = parts.pop().unwrap_or_default();
    Ok((TokenSplits { train, dev, test }, files, mismatches.is_empty()))
}

/// A token's gold label: `_` for a token of one word of the same form,
/// else one mined edit tree a word, joined by single spaces. A token
/// of one word of another form, which a treebank should not hold,
/// gives the one tree's label: no label of any head's, so it only ever
/// counts wrong.
pub fn gold_label(token: &str, words: &[String]) -> String {
    match words {
        [one] if one == token => KEEP_TOKEN.to_owned(),
        _ => words.iter().map(|w| EditTree::mine(token, w).label()).collect::<Vec<_>>().join(" "),
    }
}

/// A token's label as the head may carry it: two to
/// [`MAX_SPLIT_WORDS`] words, each mined tree giving its word back on
/// this token, the joined label no longer than a label may be and one
/// the core's split parser takes. `_` is added by [`mine_labels`]
/// itself, never mined.
fn head_label(token: &str, words: &[String]) -> Option<String> {
    if words.len() < 2 || words.len() > MAX_SPLIT_WORDS {
        return None;
    }
    let trees: Vec<EditTree> = words.iter().map(|w| EditTree::mine(token, w)).collect();
    if trees.iter().zip(words).any(|(t, w)| t.apply(token).as_deref() != Some(w.as_str())) {
        return None;
    }
    let label = trees.iter().map(EditTree::label).collect::<Vec<_>>().join(" ");
    if label.len() > MAX_LABEL || parse_split(&label).is_err() {
        return None;
    }
    Some(label)
}

/// The head's labels: `_` first, then the training splits that come up
/// at least `min_freq` times, most frequent first, then by label, at
/// most the labels a head holds.
pub fn mine_labels(train: &[TokenSentence], min_freq: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for s in train {
        for (t, ws) in s.tokens.iter().zip(&s.words) {
            if let Some(l) = head_label(t, ws) {
                *counts.entry(l).or_default() += 1;
            }
        }
    }
    let mut kept: Vec<(String, usize)> = counts.into_iter().filter(|(_, c)| *c >= min_freq.max(1)).collect();
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    kept.truncate(MAX_LABELS - 1);
    let mut labels = vec![KEEP_TOKEN.to_owned()];
    labels.extend(kept.into_iter().map(|(l, _)| l));
    labels
}

/// The sentences of `part`, each token with its gold label, as the
/// receipt's tag metrics score them.
pub fn sentences(part: &[TokenSentence]) -> Vec<Sentence> {
    part.iter()
        .map(|s| Sentence {
            words: s.tokens.clone(),
            labels: s.tokens.iter().zip(&s.words).map(|(t, ws)| gold_label(t, ws)).collect(),
        })
        .collect()
}

/// The corpus record of `bank` read for the token split head, its
/// counts over tokens.
pub fn record(bank: &Treebank, splits: &TokenSplits, files: Vec<FileRecord>, recorded: bool) -> CorpusRecord {
    let counts = |p: &[TokenSentence]| Counts { sentences: p.len(), words: p.iter().map(|s| s.tokens.len()).sum() };
    CorpusRecord {
        id: format!("{}-split", bank.id),
        name: format!("{}, token split", bank.name),
        task: Task::Tags,
        licence: bank.licence.into(),
        spdx: bank.spdx.into(),
        terms: bank.terms.into(),
        attribution: bank.attribution.into(),
        recorded,
        corpus_sha256: String::new(),
        files,
        split: format!("the release's own train, dev and test files at {} ({})", bank.tag, bank.commit),
        train: counts(&splits.train),
        dev: counts(&splits.dev),
        test: counts(&splits.test),
        skipped: 0,
        recipe_only: bank.recipe_only,
    }
}

/// The token split head, `words` on `turboembed:words`, over `labels`,
/// carrying the class of the corpus's licence.
///
/// # Errors
///
/// A head the core would refuse, or a licence the zoo refuses.
pub fn spec(
    labels: Vec<String>,
    corpus: &CorpusRecord,
    pooling: WordPooling,
    bilstm: Option<(u32, u32)>,
) -> Result<HeadSpec> {
    let spec = HeadSpec {
        name: "words".into(),
        layer: "turboembed:words".into(),
        kind: HeadKind::TokenSplit,
        labels,
        scheme: Scheme::Tags,
        word_pooling: pooling,
        adapter: bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::EditTree,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        relation: None,
        license_class: Some(corpus.class()?),
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.words".into(),
    };
    spec.validate("head").map_err(|e| Error(e.message))?;
    Ok(spec)
}

/// Each sentence's token features, read `PREDICT_CHUNK` sentences at a
/// time.
///
/// # Errors
///
/// What the encoder fails on.
pub fn features(enc: &mut WordEncoder, part: &[TokenSentence], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(part.len());
    for chunk in part.chunks(PREDICT_CHUNK) {
        let tokens: Vec<Vec<&str>> = chunk.iter().map(|s| s.tokens.iter().map(String::as_str).collect()).collect();
        out.extend(enc.features(&tokens, pooling).map_err(|e| Error(e.message))?);
    }
    Ok(out)
}

/// The labels `head` gives each sentence, decoded as at inference: the
/// best-scoring label that fits each token's form.
///
/// # Errors
///
/// What the head fails on.
pub fn predict(head: &TokenSplitter, part: &[TokenSentence], features: &[Vec<f32>]) -> Result<Vec<Vec<String>>> {
    let (mut scores, mut labels, mut w) = (Vec::new(), Vec::new(), Default::default());
    let mut out = Vec::with_capacity(part.len());
    for (s, f) in part.iter().zip(features) {
        let words: Vec<u32> = (0..s.tokens.len() as u32).collect();
        let forms: Vec<&str> = s.tokens.iter().map(String::as_str).collect();
        let read = head.reads_forms().then_some(forms.as_slice());
        head.scores(f, &words, read, &mut scores, &mut w).map_err(|e| Error(e.message))?;
        head.decode(&scores, &forms, &mut labels).map_err(|e| Error(e.message))?;
        out.push(labels.iter().map(|&l| head.spec().labels[l as usize].clone()).collect());
    }
    Ok(out)
}

/// The majority baseline: each token's commonest training label, ties
/// to the smaller label, `_` for a token training never saw. Keeping a
/// token whole is the guess that is never ill-formed.
pub fn baseline(train: &[TokenSentence], test: &[TokenSentence]) -> Vec<Vec<String>> {
    let mut counts: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    for s in train {
        for (t, ws) in s.tokens.iter().zip(&s.words) {
            *counts.entry(t.as_str()).or_default().entry(gold_label(t, ws)).or_default() += 1;
        }
    }
    let by_token: BTreeMap<&str, String> = counts
        .into_iter()
        .map(|(t, ls)| {
            let best = ls
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map_or_else(|| KEEP_TOKEN.to_owned(), |(l, _)| l);
            (t, best)
        })
        .collect();
    test.iter()
        .map(|s| {
            s.tokens
                .iter()
                .map(|t| by_token.get(t.as_str()).cloned().unwrap_or_else(|| KEEP_TOKEN.to_owned()))
                .collect()
        })
        .collect()
}
