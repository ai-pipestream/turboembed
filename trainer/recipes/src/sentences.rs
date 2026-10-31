//! Sentence boundary documents on MASC 3.0.0 and UD treebanks (#1725):
//! multi-sentence documents with their original spacing (each CoNLL-U
//! sentence's required `# text`), exact-offset sentence span
//! scoring, and the newline and rule-based sentence splitter baselines.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::Path;

use turbo::head::tagger::Work;
use turbo::head::{SentenceBoundary, boundary};
use turbo_analysis::Span;
use turbo_formats::conllu::{ConlluReader, misc_trailing_space};
use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_sentdetect::{RuleSplitter, newline_spans};
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, check_corpus};
use turbo_train_recipes::postag::read_samples;
use turbo_trainer::rng::Rng;

use crate::corpus::{CorpusRecord, Counts, Sentence, Task, check_pin, read_file};
use crate::eval::{Comparison, Metrics, SpanScore};
use crate::masc_pos::{DEV_EVERY, SPDX as MASC_SPDX};
use crate::ud::{TREEBANKS, Treebank};
use crate::{Error, Result};

/// Maximum sentences grouped into one document when a CoNLL-U split has
/// no `# newdoc` markers, and the maximum sentences in one training
/// passage sliced from a longer document.
pub const SENTENCES_PER_DOCUMENT: usize = 10;

/// One multi-sentence document: its rendered text, surface token strings,
/// token byte spans in `text`, gold sentence byte spans in `text`, and
/// the token index range `start..end` of each gold sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub text: String,
    pub words: Vec<String>,
    pub word_spans: Vec<Span>,
    pub sentences: Vec<Span>,
    pub sentence_words: Vec<Range<usize>>,
}

impl Document {
    /// The document as a single `SentenceBoundary` sequence, where the
    /// last token of each gold sentence carries `"last"` and every other
    /// token carries `"inside"`.
    pub fn to_sentence(&self) -> Sentence {
        let mut labels = vec!["inside".to_owned(); self.words.len()];
        for r in &self.sentence_words {
            if r.end > r.start {
                labels[r.end - 1] = "last".to_owned();
            }
        }
        Sentence { words: self.words.clone(), labels }
    }

    /// Slices the document into multi-sentence training passages of at
    /// most `max_sentences` consecutive sentences without crossing a
    /// sentence or document boundary.
    pub fn train_passages(&self, max_sentences: usize) -> Vec<Sentence> {
        let step = max_sentences.max(1);
        self.sentence_words
            .chunks(step)
            .filter_map(|chunk| {
                let start = chunk.first()?.start;
                let end = chunk.last()?.end;
                let words = self.words[start..end].to_vec();
                let mut labels = vec!["inside".to_owned(); words.len()];
                for r in chunk {
                    labels[r.end - 1 - start] = "last".to_owned();
                }
                Some(Sentence { words, labels })
            })
            .collect()
    }
}

/// A corpus of multi-sentence documents read and checked.
#[derive(Debug, Clone)]
pub struct SentenceCorpus {
    pub record: CorpusRecord,
    pub language: &'static str,
    pub train: Vec<Document>,
    pub dev: Vec<Document>,
    pub test: Vec<Document>,
}

fn counts(docs: &[Document]) -> Counts {
    Counts { sentences: docs.iter().map(|d| d.sentences.len()).sum(), words: docs.iter().map(|d| d.words.len()).sum() }
}

fn bank_language(bank: &Treebank) -> &'static str {
    bank.id.strip_prefix("ud-").and_then(|s| s.split('-').next()).unwrap_or("en")
}

/// Reads a sentence boundary corpus (`masc-pos` / `masc-sentences` or a
/// UD treebank in [`TREEBANKS`]).
pub fn read(corpus_id: &str, dir: &Path, allow_unrecorded: bool) -> Result<SentenceCorpus> {
    if matches!(corpus_id, "masc-pos" | "masc" | "masc-sentences") {
        return read_masc(dir, allow_unrecorded);
    }
    let bank_id = corpus_id.strip_suffix("-sentences").unwrap_or(corpus_id);
    if let Some(bank) = TREEBANKS.iter().find(|b| b.id == bank_id) {
        return read_ud(bank, dir, allow_unrecorded);
    }
    Err(Error(format!("unknown sentence corpus {corpus_id:?} (expected masc-pos or a UD treebank such as ud-en-ewt)")))
}

fn read_masc(dir: &Path, allow_unrecorded: bool) -> Result<SentenceCorpus> {
    let digest = check_corpus(dir, RELEASE, allow_unrecorded).map_err(|e| Error(format!("MASC: {e}")))?;
    let recorded = RELEASE.is(&digest);
    let corpus = MascCorpus::read(dir).map_err(|e| Error(format!("MASC: {e}")))?;
    let split = SPLIT.apply(corpus, dir).map_err(|e| Error(format!("MASC: {e}")))?;

    let (mut train, mut dev, mut test) = (Vec::new(), Vec::new(), Vec::new());
    let mut skipped = 0;
    for (i, doc) in split.train.into_documents().into_iter().enumerate() {
        let dst = if i % DEV_EVERY == DEV_EVERY - 1 { &mut dev } else { &mut train };
        if let Some(d) = masc_document(doc, &mut skipped)? {
            dst.push(d);
        }
    }
    for doc in split.held_out.into_documents() {
        if let Some(d) = masc_document(doc, &mut skipped)? {
            test.push(d);
        }
    }
    let record = CorpusRecord {
        id: "masc-sentences".into(),
        name: masc::NAME.into(),
        task: Task::Tags,
        licence: "CC BY 3.0 US".into(),
        spdx: MASC_SPDX.into(),
        terms: masc::TERMS.into(),
        attribution: masc::ATTRIBUTION.into(),
        recorded,
        corpus_sha256: digest.sha256,
        files: Vec::new(),
        split: format!(
            "test: documents at {} modulo {} with {} left out (the recipes' split); dev: training-side \
             documents at {} modulo {DEV_EVERY}",
            SPLIT.held_out_at(),
            SPLIT.every(),
            masc::EXCLUSIONS.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            DEV_EVERY - 1
        ),
        train: counts(&train),
        dev: counts(&dev),
        test: counts(&test),
        skipped,
        recipe_only: false,
    };
    Ok(SentenceCorpus { record, language: "en", train, dev, test })
}

fn masc_document(doc: MascDocument, skipped: &mut usize) -> Result<Option<Document>> {
    let raw = doc.text().to_owned();
    let one = MascCorpus::from_documents(vec![doc]);
    let mut text = String::new();
    let mut words = Vec::new();
    let mut word_spans = Vec::new();
    let mut sentences = Vec::new();
    let mut sentence_words = Vec::new();
    let mut prev_sent_raw_end: Option<usize> = None;

    let sk = read_samples(&one, |sample| {
        let tokens = sample.tokens();
        let first_raw = tokens[0].span.start as usize;
        if let Some(prev_end) = prev_sent_raw_end {
            let gap = (prev_end <= first_raw).then(|| raw.get(prev_end..first_raw)).flatten();
            match gap {
                Some(g) if g.chars().all(char::is_whitespace) => text.push_str(g),
                Some(g) if g.contains(['\n', '\r']) => text.push('\n'),
                _ => text.push(' '),
            }
        }
        let sent_first_word = words.len();
        let mut raw_cursor = first_raw;
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                let tok_start = tok.span.start as usize;
                let gap = (raw_cursor <= tok_start).then(|| raw.get(raw_cursor..tok_start)).flatten();
                match gap {
                    Some(g) if g.chars().all(char::is_whitespace) => text.push_str(g),
                    _ => text.push(' '),
                }
            }
            let form = {
                let trimmed = tok.as_str().trim();
                if trimmed.is_empty() { tok.as_str() } else { trimmed }
            };
            let start = u32::try_from(text.len()).expect("MASC document fits in u32");
            text.push_str(form);
            let end = u32::try_from(text.len()).expect("MASC document fits in u32");
            words.push(form.to_owned());
            word_spans.push(Span::new(start, end));
            raw_cursor = raw_cursor.max(tok.span.end as usize);
        }
        let sent_last_word = words.len();
        sentences.push(Span::new(word_spans[sent_first_word].start, word_spans[sent_last_word - 1].end));
        sentence_words.push(sent_first_word..sent_last_word);
        prev_sent_raw_end = Some(raw_cursor);
    })
    .map_err(|e| Error(format!("MASC: {e}")))?;
    *skipped += sk;
    if sentences.is_empty() {
        return Ok(None);
    }
    Ok(Some(Document { text, words, word_spans, sentences, sentence_words }))
}

struct UdSent {
    new_document: bool,
    new_paragraph: bool,
    text: String,
    tokens: Vec<(String, Span)>,
    trailing_sep: String,
}

fn build_ud_sentence(name: &str, s: &turbo_formats::conllu::Sentence<'_>) -> Result<Option<UdSent>> {
    let words: Vec<_> = s.word_lines.iter().filter(|w| !w.id.as_str().contains('.')).collect();
    if words.is_empty() {
        return Ok(None);
    }
    let trailing_sep = misc_trailing_space(words.last().expect("non-empty").misc.as_str());
    let sid = s.sentence_id.as_ref().map_or("<unnamed>", |f| f.as_str());
    let raw = s
        .text
        .as_ref()
        .ok_or_else(|| Error(format!("{name}:{}: sentence {sid:?} has no # text, the source of its offsets", s.line)))?
        .as_str();
    let (text, tokens) = {
        let (mut cursor, mut first_start) = (0usize, 0usize);
        let mut spans = Vec::with_capacity(words.len());
        for (idx, w) in words.iter().enumerate() {
            let trimmed = w.form.as_str().trim();
            let form = if trimmed.is_empty() { w.form.as_str() } else { trimmed };
            let ws_len: usize = raw[cursor..].chars().take_while(|c| c.is_whitespace()).map(char::len_utf8).sum();
            let tok_start = cursor + ws_len;
            if !raw[tok_start..].starts_with(form) {
                return Err(Error(format!(
                    "{name}:{}: token {form:?} in sentence {sid:?} does not align with # text = {raw:?} at byte {tok_start}",
                    w.line
                )));
            }
            let tok_end = tok_start + form.len();
            if idx == 0 {
                first_start = tok_start;
            }
            spans.push((form.to_owned(), tok_start, tok_end));
            cursor = tok_end;
        }
        if !raw[cursor..].chars().all(char::is_whitespace) {
            return Err(Error(format!(
                "{name}:{}: trailing text {:?} in sentence {sid:?} after last token in # text = {raw:?}",
                s.line,
                &raw[cursor..]
            )));
        }
        let sent_text = raw[first_start..cursor].to_owned();
        let tokens = spans
            .into_iter()
            .map(|(form, a, b)| {
                let start = u32::try_from(a - first_start).expect("sentence fits in u32");
                let end = u32::try_from(b - first_start).expect("sentence fits in u32");
                (form, Span::new(start, end))
            })
            .collect();
        (sent_text, tokens)
    };
    Ok(Some(UdSent { new_document: s.new_document, new_paragraph: s.new_paragraph, text, tokens, trailing_sep }))
}

fn parse_ud_sentences(name: &str, bytes: &[u8]) -> Result<(Vec<UdSent>, usize)> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{name}: UTF-8: {e}")))?;
    let reader = ConlluReader::new(text).map_err(|e| Error(format!("{name}: {e}")))?;
    let mut out = Vec::new();
    let mut skipped = 0;
    for s in reader {
        let s = s.map_err(|e| Error(format!("{name}: {e}")))?;
        match build_ud_sentence(name, &s)? {
            Some(sent) => out.push(sent),
            None => skipped += 1,
        }
    }
    Ok((out, skipped))
}

fn ud_documents(sents: &[UdSent]) -> Result<Vec<Document>> {
    if sents.is_empty() {
        return Ok(Vec::new());
    }
    let has_newdoc = sents.iter().skip(1).any(|s| s.new_document);
    let mut groups: Vec<&[UdSent]> = Vec::new();
    if has_newdoc {
        let mut start = 0;
        for i in 1..sents.len() {
            if sents[i].new_document {
                groups.push(&sents[start..i]);
                start = i;
            }
        }
        groups.push(&sents[start..]);
    } else {
        groups.extend(sents.chunks(SENTENCES_PER_DOCUMENT));
    }

    let mut docs = Vec::with_capacity(groups.len());
    for group in groups {
        let mut text = String::new();
        let mut words = Vec::new();
        let mut word_spans = Vec::new();
        let mut sentences = Vec::new();
        let mut sentence_words = Vec::new();
        for (s_idx, s) in group.iter().enumerate() {
            if s_idx > 0 {
                let sep = &group[s_idx - 1].trailing_sep;
                if sep == " " && s.new_paragraph {
                    text.push('\n');
                } else {
                    text.push_str(sep);
                }
            }
            let sent_start = u32::try_from(text.len()).map_err(|_| Error("UD document exceeds u32".into()))?;
            text.push_str(&s.text);
            let sent_end = u32::try_from(text.len()).map_err(|_| Error("UD document exceeds u32".into()))?;
            let first_w = words.len();
            for (form, rel_span) in &s.tokens {
                words.push(form.clone());
                word_spans.push(Span::new(sent_start + rel_span.start, sent_start + rel_span.end));
            }
            let last_w = words.len();
            sentences.push(Span::new(sent_start, sent_end));
            sentence_words.push(first_w..last_w);
        }
        docs.push(Document { text, words, word_spans, sentences, sentence_words });
    }
    Ok(docs)
}

fn read_ud(bank: &Treebank, dir: &Path, allow_unrecorded: bool) -> Result<SentenceCorpus> {
    let mut files = Vec::new();
    let mut recorded = true;
    let mut bad = Vec::new();
    let mut split_docs = Vec::new();
    let mut skipped = 0;
    for pin in [bank.train, bank.dev, bank.test] {
        let (f, bytes) = read_file(dir, pin.path)?;
        if let Err(m) = check_pin(&f, &pin) {
            recorded = false;
            bad.push(m);
        }
        let (sents, sk) = parse_ud_sentences(pin.path, &bytes)?;
        skipped += sk;
        split_docs.push(ud_documents(&sents)?);
        files.push(f);
    }
    if !recorded && !allow_unrecorded {
        return Err(Error(format!("corpus {} is not the recorded release:\n  {}", bank.id, bad.join("\n  "))));
    }
    let [train, dev, test]: [Vec<Document>; 3] = split_docs.try_into().unwrap();
    let record = CorpusRecord {
        id: format!("{}-sentences", bank.id),
        name: format!("{} ({} {})", bank.name, bank.tag, &bank.commit[..8]),
        task: Task::Tags,
        licence: bank.licence.into(),
        spdx: bank.spdx.into(),
        terms: bank.terms.into(),
        attribution: bank.attribution.into(),
        recorded,
        corpus_sha256: String::new(),
        files,
        split: format!("official {} train, dev and test files", bank.name),
        train: counts(&train),
        dev: counts(&dev),
        test: counts(&test),
        skipped,
        recipe_only: bank.recipe_only,
    };
    Ok(SentenceCorpus { record, language: bank_language(bank), train, dev, test })
}

/// Predicts sentence byte spans on `docs` using `head` and per-document
/// word `features`.
pub fn predict(head: &SentenceBoundary, docs: &[Document], features: &[Vec<f32>]) -> Result<Vec<Vec<Span>>> {
    let mut scores = Vec::new();
    let mut ends = Vec::new();
    let mut work = Work::default();
    let mut out = Vec::with_capacity(docs.len());
    for (doc, feat) in docs.iter().zip(features) {
        let words: Vec<u32> = (0..doc.words.len() as u32).collect();
        let forms: Vec<&str> = doc.words.iter().map(String::as_str).collect();
        head.scores(feat, &words, head.reads_forms().then_some(&forms), &mut scores, &mut work)
            .map_err(|e| Error(e.to_string()))?;
        head.decode(&scores, doc.words.len(), &mut ends).map_err(|e| Error(e.to_string()))?;
        let ranges = boundary::sentences(&ends).map_err(|e| Error(e.to_string()))?;
        let spans = ranges
            .into_iter()
            .map(|r| Span::new(doc.word_spans[r.start].start, doc.word_spans[r.end - 1].end))
            .collect();
        out.push(spans);
    }
    Ok(out)
}

/// Predicts sentence byte spans on `docs` with the newline baseline (one
/// sentence per non-blank line).
pub fn newline_baseline(docs: &[Document]) -> Vec<Vec<Span>> {
    docs.iter()
        .map(|d| {
            let mut spans = Vec::new();
            newline_spans(&d.text, &mut spans);
            spans
        })
        .collect()
}

/// Predicts sentence byte spans on `docs` with the rule/abbreviation
/// sentencer in `analysis/sentdetect` for `language`.
pub fn rule_baseline(language: &str, docs: &[Document]) -> Vec<Vec<Span>> {
    let splitter = RuleSplitter::new(language);
    docs.iter()
        .map(|d| {
            let mut spans = Vec::new();
            splitter.split(&d.text, &mut spans);
            spans
        })
        .collect()
}

/// Scores predicted sentence spans against `docs` by exact byte offset
/// match.
pub fn evaluate_spans(docs: &[Document], pred: &[Vec<Span>]) -> Metrics {
    let mut all = SpanScore::default();
    for (doc, got) in docs.iter().zip(pred) {
        let want: BTreeSet<Span> = doc.sentences.iter().copied().collect();
        let correct = got.iter().filter(|s| want.contains(s)).count();
        all.add(want.len(), got.len(), correct);
    }
    let finished = all.finish();
    let mut per_type = BTreeMap::new();
    per_type.insert("sentence".to_owned(), finished);
    Metrics::Spans { all: finished, per_type, coarse: BTreeMap::new() }
}

/// Paired bootstrap of `model` against `baseline` over `docs`, scored by
/// micro sentence span F1.
pub fn bootstrap_spans(
    docs: &[Document],
    model: &[Vec<Span>],
    baseline: &[Vec<Span>],
    resamples: usize,
    seed: u64,
) -> Comparison {
    let tallies = |pred: &[Vec<Span>]| -> Vec<(usize, usize, usize)> {
        docs.iter()
            .zip(pred)
            .map(|(d, got)| {
                let want: BTreeSet<Span> = d.sentences.iter().copied().collect();
                let correct = got.iter().filter(|s| want.contains(s)).count();
                (want.len(), got.len(), correct)
            })
            .collect()
    };
    let f1 = |t: &[(usize, usize, usize)], idx: &dyn Fn(usize) -> usize| -> f64 {
        let mut s = SpanScore::default();
        for i in 0..t.len() {
            let (g, f, c) = t[idx(i)];
            s.add(g, f, c);
        }
        s.finish().f1
    };
    let (m, b) = (tallies(model), tallies(baseline));
    let (ms, bs) = (f1(&m, &|i| i), f1(&b, &|i| i));
    let mut rng = Rng::new(seed);
    let mut deltas: Vec<f64> = (0..resamples)
        .map(|_| {
            let pick: Vec<usize> = (0..docs.len()).map(|_| rng.below(docs.len())).collect();
            f1(&m, &|i| pick[i]) - f1(&b, &|i| pick[i])
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let at = |q: f64| if deltas.is_empty() { 0.0 } else { deltas[((deltas.len() - 1) as f64 * q).round() as usize] };
    Comparison { model: ms, baseline: bs, delta: ms - bs, low: at(0.025), high: at(0.975), resamples, seed }
}
