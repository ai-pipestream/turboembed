//! Four-way Qasper boundary/retrieval ablation. No training or tuning.
//! Usage: text_signals QASPER_ARCHIVE SENTENCE_MODEL OUTPUT_JSON
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};
use turbo_analysis::{Annotator, Document, Ix, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_chunker::eval::{Segmentation, boundary};
use turbo_chunker::terms::Bm25;
use turbo_chunker::{
    Budget, Costs, DiscourseBoundaries, DiscourseLanguage, Lengths, LexicalBoundaries, StructureBoundaries,
    WeighTextBoundaries, Weight, segment,
};
use turbo_eval::lexical::{Lexical, Search, TermRule};
use turbo_eval::{metrics, read, spans};
use turbo_goldens::{hex::to_hex, sha256};
use turbo_maxent::ModelFile;
use turbo_sentdetect::{SentenceDetector, SentenceModel, WhitespaceMode};
use turbo_stem::{TermAnalyzer, TermAnnotator};

const CORPUS_SHA: &str = "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72";
const MODEL_SHA: &str = "4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8";
const BUDGET: Budget = Budget { max: 254, target: 200, size_weight: 1000, overhead: 0 };

#[derive(Default)]
struct Arm {
    texts: Vec<String>,
    doc: Vec<u32>,
    span: Vec<spans::Span>,
    boundary: Vec<Value>,
    segmentation_ms: f64,
}

/// Exactly the layout used by the Qasper reader, plus byte ranges of
/// paragraphs/headings and starts of sections. Abstract is its own segment.
fn layout(paper: &Value) -> (String, Vec<Span>, Vec<usize>) {
    let mut text = String::new();
    let mut units = Vec::new();
    let mut sections = Vec::new();
    let mut add = |part: &str, first: &mut bool| {
        if part.is_empty() {
            return;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        if *first {
            sections.push(text.len());
            *first = false;
        }
        let start = text.len();
        text.push_str(part);
        units.push(Span::new(start as u32, text.len() as u32));
    };
    add(paper["abstract"].as_str().unwrap(), &mut true);
    for s in paper["full_text"].as_array().unwrap() {
        let mut first = true;
        if let Some(name) = s["section_name"].as_str() {
            add(name, &mut first);
        }
        for p in s["paragraphs"].as_array().unwrap() {
            add(p.as_str().unwrap(), &mut first);
        }
    }
    (text, units, sections)
}

fn analyze(text: &str, units: &[Span], detector: &SentenceDetector) -> (Sentences, Tokens, Lengths) {
    let mut sentences = Spans::new(ViewId::SOURCE);
    for u in units {
        let mut detected = Vec::new();
        detector.detect(&text[u.range()], &mut detected);
        for (s, _) in detected {
            sentences.push(Span::new(s.start + u.start, s.end + u.start));
        }
    }
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut sentence = Vec::new();
    let mut lengths = Vec::new();
    let tokenizer = TermAnalyzer::builder().build();
    for (i, span) in sentences.iter() {
        let words = tokenizer.analyze(&text[span.range()]).unwrap();
        lengths.push(words.len() as u32);
        for word in words {
            let s = word.span().unwrap();
            tokens.push(Span::new(s.start + span.start, s.end + span.start));
            sentence.push(i);
        }
    }
    (Sentences { spans: sentences }, Tokens { spans: tokens, sentence: Some(sentence) }, Lengths::of(&lengths).unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: text_signals QASPER_ARCHIVE SENTENCE_MODEL OUTPUT_JSON".into());
    }
    if Path::new(&args[2]).exists() {
        return Err("output already exists".into());
    }
    let archive = std::fs::read(&args[0])?;
    if to_hex(&sha256(&archive)) != CORPUS_SHA {
        return Err("Qasper archive checksum differs".into());
    }
    let raw = read::tar::gzip_member("Qasper", &archive, "qasper-test-v0.3.json")?;
    let papers: BTreeMap<String, Value> = serde_json::from_slice(&raw)?;
    let set =
        read::qasper::read_json("qasper", "qasper-test-v0.3.json", std::str::from_utf8(&raw)?, CORPUS_SHA.into())?;
    let file = ModelFile::read(Path::new(&args[1]))?;
    if file.sha256().to_hex() != MODEL_SHA {
        return Err("sentence model checksum differs".into());
    }
    let detector = SentenceDetector::new(SentenceModel::open(file)?, WhitespaceMode::Unicode);
    let term_step = TermAnnotator::for_language("en").unwrap();
    let lexical = LexicalBoundaries::default();
    let discourse = DiscourseBoundaries { language: DiscourseLanguage::English, discount: 1000 };
    let mut arms: [Arm; 4] = std::array::from_fn(|_| Arm::default());
    let (mut sentence_count, mut token_count, mut section_count, mut marker_gaps) = (0, 0, 0, 0);
    let mut changed_cuts = [0usize; 4];
    for (di, d) in set.docs.iter().enumerate() {
        let (text, units, sections) = layout(&papers[&d.id]);
        if text != d.text {
            return Err("layout differs from Qasper reader".into());
        }
        let document = Document::new(text);
        let view = document.get::<View>().unwrap();
        let (sentences, tokens, lengths) = analyze(view.text(), &units, &detector);
        let n = sentences.spans.len();
        let mut gold: Vec<usize> = sections
            .iter()
            .map(|&at| sentences.spans.columns().0.partition_point(|&s| (s as usize) < at))
            .filter(|&s| s > 0 && s < n)
            .collect();
        gold.dedup();
        section_count += gold.len();
        let reference = Segmentation::new(n, gold)?;
        sentence_count += n;
        token_count += tokens.spans.len();
        let (forms,) = term_step.annotate((view, &tokens, None, None, None))?;
        let structure = turbo_structure::detect(view.text(), turbo_structure::Syntax::Plain, ViewId::SOURCE);
        let (base,) = StructureBoundaries { costs: Costs::default() }.annotate((&sentences, Some(&structure)))?;
        let (lex,) = lexical.annotate((view, &sentences, &tokens, &forms))?;
        let (disc,) = discourse.annotate((view, &sentences))?;
        marker_gaps += disc.cost.iter().filter(|&&c| c < 0).count();
        let mut baseline = Vec::new();
        for (ai, arm) in arms.iter_mut().enumerate() {
            let start = Instant::now();
            let step = WeighTextBoundaries {
                base: Weight::ONE,
                lexical: Weight(if ai & 1 != 0 { 1000 } else { 0 }),
                discourse: Weight(if ai & 2 != 0 { 1000 } else { 0 }),
            };
            let (cost,) = step.annotate((&sentences, &base, Some(&lex), Some(&disc)))?;
            // Word-token budget isolates these signals from any embedding model.
            // The public optimizer is used without structure's hard-limit runs.
            let parts = segment(&cost.cost, |a, b| lengths.run(a, b), BUDGET, |_, _| 0);
            arm.segmentation_ms += start.elapsed().as_secs_f64() * 1000.0;
            let cuts: Vec<usize> = parts.iter().skip(1).map(|p| p.start).collect();
            if ai == 0 {
                baseline = cuts.clone();
            }
            changed_cuts[ai] += usize::from(cuts != baseline);
            let scored = boundary::score(&reference, &Segmentation::new(n, cuts.clone())?)?;
            arm.boundary.push(json!({"id":d.id,"sentences":n,"gold_boundaries":reference.boundaries(),"cuts":cuts,
                "pk":scored.pk,"window_diff":scored.window_diff,"f1":scored.exact.f1}));
            for p in parts {
                let first = sentences.spans.get(Ix::new(p.start as u32));
                let last = sentences.spans.get(Ix::new(p.end as u32 - 1));
                arm.span.push(spans::Span::new(first.start as usize, last.end as usize)?);
                arm.doc.push(di as u32);
                arm.texts.push(view.text()[first.start as usize..last.end as usize].to_owned());
            }
        }
    }
    let lexical_retrieval = Lexical { model: Bm25::LUCENE, terms: TermRule::English, dump: None };
    let ids: Vec<_> = set.docs.iter().map(|d| d.id.clone()).collect();
    let queries: Vec<_> = set.queries.iter().map(|q| (q.id.as_str(), q.text.as_str())).collect();
    let mut records = Vec::new();
    for (ai, arm) in arms.iter().enumerate() {
        eprintln!("retrieval arm {ai}: {} chunks, {} queries", arm.texts.len(), queries.len());
        let index = lexical_retrieval.index(&arm.texts, &arm.doc)?;
        let search = Search { index: &index, doc: &arm.doc, span: &arm.span, docs: &ids };
        let (docs, chunks) = lexical_retrieval.search(&search, &queries)?;
        let doc_scores = metrics::evaluate(&docs, &set.qrels)?;
        let span_scores = spans::evaluate_spans(&chunks, set.spans.as_ref().unwrap())?;
        let avg =
            |key: &str| arm.boundary.iter().map(|b| b[key].as_f64().unwrap()).sum::<f64>() / arm.boundary.len() as f64;
        records.push(json!({"name":(["baseline","lexical","discourse","both"][ai]),
            "chunks":arm.texts.len(),"changed_documents":changed_cuts[ai],
            "pk":avg("pk"),"window_diff":avg("window_diff"),"f1":avg("f1"),
            "segmentation_ms":arm.segmentation_ms,"doc_scores":{"ndcg_at_10":doc_scores.ndcg_at_10,"recall":doc_scores.recall,
                "per_query":doc_scores.per_query.iter().map(|q|json!({"query":q.query,"ndcg_at_10":q.ndcg_at_10,"recall":q.recall})).collect::<Vec<_>>()},
            "span_scores":{"mean":span_scores.mean,"per_query":span_scores.per_query.iter().map(|q|json!({"query":q.query,"scores":q.scores})).collect::<Vec<_>>()},
            "boundary_documents":arm.boundary}));
    }
    let report = json!({"schema":1,"corpus":"Qasper v0.3 test, CC BY 4.0","corpus_archive_sha256":CORPUS_SHA,
        "corpus_member_sha256":to_hex(&sha256(&raw)),"sentence_model_sha256":MODEL_SHA,
        "sentence_abbreviations_hash":format!("{:016x}",detector.abbreviations().config_hash()),
        "term_analyzer_hash":format!("{:016x}",term_step.config_hash()),
        "lexical_hash":format!("{:016x}",lexical.config_hash()),"discourse_hash":format!("{:016x}",discourse.config_hash()),
        "budget":{"unit":"word tokens","max":254,"target":200,"size_weight":1000,"overhead":0},
        "boundary_reference":"section starts, abstract separate; silver proxy, not human topic labels",
        "sentencing":"pinned maxent detector per paragraph or heading; no sentence crosses a layout unit",
        "retrieval":"corpus-wide BM25 English, k1=1.2 b=0.75; all official questions and evidence",
        "documents":set.docs.len(),"queries":queries.len(),"sentences":sentence_count,"tokens":token_count,
        "section_boundaries":section_count,"marker_gaps":marker_gaps,"unplaced_evidence":set.unplaced.len(),"arms":records});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
