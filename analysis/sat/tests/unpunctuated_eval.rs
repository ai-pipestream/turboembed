//! Routing unpunctuated scripts (`Thai`, `Lao`, `Khmer`) from
//! `RuleSentenceDetector` / `RuleSplitter` to `SatAnnotator`, and measuring
//! sentence F1 against the rule splitter on Thai, Khmer and Lao gold corpora.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Span};
use turbo_analysis_layers::Sentences;
use turbo_sat::{SatAnnotator, SatOptions, rule_detector_with_sat, rule_splitter_with_sat};
use turbo_sentdetect::{
    RuleSentenceDetector, RuleSplitter, UNPUNCTUATED_SCRIPTS, UnpunctuatedSplitter, has_unpunctuated_script,
};

fn tiny_bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-sat-bundle")
}

#[test]
fn script_detection_identifies_thai_lao_and_khmer() {
    assert_eq!(UNPUNCTUATED_SCRIPTS.len(), 3);
    assert!(has_unpunctuated_script("ภาษาไทย"));
    assert!(has_unpunctuated_script("ພາສາລາວ"));
    assert!(has_unpunctuated_script("ភាសាខ្មែរ"));
    assert!(!has_unpunctuated_script("Plain English sentences."));
    assert!(!has_unpunctuated_script("မြန်မာဘာသာ။"));
}

#[test]
fn punctuated_scripts_stay_on_rules_and_unpunctuated_scripts_route_to_sat() {
    let dir = tiny_bundle();
    let opts = SatOptions { threshold: 0.0, ..SatOptions::default() };
    let plain = RuleSentenceDetector::new("en");
    let with_sat = rule_detector_with_sat("en", Some((&dir, &opts))).unwrap();
    assert_ne!(plain.config_hash(), with_sat.config_hash());
    assert!(plain.model().is_none());
    assert!(with_sat.model().is_some());
    assert!(with_sat.execution().is_some());

    // Pure English paragraph never calls the fallback: spans match plain rules
    // even though the fallback threshold is 0.0 (which cuts after every token).
    let en_text = "Dr. Smith went to Washington. He arrived at 5 p.m. and met Mrs. Jones.";
    let mut plain_en = Vec::new();
    let mut sat_en = Vec::new();
    plain.splitter.split(en_text, &mut plain_en);
    with_sat.splitter.split(en_text, &mut sat_en);
    assert_eq!(plain_en, sat_en);
    assert_eq!(plain_en.len(), 2);

    // Unpunctuated Thai, Lao and Khmer paragraphs come out as 1 sentence under
    // plain rules, and are routed to SaT when the fallback is attached.
    for sample in [
        "การเปลี่ยนผ่านอำนาจอย่างสันตินั้นก็ไม่ใช่เรื่องใหม่ โคริ ชูลแมน เขียนลงบล็อกเมื่อวันจันทร์",
        "ປະເທດລາວມີພົນລະເມືອງປະມານເຈັດລ້ານຄົນ ນະຄອນຫຼວງແມ່ນວຽງຈັນ",
        "ព្រះរាជាណាចក្រកម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍ រាជធានីគឺភ្នំពេញ",
    ] {
        let mut rule_spans = Vec::new();
        plain.splitter.split(sample, &mut rule_spans);
        assert_eq!(rule_spans, vec![Span::new(0, sample.len() as u32)]);

        let mut fallback_spans = Vec::new();
        with_sat.splitter.split(sample, &mut fallback_spans);
        assert!(fallback_spans.len() > 1, "expected SaT fallback to split {sample:?}");
        assert_eq!(fallback_spans.first().unwrap().start, 0);
        assert_eq!(fallback_spans.last().unwrap().end, sample.len() as u32);
        for w in fallback_spans.windows(2) {
            assert!(w[0].start < w[0].end);
            assert!(w[0].end <= w[1].start);
            assert!(sample.is_char_boundary(w[0].start as usize));
            assert!(sample.is_char_boundary(w[0].end as usize));
        }
        let mut fallback_bounds = Vec::new();
        with_sat.splitter.boundaries(sample, &mut fallback_bounds);
        let expected_bounds: Vec<u32> = fallback_spans[1..].iter().map(|s| s.start).collect();
        assert_eq!(fallback_bounds, expected_bounds);
    }

    // SatAnnotator::into_rule_splitter agrees between split and boundaries on Thai text.
    let sat_th = SatAnnotator::open(&dir, &opts).unwrap().into_rule_splitter("th");
    let thai_pair = "ภาษาไทย ภาษาไทย";
    let mut pair_spans = Vec::new();
    let mut pair_bounds = Vec::new();
    sat_th.split(thai_pair, &mut pair_spans);
    sat_th.boundaries(thai_pair, &mut pair_bounds);
    assert!(pair_spans.len() > 1);
    assert_eq!(pair_bounds, pair_spans[1..].iter().map(|s| s.start).collect::<Vec<_>>());
}

#[test]
fn analyzer_batch_routes_mixed_documents_and_preserves_provenance() {
    let dir = tiny_bundle();
    let opts = SatOptions { threshold: 0.0, ..SatOptions::default() };
    let sat = SatAnnotator::open(&dir, &opts).unwrap();
    let detector = sat.into_rule_detector("th");
    let analyzer = Analyzer::builder().add(detector).build().unwrap();

    let mut docs = vec![
        Document::new("First English sentence. Second English sentence."),
        Document::new("ประโยคแรกของย่อหน้า ประโยคที่สองของย่อหน้า\n\nย่อหน้าที่สองของเอกสาร"),
        Document::new("प्रथम वाक्य। द्वितीय वाक्य।"),
    ];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let s0 = docs[0].require(&Key::<Sentences>::DEFAULT).unwrap();
    assert_eq!(s0.spans.len(), 2);
    let s1 = docs[1].require(&Key::<Sentences>::DEFAULT).unwrap();
    assert!(s1.spans.len() >= 3);
    let prov = docs[1].provenance(Key::<Sentences>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, "turbo-sentdetect:rules");
    assert!(prov.model.is_some());
    let s2 = docs[2].require(&Key::<Sentences>::DEFAULT).unwrap();
    assert_eq!(s2.spans.len(), 2);
}

#[derive(Debug)]
struct BadSplitter;

impl UnpunctuatedSplitter for BadSplitter {
    fn split_batch(&self, texts: &[&str]) -> Vec<Result<Vec<Span>, turbo_analysis::Error>> {
        texts.iter().map(|_| Ok(vec![Span::new(0, 1000)])).collect()
    }

    fn config_hash(&self) -> u64 {
        42
    }
}

#[test]
fn invalid_fallback_spans_are_refused() {
    let splitter = RuleSplitter::new("th").with_unpunctuated_fallback(Arc::new(BadSplitter));
    let mut out = Vec::new();
    assert!(splitter.try_split("ภาษาไทย", &mut out).is_err());
    let mut bounds = Vec::new();
    assert!(splitter.try_boundaries("ภาษาไทย", &mut bounds).is_err());
    assert!(rule_splitter_with_sat("th", None).unwrap().unpunctuated_fallback().is_none());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Score {
    paragraphs: usize,
    gold_spans: usize,
    pred_spans: usize,
    correct_spans: usize,
    gold_bounds: usize,
    pred_bounds: usize,
    correct_bounds: usize,
}

impl Score {
    fn span_prf(self) -> (f64, f64, f64) {
        prf(self.correct_spans, self.pred_spans, self.gold_spans)
    }

    fn bound_prf(self) -> (f64, f64, f64) {
        prf(self.correct_bounds, self.pred_bounds, self.gold_bounds)
    }
}

fn prf(correct: usize, pred: usize, gold: usize) -> (f64, f64, f64) {
    let p = if pred == 0 { 0.0 } else { correct as f64 / pred as f64 };
    let r = if gold == 0 { 0.0 } else { correct as f64 / gold as f64 };
    let f1 = if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
    (p, r, f1)
}

/// Joins sentences into paragraphs of `group` consecutive sentences separated
/// by a single space, recording each sentence's trimmed byte span and internal
/// boundary starts.
fn build_paragraphs(sentences: &[String], group: usize) -> Vec<(String, Vec<(u32, u32)>)> {
    let mut out = Vec::new();
    for chunk in sentences.chunks(group) {
        let mut text = String::new();
        let mut spans = Vec::with_capacity(chunk.len());
        for s in chunk {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(trimmed);
            let end = text.len() as u32;
            spans.push((start, end));
        }
        if !spans.is_empty() {
            out.push((text, spans));
        }
    }
    out
}

fn conllu_sentences(raw: &str) -> Vec<String> {
    raw.lines()
        .filter_map(|l| l.strip_prefix("# text = "))
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extracts sentences from `khPOS` (`word/TAG` tokens per line). When
/// `strip_khan` is true, sentence-final `។` (U+17D4 KHMER SIGN KHAN) tokens
/// are removed so the paragraph models unpunctuated Khmer prose.
fn khpos_sentences(raw: &str, strip_khan: bool) -> Vec<String> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || !line.contains("/KAN") {
            continue;
        }
        let mut sent = String::new();
        for item in line.split(' ') {
            if item.is_empty() {
                continue;
            }
            let word = item.rsplit_once('/').map_or(item, |(w, _)| w);
            if strip_khan && word.chars().all(|c| matches!(c, '\u{17D4}' | '\u{17D5}' | '.')) {
                continue;
            }
            if !sent.is_empty() {
                sent.push(' ');
            }
            for piece in word.split(['_', '~', '^']) {
                sent.push_str(piece);
            }
        }
        let trimmed = sent.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_owned());
        }
    }
    out
}

fn lao_sentences(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|line| {
            let mut s = String::new();
            for item in line.trim().split('\t') {
                let w = item.split_once('\u{200B}').map_or(item, |(w, _)| w);
                s.push_str(w);
            }
            s.trim().to_owned()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn evaluate(label: &str, detector: &RuleSentenceDetector, paragraphs: &[(String, Vec<(u32, u32)>)]) -> Score {
    let analyzer = Analyzer::builder().add(detector.clone()).build().unwrap();
    let mut docs: Vec<Document> = paragraphs.iter().map(|(t, _)| Document::new(t.clone())).collect();
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut score = Score {
        paragraphs: paragraphs.len(),
        gold_spans: 0,
        pred_spans: 0,
        correct_spans: 0,
        gold_bounds: 0,
        pred_bounds: 0,
        correct_bounds: 0,
    };
    for ((_, gold), doc) in paragraphs.iter().zip(&docs) {
        let pred_layer = doc.require(&Key::<Sentences>::DEFAULT).unwrap();
        let pred: Vec<(u32, u32)> = pred_layer.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        let gold_set: HashSet<(u32, u32)> = gold.iter().copied().collect();
        score.gold_spans += gold.len();
        score.pred_spans += pred.len();
        score.correct_spans += pred.iter().filter(|s| gold_set.contains(s)).count();

        let gold_b: HashSet<u32> = gold.iter().skip(1).map(|&(s, _)| s).collect();
        let pred_b: Vec<u32> = pred.iter().skip(1).map(|&(s, _)| s).collect();
        score.gold_bounds += gold_b.len();
        score.pred_bounds += pred_b.len();
        score.correct_bounds += pred_b.iter().filter(|b| gold_b.contains(b)).count();
    }
    let (sp, sr, sf1) = score.span_prf();
    let (bp, br, bf1) = score.bound_prf();
    eprintln!(
        "{label}: {} paragraphs | spans gold {} pred {} corr {} -> P {sp:.3} R {sr:.3} F1 {sf1:.3} | bounds gold {} pred {} corr {} -> P {bp:.3} R {br:.3} F1 {bf1:.3}",
        score.paragraphs,
        score.gold_spans,
        score.pred_spans,
        score.correct_spans,
        score.gold_bounds,
        score.pred_bounds,
        score.correct_bounds,
    );
    score
}

#[test]
#[ignore = "needs TURBO_SAT_BUNDLE, TURBO_THAI_PUD and TURBO_KHPOS (and optional TURBO_LAO_POS)"]
fn unpunctuated_scripts_against_gold() {
    let sat_dir = PathBuf::from(std::env::var("TURBO_SAT_BUNDLE").expect("TURBO_SAT_BUNDLE"));
    let thai_raw = std::fs::read_to_string(std::env::var("TURBO_THAI_PUD").expect("TURBO_THAI_PUD")).unwrap();
    let khmer_raw = std::fs::read_to_string(std::env::var("TURBO_KHPOS").expect("TURBO_KHPOS")).unwrap();

    // Multi-sentence paragraphs of 5 consecutive sentences joined by single spaces,
    // without newline breaks, so splitters must find sentence ends inside the text.
    let sat_opts = SatOptions { split_on_newlines: false, ..SatOptions::default() };

    let thai_paras = build_paragraphs(&conllu_sentences(&thai_raw), 5);
    let th_rules = RuleSentenceDetector::new("th");
    let th_sat = rule_detector_with_sat("th", Some((&sat_dir, &sat_opts))).unwrap();
    let th_rule_score = evaluate("th_pud (rules)", &th_rules, &thai_paras);
    let th_sat_score = evaluate("th_pud (rules+sat)", &th_sat, &thai_paras);
    assert!(th_sat_score.span_prf().2 > th_rule_score.span_prf().2);
    assert!(th_sat_score.bound_prf().2 > th_rule_score.bound_prf().2);

    let kh_unpunc_sents: Vec<String> = khpos_sentences(&khmer_raw, true).into_iter().take(1000).collect();
    let kh_unpunc_paras = build_paragraphs(&kh_unpunc_sents, 5);
    let km_rules = RuleSentenceDetector::new("km");
    let km_sat = rule_detector_with_sat("km", Some((&sat_dir, &sat_opts))).unwrap();
    let km_unpunc_rule = evaluate("khpos_unpunc (rules)", &km_rules, &kh_unpunc_paras);
    let km_unpunc_sat = evaluate("khpos_unpunc (rules+sat t=0.25)", &km_sat, &kh_unpunc_paras);
    assert!(km_unpunc_sat.span_prf().2 > km_unpunc_rule.span_prf().2);
    assert!(km_unpunc_sat.bound_prf().2 > km_unpunc_rule.bound_prf().2);

    let sat_opts_025 = SatOptions { threshold: 0.025, split_on_newlines: false, ..SatOptions::default() };
    let km_sat_025 = rule_detector_with_sat("km", Some((&sat_dir, &sat_opts_025))).unwrap();
    let km_unpunc_sat_025 = evaluate("khpos_unpunc (rules+sat t=0.025)", &km_sat_025, &kh_unpunc_paras);
    assert!(km_unpunc_sat_025.span_prf().2 > km_unpunc_sat.span_prf().2);
    assert!(km_unpunc_sat_025.bound_prf().2 > km_unpunc_sat.bound_prf().2);

    let kh_raw_sents: Vec<String> = khpos_sentences(&khmer_raw, false).into_iter().take(1000).collect();
    let kh_raw_paras = build_paragraphs(&kh_raw_sents, 5);
    let km_raw_rule = evaluate("khpos_raw (rules)", &km_rules, &kh_raw_paras);
    let km_raw_sat = evaluate("khpos_raw (rules+sat)", &km_sat, &kh_raw_paras);
    assert_eq!(km_raw_sat.correct_spans, km_raw_rule.correct_spans);
    assert_eq!(km_raw_sat.correct_bounds, km_raw_rule.correct_bounds);
    assert!(km_raw_sat.span_prf().2 > 0.85);
    assert!(km_raw_sat.bound_prf().2 > 0.90);

    if let Ok(lao_path) = std::env::var("TURBO_LAO_POS") {
        let lao_raw = std::fs::read_to_string(lao_path).unwrap();
        let lao_sents: Vec<String> = lao_sentences(&lao_raw).into_iter().take(1000).collect();
        let lao_paras = build_paragraphs(&lao_sents, 5);
        let lo_rules = RuleSentenceDetector::new("lo");
        let lo_sat = rule_detector_with_sat("lo", Some((&sat_dir, &sat_opts))).unwrap();
        let lo_rule_score = evaluate("lao_pos (rules)", &lo_rules, &lao_paras);
        let lo_sat_score = evaluate("lao_pos (rules+sat)", &lo_sat, &lao_paras);
        assert!(lo_sat_score.span_prf().2 > lo_rule_score.span_prf().2);
        assert!(lo_sat_score.bound_prf().2 > lo_rule_score.bound_prf().2);
    }
}
