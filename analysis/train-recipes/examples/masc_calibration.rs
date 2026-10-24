//! Real-corpus calibration evaluation on MASC 3.0.0 (non-Twitter split) for:
//! 1. DocumentCategorizerME (4-class genre passages, document-disjoint splits)
//! 2. NameFinderME token-level calibration (`Task::TokenTagging`)
//! 3. CrfNameFinder span-marginal calibration (`Task::SpanTagging`, BIO and BILOU)
use std::path::Path;
use std::sync::Arc;
use turbo_analysis::{Annotator, Document, Ix, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_eval::calibration::{coverage, reliability};
use turbo_formats::Field;
use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_maxent::ModelFile;
use turbo_namefind::calibrated::{
    CalibratedCrfNameAnnotator, CalibratedNameAnnotator, crf_identity, crf_identity_with_candidates,
    identity as namefind_identity,
};
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{CrfNameFinder, MODEL_ENTRY_NAME, NameCodec, NameFinder};
use turbo_train::calibration::{Calibration, Identity, Method, Policy, Predictor, Samples, fit};
use turbo_train::crf::CrfTrainer;
use turbo_train::{Event, Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::doccat::{
    DoccatFactory, DocumentCategorizerME, doccat_calibrated, train_doccat_from_events,
};
use turbo_train_components::namefind::{TokenNameFinderFactory, train_crf_name_model, train_name_model};
use turbo_train_components::package::Stamp;
use turbo_train_recipes::{masc, namefind::read_samples, spans::SpanTypes};

fn doc_genre(doc: &MascDocument, root: &Path) -> &'static str {
    let rel = doc.path().strip_prefix(root).unwrap_or(doc.path()).to_string_lossy();
    if rel.contains("spoken/") {
        "spoken"
    } else if rel.contains("newspaper") || rel.contains("journal") || rel.contains("blog") {
        "news-press"
    } else if rel.contains("letters") || rel.contains("email") || rel.contains("spam") {
        "correspondence"
    } else {
        "prose-reference"
    }
}

fn doc_passages<'a>(docs: &'a [MascDocument], root: &Path) -> Result<Vec<(&'static str, Vec<&'a str>)>, String> {
    let mut out = Vec::new();
    for doc in docs {
        let genre = doc_genre(doc, root);
        let mut buf: Vec<&'a str> = Vec::new();
        for sent in doc.sentences() {
            let Ok(spans) = doc.token_spans(sent) else { continue };
            for span in spans {
                let s = &doc.text()[span.range()];
                if !s.is_empty() {
                    buf.push(s);
                }
            }
            if buf.len() >= 40 {
                out.push((genre, std::mem::take(&mut buf)));
            }
        }
        if buf.len() >= 10 {
            out.push((genre, buf));
        }
    }
    Ok(out)
}

fn sample_doc(s: &turbo_formats::NameSample<'_>) -> (Document, Sentences, Tokens) {
    let words: Vec<&str> = s.tokens().iter().map(Field::as_str).collect();
    let text = words.join(" ");
    let doc = Document::new(text.as_str());
    let mut sents = Sentences { spans: Spans::new(ViewId::SOURCE) };
    sents.spans.push(Span::new(0, text.len() as u32));
    let mut toks = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: Some(vec![Ix::new(0); words.len()]) };
    let mut off = 0u32;
    for w in &words {
        let end = off + w.len() as u32;
        toks.spans.push(Span::new(off, end));
        off = end + 1;
    }
    (doc, sents, toks)
}

fn print_methods(
    title: &str,
    id: &Identity,
    (fit_rows, fit_gold): (&[Vec<f64>], &[usize]),
    (conf_rows, conf_gold): (&[Vec<f64>], &[usize]),
    (test_rows, test_gold): (&[Vec<f64>], &[usize]),
) -> Result<Vec<Calibration>, String> {
    let raw = reliability(test_rows, test_gold, 10)?;
    println!(
        "\n=== {title} ===\nlabels={:?} fit_n={} conformal_n={} test_n={}",
        id.labels,
        fit_rows.len(),
        conf_rows.len(),
        test_rows.len()
    );
    println!("Raw        : ECE={:.6} Brier={:.6} accuracy={:.6}", raw.ece, raw.brier, raw.accuracy);
    let mut out = Vec::with_capacity(3);
    for (name, method) in
        [("Temperature", Method::Temperature), ("Platt      ", Method::Platt), ("Isotonic   ", Method::Isotonic)]
    {
        let cal = fit(
            id.clone(),
            method,
            Samples { split: "masc-fit", probabilities: fit_rows, gold: fit_gold },
            Samples { split: "masc-conformal", probabilities: conf_rows, gold: conf_gold },
        )?;
        let pred = Predictor::new(
            cal.clone(),
            id,
            Policy { coverage: 0.90, max_set_size: id.labels.len(), min_confidence: 0.0 },
        )?;
        let preds: Vec<_> = test_rows.iter().map(|r| pred.predict(r)).collect::<Result<_, _>>()?;
        let after = reliability(&preds.iter().map(|p| p.calibrated.clone()).collect::<Vec<_>>(), test_gold, 10)?;
        let cov = coverage(&preds, test_gold)?;
        let single_pred =
            Predictor::new(cal.clone(), id, Policy { coverage: 0.90, max_set_size: 1, min_confidence: 0.50 })?;
        let single_preds: Vec<_> = test_rows.iter().map(|r| single_pred.predict(r)).collect::<Result<_, _>>()?;
        let single_cov = coverage(&single_preds, test_gold)?;
        println!(
            "{name}: ECE={:.6} Brier={:.6} set_cov@90={:.6} (cutoff={:.6}, mean_size={:.3}) single_deliv_cov={:.6} abstain_rate={:.6} artifact={}",
            after.ece,
            after.brier,
            cov.set_coverage,
            pred.cutoff(),
            cov.mean_set_size,
            single_cov.delivered_coverage,
            single_cov.abstention_rate,
            cal.id()
        );
        out.push(cal);
    }
    Ok(out)
}

fn eval_doccat(
    train_docs: &[MascDocument],
    held_docs: &[MascDocument],
    root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let train_samples = doc_passages(train_docs, root)?;
    let mut fit_docs = Vec::new();
    let mut conf_docs = Vec::new();
    let mut test_docs = Vec::new();
    for (i, d) in held_docs.iter().cloned().enumerate() {
        match i % 3 {
            0 => fit_docs.push(d),
            1 => conf_docs.push(d),
            _ => test_docs.push(d),
        }
    }
    let fit_samples = doc_passages(&fit_docs, root)?;
    let conf_samples = doc_passages(&conf_docs, root)?;
    let test_samples = doc_passages(&test_docs, root)?;
    let events: Vec<Event> = train_samples
        .iter()
        .map(|(genre, toks)| {
            let feats: Vec<String> = toks.iter().map(|w| format!("bow={w}")).collect();
            let refs: Vec<&str> = feats.iter().map(String::as_str).collect();
            Event::from_strs(*genre, &refs)
        })
        .collect();
    let factory = DoccatFactory::default();
    let mut params = TrainingParameters::new();
    params.put(Key::ALGORITHM, "MAXENT");
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 2);
    let trained = train_doccat_from_events(
        "en",
        &events,
        &params,
        &factory,
        &mut TrainingConfiguration::default(),
        Stamp::at(0),
    )?;
    let model = trained.doccat_model(&factory)?;
    let cat = DocumentCategorizerME::new(model);
    let id = doccat_calibrated::identity(&cat, "masc-3.0.0-genres")?;
    let collect = |samples: &[(&'static str, Vec<&str>)]| -> Result<(Vec<Vec<f64>>, Vec<usize>), String> {
        let mut rows = Vec::with_capacity(samples.len());
        let mut gold = Vec::with_capacity(samples.len());
        for (genre, toks) in samples {
            let scores = cat.categorize(toks);
            let g = id.labels.iter().position(|l| l == genre).ok_or("unknown category")?;
            rows.push(scores);
            gold.push(g);
        }
        Ok((rows, gold))
    };
    let (f_r, f_g) = collect(&fit_samples)?;
    let (c_r, c_g) = collect(&conf_samples)?;
    let (t_r, t_g) = collect(&test_samples)?;
    print_methods("MASC 3.0.0 DocumentCategorizerME (4 genres)", &id, (&f_r, &f_g), (&c_r, &c_g), (&t_r, &t_g))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: masc_calibration MASC_DIRECTORY")?;
    let root = Path::new(&path);
    let digest = masc::corpus_digest(root)?;
    if !masc::RELEASE.is(&digest) {
        return Err("MASC corpus digest differs from recorded release".into());
    }
    println!("masc_corpus_sha256={} files={}", digest.sha256, digest.files);
    let split = masc::SPLIT.apply(MascCorpus::read(root)?, root)?;
    eval_doccat(split.train.documents(), split.held_out.documents(), root)?;

    let types =
        SpanTypes { types: &["person", "location", "organization", "date"], renamed: &[("org", "organization")] };
    let (mut train_all, mut held_all) = (Vec::new(), Vec::new());
    read_samples(&split.train, &types, |s| {
        train_all.push(s);
        Ok(())
    })?;
    read_samples(&split.held_out, &types, |s| {
        held_all.push(s);
        Ok(())
    })?;
    let sample_evenly = |samples: Vec<_>, limit: usize| {
        let n = samples.len();
        samples
            .into_iter()
            .enumerate()
            .filter_map(|(i, s)| (n <= limit || i * limit / n != (i + 1) * limit / n).then_some(s))
            .collect::<Vec<_>>()
    };
    let train_ner = sample_evenly(train_all, 2048);
    let held_ner = sample_evenly(held_all, 900);
    let (mut fit_ner, mut conf_ner, mut test_ner) = (Vec::new(), Vec::new(), Vec::new());
    for (i, s) in held_ner.into_iter().enumerate() {
        match i % 3 {
            0 => fit_ner.push(s),
            1 => conf_ner.push(s),
            _ => test_ner.push(s),
        }
    }

    // 1. Token-level NameFinderME
    let factory_bio = TokenNameFinderFactory::new("en", None, NameCodec::Bio)?;
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 2);
    let maxent_trained =
        train_name_model(&train_ner, &factory_bio, None, &params, &mut TrainingConfiguration::default(), Stamp::at(0))?;
    let maxent_file = Arc::new(ModelFile::open(maxent_trained.package.to_bytes())?);
    let maxent_model = maxent_file.maxent(MODEL_ENTRY_NAME)?;
    let maxent_id = namefind_identity(&maxent_file, "masc-3.0.0-ner-tokens")?;
    let maxent_labels = maxent_id.labels.clone();
    let collect_tokens = |split_sents: &[turbo_formats::NameSample<'_>]| {
        let mut finder = NameFinder::new(&maxent_file, &maxent_model, PatternMode::Utf16Units).unwrap();
        let mut rows = Vec::new();
        let mut gold = Vec::new();
        for s in split_sents {
            if s.clear_adaptive_data() {
                finder.clear_adaptive_data();
            }
            let words: Vec<&str> = s.tokens().iter().map(Field::as_str).collect();
            let spans: Vec<_> = s
                .names()
                .iter()
                .map(|n| turbo_sequence::codec::Span::new(n.start as usize, n.end as usize, n.kind))
                .collect();
            let outcomes = NameCodec::Bio.encode(&spans, words.len());
            let (_, scored) = finder.find_scored(&words);
            for (row, out) in scored.into_iter().zip(outcomes) {
                if let Some(g) = maxent_labels.iter().position(|l| l == &out) {
                    rows.push(row);
                    gold.push(g);
                }
            }
        }
        (rows, gold)
    };
    let (tf_r, tf_g) = collect_tokens(&fit_ner);
    let (tc_r, tc_g) = collect_tokens(&conf_ner);
    let (tt_r, tt_g) = collect_tokens(&test_ner);
    let token_cals = print_methods(
        "MASC 3.0.0 NameFinderME Token-Level (BIO)",
        &maxent_id,
        (&tf_r, &tf_g),
        (&tc_r, &tc_g),
        (&tt_r, &tt_g),
    )?;
    let token_policy = Policy { coverage: 0.90, max_set_size: maxent_id.labels.len(), min_confidence: 0.0 };
    for cal in token_cals {
        let ann = CalibratedNameAnnotator::new(maxent_file.clone(), cal, "masc-3.0.0-ner-tokens", token_policy)?;
        if let Some(s) = test_ner.iter().find(|s| !s.tokens().is_empty()) {
            let (doc, sents, toks) = sample_doc(s);
            let (_, side) = ann.annotate((doc.get::<View>().unwrap(), &sents, &toks))?;
            assert_eq!(side.len(), s.tokens().len());
        }
    }

    // 2. Span-level CrfNameFinder (BIO and BILOU)
    for (codec_name, codec) in [("BIO", NameCodec::Bio), ("BILOU", NameCodec::Bilou)] {
        let factory = TokenNameFinderFactory::new("en", None, codec)?;
        let crf = train_crf_name_model(
            &factory,
            &train_ner,
            &CrfTrainer { codec, cutoff: 2, l2: 1.0, iterations: 100 },
            Stamp::at(0),
        )?;
        let crf_file = Arc::new(ModelFile::open(crf.package.to_bytes())?);
        let span_labels = crf.model.span_outcomes();
        let domain = format!("masc-3.0.0-ner-spans-{}", codec_name.to_ascii_lowercase());
        let crf_id = crf_identity_with_candidates(&crf_file, &domain, 4)?;
        let collect_spans =
            |split_sents: &[turbo_formats::NameSample<'_>], k: usize| -> Result<(Vec<Vec<f64>>, Vec<usize>), String> {
                let mut finder = CrfNameFinder::from_file(&crf.model, &crf_file, PatternMode::Utf16Units)
                    .map_err(|e| e.to_string())?;
                let mut rows = Vec::new();
                let mut gold = Vec::new();
                for s in split_sents {
                    let words: Vec<&str> = s.tokens().iter().map(Field::as_str).collect();
                    let spans = if k == 1 {
                        let (names, dists) = finder.find_scored_spans(&words).map_err(|e| e.to_string())?;
                        names.into_iter().zip(dists).map(|(n, d)| (n.span, d)).collect::<Vec<_>>()
                    } else {
                        finder.candidate_spans(&words, k).map_err(|e| e.to_string())?
                    };
                    for (cand, dist) in spans {
                        let g = s
                            .names()
                            .iter()
                            .find(|n| n.start as usize == cand.start && n.end as usize == cand.end)
                            .and_then(|n| n.kind)
                            .and_then(|k| span_labels.iter().position(|l| l == k))
                            .unwrap_or(0);
                        rows.push(dist);
                        gold.push(g);
                    }
                }
                Ok((rows, gold))
            };
        let (sf_r, sf_g) = collect_spans(&fit_ner, 4)?;
        let (sc_r, sc_g) = collect_spans(&conf_ner, 4)?;
        let (st_r, st_g) = collect_spans(&test_ner, 4)?;
        let span_cals = print_methods(
            &format!("MASC 3.0.0 CrfNameFinder Span-Level ({codec_name}, k=4)"),
            &crf_id,
            (&sf_r, &sf_g),
            (&sc_r, &sc_g),
            (&st_r, &st_g),
        )?;
        let span_policy = Policy { coverage: 0.90, max_set_size: crf_id.labels.len(), min_confidence: 0.0 };
        for cal in span_cals {
            let ann = CalibratedCrfNameAnnotator::with_candidates(crf_file.clone(), cal, &domain, span_policy, 4)?;
            let mut count = 0usize;
            for s in &test_ner {
                if s.tokens().is_empty() {
                    continue;
                }
                let (doc, sents, toks) = sample_doc(s);
                let (ents, side) = ann.annotate((doc.get::<View>().unwrap(), &sents, &toks))?;
                assert_eq!(ents.len(), side.len());
                count += side.len();
            }
            assert_eq!(count, st_r.len());
        }
        let top1_domain = format!("{domain}-top1");
        let top1_id = crf_identity(&crf_file, &top1_domain)?;
        let (tf1_r, tf1_g) = collect_spans(&fit_ner, 1)?;
        let (tc1_r, tc1_g) = collect_spans(&conf_ner, 1)?;
        let (tt1_r, _) = collect_spans(&test_ner, 1)?;
        let top1_cal = fit(
            top1_id,
            Method::Temperature,
            Samples { split: "masc-fit", probabilities: &tf1_r, gold: &tf1_g },
            Samples { split: "masc-conformal", probabilities: &tc1_r, gold: &tc1_g },
        )?;
        let top1_ann = CalibratedCrfNameAnnotator::new(crf_file, top1_cal, &top1_domain, span_policy)?;
        let mut top1_count = 0usize;
        for s in &test_ner {
            if s.tokens().is_empty() {
                continue;
            }
            let (doc, sents, toks) = sample_doc(s);
            let (ents, side) = top1_ann.annotate((doc.get::<View>().unwrap(), &sents, &toks))?;
            assert_eq!(ents.len(), side.len());
            top1_count += side.len();
        }
        assert_eq!(top1_count, tt1_r.len());
    }
    Ok(())
}
