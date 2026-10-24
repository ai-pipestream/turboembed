//! Real-corpus calibration evaluation on MASC 3.0.0 (non-Twitter split) for:
//! 1. DocumentCategorizerME (4-class genre passages, document-disjoint splits)
//! 2. NameFinderME token-level calibration (`Task::TokenTagging`)
//! 3. CrfNameFinder span-marginal calibration (`Task::SpanTagging`, BIO and BILOU)
use std::path::Path;
use turbo_eval::calibration::{coverage, reliability};
use turbo_formats::Field;
use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{CrfNameFinder, MODEL_ENTRY_NAME, NameCodec, NameFinder};
use turbo_train::calibration::{Identity, Method, Policy, Predictor, Samples, Task, fit};
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

fn print_methods(
    title: &str,
    id: &Identity,
    (fit_rows, fit_gold): (&[Vec<f64>], &[usize]),
    (conf_rows, conf_gold): (&[Vec<f64>], &[usize]),
    (test_rows, test_gold): (&[Vec<f64>], &[usize]),
) -> Result<(), String> {
    let raw = reliability(test_rows, test_gold, 10)?;
    println!(
        "\n=== {title} ===\nlabels={:?} fit_n={} conformal_n={} test_n={}",
        id.labels,
        fit_rows.len(),
        conf_rows.len(),
        test_rows.len()
    );
    println!("Raw        : ECE={:.6} Brier={:.6} accuracy={:.6}", raw.ece, raw.brier, raw.accuracy);
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
    }
    Ok(())
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
    let maxent_file = ModelFile::open(maxent_trained.package.to_bytes())?;
    let maxent_model = maxent_file.maxent(MODEL_ENTRY_NAME)?;
    let maxent_labels: Vec<String> = maxent_model
        .outcome_texts()
        .map_err(|e| format!("non-UTF-8 outcome: {e:?}"))?
        .into_iter()
        .map(str::to_owned)
        .collect();
    let maxent_id = Identity {
        model: "opennlp:NameFinderME".into(),
        revision: maxent_file.opennlp_version().into(),
        sha256: maxent_file.sha256().to_hex(),
        task: Task::TokenTagging,
        domain: "masc-3.0.0-ner-tokens".into(),
        labels: maxent_labels.clone(),
        inference_hash: 0,
    };
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
    print_methods(
        "MASC 3.0.0 NameFinderME Token-Level (BIO)",
        &maxent_id,
        (&tf_r, &tf_g),
        (&tc_r, &tc_g),
        (&tt_r, &tt_g),
    )?;

    // 2. Span-level CrfNameFinder (BIO and BILOU)
    for (codec_name, codec) in [("BIO", NameCodec::Bio), ("BILOU", NameCodec::Bilou)] {
        let factory = TokenNameFinderFactory::new("en", None, codec)?;
        let crf = train_crf_name_model(
            &factory,
            &train_ner,
            &CrfTrainer { codec, cutoff: 2, l2: 1.0, iterations: 100 },
            Stamp::at(0),
        )?;
        let crf_file = ModelFile::open(crf.package.to_bytes())?;
        let span_labels = crf.model.span_outcomes();
        let crf_id = Identity {
            model: "turbo-namefind:crf".into(),
            revision: crf_file.opennlp_version().into(),
            sha256: crf_file.sha256().to_hex(),
            task: Task::SpanTagging,
            domain: format!("masc-3.0.0-ner-spans-{}", codec_name.to_ascii_lowercase()),
            labels: span_labels.clone(),
            inference_hash: 0,
        };
        let collect_spans =
            |split_sents: &[turbo_formats::NameSample<'_>]| -> Result<(Vec<Vec<f64>>, Vec<usize>), String> {
                let mut finder = CrfNameFinder::from_file(&crf.model, &crf_file, PatternMode::Utf16Units)
                    .map_err(|e| e.to_string())?;
                let mut rows = Vec::new();
                let mut gold = Vec::new();
                for s in split_sents {
                    let words: Vec<&str> = s.tokens().iter().map(Field::as_str).collect();
                    for (cand, dist) in finder.candidate_spans(&words, 4).map_err(|e| e.to_string())? {
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
        let (sf_r, sf_g) = collect_spans(&fit_ner)?;
        let (sc_r, sc_g) = collect_spans(&conf_ner)?;
        let (st_r, st_g) = collect_spans(&test_ner)?;
        print_methods(
            &format!("MASC 3.0.0 CrfNameFinder Span-Level ({codec_name})"),
            &crf_id,
            (&sf_r, &sf_g),
            (&sc_r, &sc_g),
            (&st_r, &st_g),
        )?;
    }
    Ok(())
}
