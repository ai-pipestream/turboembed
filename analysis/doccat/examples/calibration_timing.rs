//! Release timing driver: MODE (maxent|svm|names), TOKEN_COUNT, optional calibrated.
//! The name case measures decoding plus token calibration, not layer construction.
use std::{hint::black_box, sync::Arc, time::Instant};
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_doccat::*;
use turbo_maxent::ModelFile;
use turbo_sequence::{
    PatternMode,
    namefind::{MODEL_ENTRY_NAME, NameFinder},
};
use turbo_train::calibration::{Calibration, DataIdentity, Identity, Policy, Predictor, Task, Transform};
use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters, trainer};
fn maxent() -> DocumentCategorizerME {
    let events = vec![
        Event::from_strs("pos", &["bow=good"]),
        Event::from_strs("pos", &["bow=great"]),
        Event::from_strs("neg", &["bow=bad"]),
        Event::from_strs("neg", &["bow=awful"]),
    ];
    let mut params = TrainingParameters::new();
    params.put(Key::ALGORITHM, "MAXENT");
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let (trained, _) = trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut package =
        turbo_train::ModelPackage::new(turbo_train::Manifest::new("DocumentCategorizerME", "en", 1700000000000));
    package.add_entry("doccat.model", trained.write().unwrap());
    DocumentCategorizerME::new(DoccatModel::open(ModelFile::open(package.to_bytes()).unwrap()).unwrap())
}
fn svm() -> SvmCategorizer {
    let model = "svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 2\nrho 0.25\nlabel 0 1\nprobA -2.0\nprobB 0.0\nnr_sv 1 1\nSV\n1.0 1:1.0\n-1.0 2:1.0\n";
    let side = "svm-doccat 1\nlanguage\teng\nweighting\tBINARY\nselection\tNONE\t-1\nscaling\tnone\ngenerator\tbag-of-words\ncategory\t0\tpos\ncategory\t1\tneg\nfeature\t1\t1.0\t0.0\t1.0\tbow=good\nfeature\t2\t1.0\t0.0\t1.0\tbow=bad\n";
    SvmCategorizer::new(SvmDoccatModel::from_text(model, side).unwrap()).unwrap()
}
fn file() -> Arc<ModelFile> {
    let events = [
        Event::from_strs("person-start", &["w=Alice"]),
        Event::from_strs("person-start", &["w=Bob"]),
        Event::from_strs("other", &["w=walked"]),
        Event::from_strs("other", &["w=home"]),
    ];
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let (trained, _) = turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut package = ModelPackage::new(Manifest::new("TokenNameFinder", "en", 1700000000000));
    package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&trained).unwrap());
    Arc::new(ModelFile::open(package.to_bytes()).unwrap())
}

fn artifact(id: turbo_train::calibration::Identity) -> Calibration {
    Calibration::new(
        id,
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: "a".repeat(64) },
        DataIdentity { split: "conf".into(), sha256: "b".repeat(64) },
        vec![0.7; 1000],
    )
    .unwrap()
}
fn measure(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..100 {
        f();
    }
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_nanos() as f64 / n as f64
}
fn main() {
    let mode = std::env::args().nth(1).unwrap();
    let calibrated = std::env::args().nth(3).is_some_and(|s| s == "calibrated");
    let checked = std::env::args().nth(4).is_some_and(|s| s == "checked");
    let count: usize = std::env::args().nth(2).unwrap_or("128".into()).parse().unwrap();
    let text = vec!["good"; count].join(" ");
    let doc = Document::new(text.as_str());
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for i in 0..count {
        spans.push(Span::new((i * 5) as u32, (i * 5 + 4) as u32));
    }
    let tokens = Tokens { spans, sentence: None };
    let ns = match mode.as_str() {
        "maxent" => {
            let c = maxent();
            if !calibrated {
                println!(
                    "{} {} {:.2}",
                    mode,
                    count,
                    measure(
                        || {
                            let out = c.annotate((view, &tokens)).unwrap();
                            if checked {
                                out.0.check().unwrap();
                            }
                            black_box(out);
                        },
                        500000
                    )
                );
                return;
            }
            let c = {
                let a = artifact(turbo_doccat::calibrated::identity(&c, "bench").unwrap());
                turbo_doccat::calibrated::CalibratedCategorizer::new(
                    c,
                    a,
                    "bench",
                    Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 },
                )
                .unwrap()
            };
            measure(
                || {
                    let out = c.annotate((view, &tokens)).unwrap();
                    if checked {
                        out.0.check().unwrap();
                        out.1.check().unwrap();
                    }
                    black_box(out);
                },
                500000,
            )
        }
        "svm" => {
            let c = svm();
            if !calibrated {
                println!(
                    "{} {} {:.2}",
                    mode,
                    count,
                    measure(
                        || {
                            let out = c.annotate((view, &tokens)).unwrap();
                            if checked {
                                out.0.check().unwrap();
                            }
                            black_box(out);
                        },
                        500000
                    )
                );
                return;
            }
            let c = {
                let a = artifact(turbo_doccat::calibrated::identity(&c, "bench").unwrap());
                turbo_doccat::calibrated::CalibratedCategorizer::new(
                    c,
                    a,
                    "bench",
                    Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 },
                )
                .unwrap()
            };
            measure(
                || {
                    let out = c.annotate((view, &tokens)).unwrap();
                    if checked {
                        out.0.check().unwrap();
                        out.1.check().unwrap();
                    }
                    black_box(out);
                },
                500000,
            )
        }
        "names" => {
            let f = file();
            let m = f.maxent(MODEL_ENTRY_NAME).unwrap();
            let mut finder = NameFinder::new(&f, &m, PatternMode::Utf16Units).unwrap();
            let words = vec!["Alice"; count];
            let p = {
                let id = Identity {
                    model: "benchmark-namefinder".into(),
                    revision: "1".into(),
                    sha256: f.sha256().to_hex(),
                    task: Task::TokenTagging,
                    domain: "bench".into(),
                    labels: m.outcome_texts().unwrap().into_iter().map(str::to_owned).collect(),
                    inference_hash: 1,
                };
                Predictor::new(
                    artifact(id.clone()),
                    &id,
                    Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 },
                )
                .unwrap()
            };
            measure(
                || {
                    finder.clear_adaptive_data();
                    if calibrated {
                        let (names, rows) = finder.find_scored_flat(&words);
                        black_box(names);
                        let mut scores = [0.0; 2];
                        let mut set = [0; 2];
                        for row in rows.as_chunks::<2>().0 {
                            black_box(p.predict_into(row, &mut scores, &mut set).unwrap());
                        }
                    } else {
                        black_box(finder.find(&words));
                    }
                },
                3000,
            )
        }
        _ => panic!(),
    };
    println!("{mode} {count} {ns:.2}");
}
