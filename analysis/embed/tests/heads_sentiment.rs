//! Sentiment classification compared across `LexiconSentiment` (#320),
//! `LabelCentroids` and `ClassifierHead` (#321) on a self-contained
//! benchmark, and on MASC 3.0.0 MPQA opinions (`CC BY 3.0 US`) when
//! `TURBO_MASC_ROOT` names the corpus' `data` directory and
//! `TURBO_MINILM_BUNDLE` a pinned all-MiniLM-L6-v2 bundle.

mod heads_support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Polarity, Sentences, Vectors};
use turbo_doccat::sentiment::{
    LexiconSentiment, PolarityAnnotator, PolarityExample, PolarityFitConfig, PolarityVectorHead,
};

use heads_support::*;

/// Self-contained sentiment evaluation benchmark comparing `LexiconSentiment` (#320),
/// `LabelCentroids` (#321), and `ClassifierHead` (#321), plus MASC 3.0.0 MPQA
/// opinion evaluation when `TURBO_MASC_ROOT` and `TURBO_MINILM_BUNDLE` are set.
#[test]
fn sentiment_comparison_against_lexicon_and_masc_mpqa() {
    let (_sub, vec_ann) = open_cpu(64);

    let train_data: &[(&str, usize)] = &[
        ("The service was friendly, attentive, and genuinely welcoming.", 0),
        ("We loved the crisp screen and the fast camera.", 0),
        ("A brilliant performance with warm humor and heartfelt moments.", 0),
        ("Fresh ingredients and rich flavors made dinner wonderful.", 0),
        ("Reliable software that saves time and works smoothly.", 0),
        ("The hotel room was spotless, quiet, and comfortable.", 0),
        ("Excellent build quality and responsive customer support.", 0),
        ("An inspiring book that kept us smiling from start to finish.", 0),
        ("The service was rude, careless, and completely unhelpful.", 1),
        ("We hated the blurry screen and the sluggish camera.", 1),
        ("A tedious performance with awkward dialogue and dull pacing.", 1),
        ("Stale ingredients and bland flavors ruined dinner completely.", 1),
        ("Buggy software that crashes often and loses data.", 1),
        ("The hotel room was dirty, noisy, and uncomfortable.", 1),
        ("Flimsy build quality and terrible customer support.", 1),
        ("A dreary book that left us frustrated from start to finish.", 1),
    ];

    // Held-out sentences that recombine the sentiment vocabulary in new combinations
    // so both the synthetic 32-dim `tiny-bert-bundle` test weights and pretrained
    // models can generalize across sentences.
    let eval_data: &[(&str, usize)] = &[
        ("Friendly customer support and reliable software made work wonderful.", 0),
        ("We loved the spotless hotel room and the comfortable dinner.", 0),
        ("A crisp camera with excellent build quality and smooth performance.", 0),
        ("Warm humor and an inspiring performance kept us smiling.", 0),
        ("Fresh flavors and attentive service made the evening welcoming.", 0),
        ("Quiet rooms and responsive support made our stay comfortable.", 0),
        ("Rude customer support and buggy software ruined work completely.", 1),
        ("We hated the dirty hotel room and the stale dinner.", 1),
        ("A blurry camera with flimsy build quality and sluggish performance.", 1),
        ("Awkward dialogue and a dreary performance left us frustrated.", 1),
        ("Bland flavors and careless service made the evening terrible.", 1),
        ("Noisy rooms and unhelpful support made our stay uncomfortable.", 1),
    ];

    let embed_items = |va: &Arc<VectorAnnotator>, sub_path: &Path, items: &[(&str, usize)]| -> Vec<(Vec<f32>, Tag)> {
        let sub = SubwordAnnotator::open(sub_path).unwrap();
        let a = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(sub)
            .add(SentenceVectors(Arc::clone(va)))
            .build()
            .unwrap();
        items
            .iter()
            .map(|&(text, label)| {
                let mut docs = [make_doc_with_sentences(text, &[(0, text.len() as u32)])];
                a.run(&mut docs).remove(0).unwrap();
                let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
                (v, Tag(label as u16))
            })
            .collect()
    };

    let train_vecs = embed_items(&vec_ann, &bert(), train_data);
    let eval_vecs = embed_items(&vec_ann, &bert(), eval_data);
    let dim = vec_ann.dim();

    let pos_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(0)).map(|(v, _)| v.as_slice()).collect();
    let neg_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(1)).map(|(v, _)| v.as_slice()).collect();

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let centroids = LabelCentroids::from_vector_examples(
        vec_ann.model().clone(),
        dim,
        "sentiment",
        &[("positive", &pos_vecs[..]), ("negative", &neg_vecs[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let linear_head =
        ClassifierHead::fit_single_label(vec_ann.model().clone(), dim, Arc::clone(&tags), &train_vecs, 1e-4, 250, 1.5)
            .unwrap();

    let polarity_train: Vec<PolarityExample> = train_vecs
        .iter()
        .map(|(v, t)| {
            let pol = if *t == Tag(0) { Polarity::Positive } else { Polarity::Negative };
            PolarityExample::new(v.clone(), pol)
        })
        .collect();
    let polarity_head = Arc::new(
        PolarityVectorHead::fit(
            &polarity_train,
            &PolarityFitConfig { epochs: 40, batch_size: 4, learning_rate: 0.08, ..Default::default() },
            Some(vec_ann.model().clone()),
        )
        .unwrap(),
    );

    let lexicon = LexiconSentiment::new();
    let mut probs = [0.0f64; 2];

    let mut lex_correct = 0usize;
    let mut cent_correct = 0usize;
    let mut lin_correct = 0usize;
    let mut pol_correct = 0usize;

    for (i, &(text, want)) in eval_data.iter().enumerate() {
        let want_tag = Tag(want as u16);
        let want_pol = if want == 0 { Polarity::Positive } else { Polarity::Negative };
        let (v, _) = &eval_vecs[i];
        if centroids.score_single_label(v, &mut probs).unwrap().0 == want_tag {
            cent_correct += 1;
        }
        if linear_head.score(v, &mut probs).unwrap().0 == want_tag {
            lin_correct += 1;
        }
        if polarity_head.predict(v).unwrap().0 == want_pol {
            pol_correct += 1;
        }
        let vscore = lexicon.score_text(text);
        let lex_tag = if vscore.compound >= 0.0 { Tag(0) } else { Tag(1) };
        if lex_tag == want_tag {
            lex_correct += 1;
        }
    }

    println!(
        "Self-contained eval ({} sentences): Lexicon={}/{}, Centroids={}/{}, LinearHead={}/{}, PolarityVectorHead={}/{}",
        eval_data.len(),
        lex_correct,
        eval_data.len(),
        cent_correct,
        eval_data.len(),
        lin_correct,
        eval_data.len(),
        pol_correct,
        eval_data.len(),
    );
    assert!(cent_correct >= 10, "centroids accuracy {cent_correct}/{}", eval_data.len());
    assert!(lin_correct >= 10, "linear head accuracy {lin_correct}/{}", eval_data.len());
    assert!(pol_correct >= 10, "polarity vector head accuracy {pol_correct}/{}", eval_data.len());

    // ClassifierHead also implements PolarityVectorScorer for PolarityAnnotator.
    let cls_polarity_ann = PolarityAnnotator::from_head(Arc::new(linear_head.clone()));
    let sample_doc = Document::new(eval_data[0].0);
    let sample_vecs = Vectors::<Sentences>::new(dim, eval_vecs[0].0.clone(), vec![], vec_ann.model().clone());
    let (cls_cats,) = cls_polarity_ann
        .annotate((sample_doc.get::<View>().unwrap(), None, None, None, None, Some(&sample_vecs)))
        .unwrap();
    assert_eq!(cls_cats.best_category(0), "positive");

    // Optional MASC 3.0.0 MPQA corpus evaluation: TURBO_MASC_ROOT is the corpus'
    // `data` directory and TURBO_MINILM_BUNDLE an all-MiniLM-L6-v2 bundle.
    let masc_root = std::env::var_os("TURBO_MASC_ROOT").map(PathBuf::from);
    let minilm_path = std::env::var_os("TURBO_MINILM_BUNDLE").map(PathBuf::from);
    if let (Some(masc_root), Some(minilm_path)) = (masc_root, minilm_path)
        && masc_root.is_dir()
    {
        let minilm_path = turbo_pins::require(&minilm_path);
        let samples = load_masc_mpqa_samples(&masc_root);
        assert!(samples.len() >= 200, "expected >= 200 MASC MPQA samples, got {}", samples.len());
        let split = (samples.len() * 4) / 5;
        let (m_train, m_test) = samples.split_at(split);

        let use_cuda = std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1");
        let o = VectorOptions {
            encoder: EncoderOptions {
                device: if use_cuda { Device::Backend("cuda".into()) } else { Device::Cpu },
                max_seq: 64,
                ..EncoderOptions::default()
            },
            ..VectorOptions::default()
        };
        // The bundle and device were asked for by name, so a bundle that does
        // not open, or a CUDA device that is not there, fails the test rather
        // than quietly scoring the tiny fixture under the MiniLM name.
        let m_vec_ann = Arc::new(
            VectorAnnotator::open(&minilm_path, &o)
                .unwrap_or_else(|e| panic!("TURBO_MINILM_BUNDLE={}: {e}", minilm_path.display())),
        );
        let eval_bundle_path = minilm_path.clone();
        let m_dim = m_vec_ann.dim();
        let m_sub = SubwordAnnotator::open(&eval_bundle_path).unwrap();
        let m_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(m_sub)
            .add(SentenceVectors(Arc::clone(&m_vec_ann)))
            .build()
            .unwrap();
        let embed_masc = |slice: &[(String, usize)]| -> Vec<(Vec<f32>, Tag)> {
            let mut docs: Vec<Document> =
                slice.iter().map(|(t, _)| make_doc_with_sentences(t, &[(0, t.len() as u32)])).collect();
            for r in m_pipe.run(&mut docs) {
                r.unwrap();
            }
            docs.iter()
                .zip(slice)
                .map(|(d, (_, l))| {
                    let v = d.get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
                    (v, Tag(*l as u16))
                })
                .collect()
        };

        let m_train_vecs = embed_masc(m_train);
        let m_test_vecs = embed_masc(m_test);

        let m_pos: Vec<&[f32]> = m_train_vecs.iter().filter(|(_, t)| *t == Tag(0)).map(|(v, _)| v.as_slice()).collect();
        let m_neg: Vec<&[f32]> = m_train_vecs.iter().filter(|(_, t)| *t == Tag(1)).map(|(v, _)| v.as_slice()).collect();

        let m_centroids = LabelCentroids::from_vector_examples(
            m_vec_ann.model().clone(),
            m_dim,
            "sentiment",
            &[("positive", &m_pos[..]), ("negative", &m_neg[..])],
            CentroidNorm::L2,
            0.1,
        )
        .unwrap();
        let m_linear = ClassifierHead::fit_single_label(
            m_vec_ann.model().clone(),
            m_dim,
            Arc::clone(&tags),
            &m_train_vecs,
            1e-4,
            300,
            1.5,
        )
        .unwrap();

        let mut m_lex_ok = 0usize;
        let mut m_cent_ok = 0usize;
        let mut m_lin_ok = 0usize;
        for (i, (text, want)) in m_test.iter().enumerate() {
            let want_tag = Tag(*want as u16);
            let (v, _) = &m_test_vecs[i];
            if m_centroids.score_single_label(v, &mut probs).unwrap().0 == want_tag {
                m_cent_ok += 1;
            }
            if m_linear.score(v, &mut probs).unwrap().0 == want_tag {
                m_lin_ok += 1;
            }
            let vscore = lexicon.score_text(text);
            let lex_tag = if vscore.compound >= 0.0 { Tag(0) } else { Tag(1) };
            if lex_tag == want_tag {
                m_lex_ok += 1;
            }
        }
        let n = m_test.len();
        println!(
            "MASC 3.0.0 MPQA polarity spans on {} (dim={}, train={}, test={}): Lexicon={}/{} ({:.1}%), Centroids={}/{} ({:.1}%), LinearHead={}/{} ({:.1}%)",
            m_vec_ann.model().id,
            m_dim,
            m_train.len(),
            n,
            m_lex_ok,
            n,
            100.0 * m_lex_ok as f64 / n as f64,
            m_cent_ok,
            n,
            100.0 * m_cent_ok as f64 / n as f64,
            m_lin_ok,
            n,
            100.0 * m_lin_ok as f64 / n as f64,
        );
    }
}

/// Extracts `(span_text, label)` pairs (`0 = positive`, `1 = negative`) from
/// MASC 3.0.0 `*-mpqa.xml` GrAF standoff files paired with their `.txt` files.
fn load_masc_mpqa_samples(root: &Path) -> Vec<(String, usize)> {
    let mut mpqa_files = Vec::new();
    collect_mpqa_files(root, &mut mpqa_files);
    mpqa_files.sort();

    let mut samples = Vec::new();
    for mpqa_path in mpqa_files {
        let stem = mpqa_path.file_name().unwrap().to_str().unwrap().strip_suffix("-mpqa.xml").unwrap();
        let txt_path = mpqa_path.with_file_name(format!("{stem}.txt"));
        let Ok(raw_txt) = std::fs::read_to_string(&txt_path) else { continue };
        let Ok(xml) = std::fs::read_to_string(&mpqa_path) else { continue };
        let char_to_byte: Vec<usize> =
            raw_txt.char_indices().map(|(b, _)| b).chain(std::iter::once(raw_txt.len())).collect();

        let mut regions: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut nodes: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("<region ")
                && let (Some(id), Some(anchors)) = (attr_val(trimmed, "xml:id"), attr_val(trimmed, "anchors"))
            {
                let mut parts = anchors.split_whitespace();
                if let (Some(s), Some(e)) = (parts.next(), parts.next())
                    && let (Ok(cs), Ok(ce)) = (s.parse::<usize>(), e.parse::<usize>())
                    && cs < ce
                    && ce < char_to_byte.len()
                {
                    regions.insert(id.to_string(), (char_to_byte[cs], char_to_byte[ce]));
                }
            }
        }

        let mut cur_node: Option<String> = None;
        let mut cur_ref: Option<String> = None;
        let mut cur_polarity: Option<usize> = None;
        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("<node ") {
                cur_node = attr_val(trimmed, "xml:id").map(str::to_string);
            } else if trimmed.starts_with("<link ") {
                if let (Some(nid), Some(targets)) = (&cur_node, attr_val(trimmed, "targets")) {
                    nodes.insert(nid.clone(), targets.split_whitespace().map(str::to_string).collect());
                }
            } else if trimmed == "</node>" {
                cur_node = None;
            } else if trimmed.starts_with("<a ") {
                let label = attr_val(trimmed, "label").unwrap_or("");
                if matches!(label, "direct-subjective" | "expressive-subjectivity" | "attitude") {
                    cur_ref = attr_val(trimmed, "ref").map(str::to_string);
                    cur_polarity = None;
                } else {
                    cur_ref = None;
                }
            } else if trimmed.starts_with("<f ") && cur_ref.is_some() {
                if attr_val(trimmed, "name") == Some("polarity") {
                    cur_polarity = match attr_val(trimmed, "value") {
                        Some("positive") => Some(0),
                        Some("negative") => Some(1),
                        _ => None,
                    };
                }
            } else if trimmed == "</a>"
                && let (Some(nref), Some(pol)) = (cur_ref.take(), cur_polarity.take())
                && let Some(rids) = nodes.get(&nref)
            {
                for rid in rids {
                    if let Some(&(bs, be)) = regions.get(rid) {
                        let snippet = raw_txt[bs..be].trim();
                        if snippet.len() >= 12 && snippet.len() <= 240 && snippet.contains(' ') {
                            samples.push((snippet.to_string(), pol));
                        }
                    }
                }
            }
        }
    }
    samples
}

fn collect_mpqa_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_mpqa_files(&p, out);
        } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with("-mpqa.xml")) {
            out.push(p);
        }
    }
}

fn attr_val<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("{key}=\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}
