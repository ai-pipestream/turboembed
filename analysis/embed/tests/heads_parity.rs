//! CPU and device (`TURBO_TEST_DEVICE` / `TURBO_TEST_REQUIRE_CUDA=1`)
//! parity for `LabelCentroids` and `ClassifierHead` (#321).

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Categories, Opinion, Opinions, Sentences, Tokens};
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[test]
fn heads_and_centroids_cpu_and_device_parity() {
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => w,
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => "cuda".to_string(),
        Err(_) => {
            println!("skipped: neither TURBO_TEST_DEVICE nor TURBO_TEST_REQUIRE_CUDA=1 is set");
            return;
        }
    };
    let device = match want.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(want.clone()),
    };
    let (sub, cpu_vec) = open_cpu(64);
    let dev_opts = VectorOptions {
        encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let dev_vec =
        Arc::new(VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")));
    let exec = &dev_vec.info().execution;
    assert_ne!(exec.backend, "cpu", "{exec:?}");

    let pos_examples = [
        "We loved the friendly staff and the delicious breakfast.",
        "This camera takes crisp photos and works wonderfully.",
    ];
    let neg_examples =
        ["We hated the rude service and the cold soggy meal.", "This camera crashes constantly and wastes battery."];
    let centroids = LabelCentroids::from_text_examples(
        &cpu_vec,
        &sub,
        "sentiment",
        &[("positive", &pos_examples[..]), ("negative", &neg_examples[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let head = ClassifierHead::fit_single_label(
        cpu_vec.model().clone(),
        32,
        tags,
        &[(centroids.centroid_of(Tag(0)), Tag(0)), (centroids.centroid_of(Tag(1)), Tag(1))],
        1e-4,
        150,
        1.5,
    )
    .unwrap();

    let text =
        "We loved the friendly staff and the delicious breakfast. We hated the rude service and the cold soggy meal.";
    let s0 = (0, 56);
    let s1 = (57, text.len() as u32);

    let run_pipeline = |vec_ann: &Arc<VectorAnnotator>| -> (Vec<f64>, Vec<f64>) {
        let doc = make_doc_with_sentences(text, &[s0, s1]);
        let tok_pipe =
            Analyzer::builder().given(Key::<Sentences>::DEFAULT).add(SimpleTokenizer::default()).build().unwrap();
        let mut docs = [doc];
        tok_pipe.run(&mut docs).remove(0).unwrap();
        let mut doc = docs.into_iter().next().unwrap();
        let opinions = {
            let tokens = doc.get::<Tokens>().unwrap();
            // Sentence 0 tokens: 0..10, Sentence 1 tokens: 10..21
            Opinions::from_items(
                tokens,
                [
                    Opinion { tokens: IxRange::new(0, 10), polarity: 0.0, intensity: 1.0, negated: false },
                    Opinion {
                        tokens: IxRange::new(10, tokens.len() as u32),
                        polarity: 0.0,
                        intensity: 1.0,
                        negated: false,
                    },
                ],
            )
            .unwrap()
        };
        doc.add(&Key::DEFAULT, opinions, Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Opinions>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(vec_ann)))
            .step(Bound::new(VectorHeadAnnotator::<Sentences>::new(centroids.clone())).write(0, "cent-sentiment"))
            .step(Bound::new(OpinionSpanHead::new(Arc::clone(vec_ann), head.clone())).write(0, "head-sentiment"))
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();

        let cent_cats = docs[0].get_key::<Categories<Sentences>>(&Key::named("cent-sentiment")).unwrap();
        let head_cats = docs[0].get_key::<Categories<Opinions>>(&Key::named("head-sentiment")).unwrap();
        assert_eq!(cent_cats.best_category(0), "positive");
        assert_eq!(cent_cats.best_category(1), "negative");
        assert_eq!(head_cats.best_category(0), "positive");
        assert_eq!(head_cats.best_category(1), "negative");
        (
            cent_cats.row_scores(0).iter().chain(cent_cats.row_scores(1)).copied().collect(),
            head_cats.row_scores(0).iter().chain(head_cats.row_scores(1)).copied().collect(),
        )
    };

    let (cpu_cent, cpu_head) = run_pipeline(&cpu_vec);
    let (dev_cent_1, dev_head_1) = run_pipeline(&dev_vec);
    let (dev_cent_2, dev_head_2) = run_pipeline(&dev_vec);

    // Repeated runs on the GPU must be bit-for-bit identical.
    assert_eq!(dev_cent_1, dev_cent_2);
    assert_eq!(dev_head_1, dev_head_2);

    // CPU and GPU scores must agree within the f32 precision tolerance.
    for (a, b) in cpu_cent.iter().zip(&dev_cent_1) {
        assert!((a - b).abs() <= 1e-5, "centroid score diff: cpu={a} dev={b}");
    }
    for (a, b) in cpu_head.iter().zip(&dev_head_1) {
        assert!((a - b).abs() <= 1e-5, "head score diff: cpu={a} dev={b}");
    }
    println!("{exec:?}: CPU/device head and centroid scores match within 1e-5");
}
