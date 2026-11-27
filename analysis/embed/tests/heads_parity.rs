//! CPU and device parity for `LabelCentroids` and `ClassifierHead` (#321).
//!
//! The device is the one `TURBO_TEST_DEVICE` names, and the test says it
//! was skipped when it names none. `TURBO_TEST_REQUIRE_CUDA`,
//! `TURBO_TEST_REQUIRE_LEVELZERO` and `TURBO_TEST_REQUIRE_METAL` do not
//! choose the device: set to anything but 0, the test fails unless
//! `TURBO_TEST_DEVICE` names a device of that backend.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Categories, Opinion, Opinions, Sentences, Tokens};
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[test]
fn heads_and_centroids_cpu_and_device_parity() {
    let Some((want, device)) = device::named_device() else {
        println!("skipped: TURBO_TEST_DEVICE names no device");
        return;
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

#[test]
fn a_required_backend_does_not_choose_the_device() {
    use heads_support::device::device_for;
    // nothing required: the device that is named, or none
    assert_eq!(device_for(None, None), Ok(None));
    assert_eq!(device_for(Some("cpu"), None), Ok(Some(Device::Cpu)));
    assert_eq!(device_for(Some("levelzero"), None), Ok(Some(Device::Backend("levelzero".into()))));
    assert_eq!(device_for(Some("1"), None), Ok(Some(Device::Index(1))));
    assert_eq!(device_for(Some("cuda"), Some("cuda")), Ok(Some(Device::Backend("cuda".into()))));
    // a required backend with no name is not a device: the test fails
    let unset = device_for(None, Some("cuda")).unwrap_err();
    assert_eq!(unset, "TURBO_TEST_REQUIRE_CUDA requires a cuda device: set TURBO_TEST_DEVICE=cuda");
    // and so do the CPU, another backend's device and an index
    for other in ["cpu", "levelzero", "0"] {
        let refused = device_for(Some(other), Some("cuda")).unwrap_err();
        assert_eq!(refused, format!("TURBO_TEST_REQUIRE_CUDA requires a cuda device, and TURBO_TEST_DEVICE={other}"));
    }
}

#[test]
fn a_backend_with_a_number_is_refused_before_any_model_loads() {
    use heads_support::device::device_for;
    // the name passes the required backend's check and is no Device
    for (named, required) in [("levelzero:0", None), ("levelzero:0", Some("levelzero")), ("cuda:1", Some("cuda"))] {
        let backend = named.split(':').next().unwrap();
        assert_eq!(
            device_for(Some(named), required).unwrap_err(),
            format!(
                "TURBO_TEST_DEVICE={named}: a head test picks a backend's first device ({backend}) or a device index"
            )
        );
    }
    // another backend's requirement is still what the run hears first
    assert_eq!(
        device_for(Some("levelzero:0"), Some("cuda")).unwrap_err(),
        "TURBO_TEST_REQUIRE_CUDA requires a cuda device, and TURBO_TEST_DEVICE=levelzero:0"
    );
}
