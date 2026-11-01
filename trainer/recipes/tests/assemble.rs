//! End-to-end tests for assembling `turbo-trainer` receipts into single-head
//! and multi-task bundles, generating `turbo-zoo` entries, enforcing licence
//! tiers through `Zoo::ensure`, and running `turbo-tagger` annotators (#1726).

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::TokenTagger;
use turbo::manifest::{Decode, HeadKind, HeadSpec, KEEP_TOKEN, LicenseClass, Scheme, WordPooling};
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, LanguageSpans, PosTags, Sentences, Subwords, Tokens, Words};
use turbo_tagger::{
    EntityTagger, LanguageTagger, MultiTaskTagger, PosTagger, SentenceSegmenter, TaggerZooError, WordEncoder,
    WordSplitTagger,
};
use turbo_trainer_recipes::assemble::assemble_bundle;
use turbo_trainer_recipes::assemble_zoo::{ZooEntryOptions, seed_blobs, zoo_artifact};
use turbo_trainer_recipes::corpus::FileRecord;
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH};
use turbo_trainer_recipes::safetensors;
use turbo_zoo::{
    Acceptances, Artifact, Blobs, Component, Directory, File, Format, Index, Kind, Licence, LicenceClass, Policy,
    Transport, Url, UserId, Zoo, ZooError,
};

struct SpyBlobs<'a> {
    blobs: &'a Blobs,
    requested: RefCell<Vec<String>>,
}

impl Transport for SpyBlobs<'_> {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        self.requested.borrow_mut().push(url.as_str().to_owned());
        self.blobs.get(url, size)
    }

    fn get_file(&self, a: &Artifact, f: &File) -> Result<Vec<u8>, String> {
        self.requested.borrow_mut().push(format!("{}:{}", a.id, f.path));
        self.blobs.get_file(a, f)
    }
}

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-assemble-1726-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn corpus_multiconer(dir: &Path) {
    let names = ["ann", "bob", "eve", "max"];
    let places = ["paris", "rome", "oslo"];
    let mut text = String::new();
    for i in 0..32 {
        let (n, p) = (names[i % names.len()], places[i % places.len()]);
        text += &format!(
            "# id {i}\tdomain=en\nthey _ _ O\nmet _ _ O\n{n} _ _ B-OtherPER\nin _ _ O\n{p} _ _ B-HumanSettlement\n\n"
        );
    }
    fs::create_dir_all(dir.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(dir.join(format!("EN-English/{f}.conll")), &text).unwrap();
    }
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn train_ner_run(data: &Path, out: &Path) -> Receipt {
    let b = tiny();
    let args = [
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        data.to_str().unwrap(),
        "--bundle",
        b.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--crf",
        "--epochs",
        "4",
        "--batch",
        "8",
        "--lr",
        "0.02",
        "--commit",
        "1726test",
    ];
    let o = run_cli(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let mut r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    r.corpus.recorded = true;
    fs::write(out.join("receipt.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    r
}

fn make_spec(name: &str, layer: &str, kind: HeadKind, scheme: Scheme, decode: Decode, labels: &[&str]) -> HeadSpec {
    HeadSpec {
        name: name.into(),
        layer: layer.into(),
        kind,
        labels: labels.iter().map(|&s| s.to_owned()).collect(),
        scheme,
        word_pooling: WordPooling::First,
        adapter: None,
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        layer_mix: None,
        span: None,
        relation: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: format!("heads.{name}"),
    }
}

fn zoo_opts(id: &str, kind: Kind, component: Component, lang: &str) -> ZooEntryOptions {
    ZooEntryOptions {
        id: id.into(),
        kind,
        component: Some(component),
        language: Some(lang.into()),
        version: "1".into(),
        license_terms: "https://www.apache.org/licenses/LICENSE-2.0".into(),
        url_prefix: "https://models.turboembed.org/zoo".into(),
        origin: None,
    }
}

fn synthesize_run(base_run: &Path, dest: &Path, spec: HeadSpec, spdx: &str, terms: &str, recipe_only: bool) -> Receipt {
    fs::create_dir_all(dest).unwrap();
    let mut r: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    r.corpus.spdx = spdx.into();
    r.corpus.licence = spdx.into();
    r.corpus.terms = terms.into();
    r.corpus.recipe_only = recipe_only;
    let class = r.corpus.class().unwrap();
    let mut spec = spec;
    spec.license_class = Some(class);
    let tensors = tagger_tensors(&spec.tensor_prefix, 32, spec.labels.len());
    let spec_bytes = serde_json::to_vec_pretty(&spec).unwrap();
    let tr = r.training.as_ref().unwrap();
    let meta = [(CHECKPOINT_EPOCH, tr.best_epoch.to_string()), (CHECKPOINT_DEV, tr.dev[tr.best_epoch].to_string())];
    let st_bytes = safetensors::write_with(&tensors, &meta);
    fs::write(dest.join("head.json"), &spec_bytes).unwrap();
    fs::write(dest.join("head.safetensors"), &st_bytes).unwrap();
    r.head = Some(spec);
    r.outputs = vec![
        FileRecord { path: "head.safetensors".into(), size: st_bytes.len() as u64, sha256: sha256_hex(&st_bytes) },
        FileRecord { path: "head.json".into(), size: spec_bytes.len() as u64, sha256: sha256_hex(&spec_bytes) },
    ];
    fs::write(dest.join("receipt.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    r
}

fn tagger_tensors(prefix: &str, hidden: usize, labels: usize) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let mut w = vec![0.0f32; labels * hidden];
    for i in 0..labels.min(hidden) {
        w[i * hidden + i] = 1.0;
    }
    let mut b = vec![0.0f32; labels];
    if labels > 1 {
        b[1] = 2.0;
    }
    vec![
        (format!("{prefix}.out.weight"), vec![labels as u64, hidden as u64], w),
        (format!("{prefix}.out.bias"), vec![labels as u64], b),
    ]
}

fn tokenize_words(text: &str, words: &[&str]) -> (Tokens, Vec<Span>) {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for w in words {
        let rel = text[cursor..].find(w).unwrap();
        let start = (cursor + rel) as u32;
        let end = start + w.len() as u32;
        let sp = Span::new(start, end);
        spans.push(sp);
        out.push(sp);
        cursor = end as usize;
    }
    let count = spans.len();
    (Tokens { spans, sentence: Some(vec![Ix::new(0); count]) }, out)
}

#[test]
fn single_head_and_multitask_bundles_load_through_zoo_and_match_receipt_predictions() {
    let data = scratch("e2e-data");
    corpus_multiconer(&data);
    let ner_run = scratch("e2e-ner-run");
    let ner_receipt = train_ner_run(&data, &ner_run);

    let single_bundle = scratch("e2e-single-bundle");
    let zoo_entry_file = scratch("e2e-zoo-entry.json");
    let cli = run_cli(&[
        "assemble",
        "--bundle",
        tiny().to_str().unwrap(),
        "--run",
        ner_run.to_str().unwrap(),
        "--out",
        single_bundle.to_str().unwrap(),
        "--zoo-id",
        "turbo-en-multiconer-test-ner-1",
        "--component",
        "COMPONENT_NAME_FINDER",
        "--language",
        "en",
        "--license-terms",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "--url-prefix",
        "https://models.turboembed.org/zoo",
        "--zoo-out",
        zoo_entry_file.to_str().unwrap(),
    ]);
    assert!(cli.status.success(), "{}", String::from_utf8_lossy(&cli.stderr));

    let opened = Bundle::open(&single_bundle).unwrap();
    assert_eq!(opened.manifest.license_tier(), Some(LicenseClass::ShareAlike));
    assert_eq!(opened.manifest.head_provenance.len(), 1);
    assert_eq!(opened.manifest.head_provenance[0].head, "ner");
    assert_eq!(opened.manifest.head_provenance[0].corpus_id, "multiconer2-en");

    let artifact: Artifact = serde_json::from_slice(&fs::read(&zoo_entry_file).unwrap()).unwrap();
    assert_eq!(artifact.kind, Kind::Head);
    assert_eq!(artifact.version, "1");
    assert_eq!(artifact.class, LicenceClass::ShareAlike);
    assert_eq!(artifact.component, Some(Component::NameFinder));

    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes()).unwrap();
    let user: UserId = "alice".parse().unwrap();
    let mut acc = Acceptances::default();
    acc.accept(&user, Licence::CcBySa40, 1).unwrap();

    let dir_transport = Directory::new(&single_bundle);
    let cache_dir = scratch("e2e-zoo-cache-dir");
    let zoo_dir = Zoo::new(index, cache_dir, Policy::default(), &dir_transport);
    let entity_tagger =
        EntityTagger::from_zoo_language(&zoo_dir, "en", None, &EncoderOptions::default(), &user, &acc).unwrap();

    let sample_words = ["they", "met", "ann", "in", "paris"];
    let sample_text = "they met ann in paris";
    let doc = Document::new(sample_text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (toks, word_spans) = tokenize_words(sample_text, &sample_words);
    let (entities,) = entity_tagger.annotate((view, &toks, None::<&Subwords>, None::<&Sentences>)).unwrap();

    let mut enc = WordEncoder::open(&tiny(), &EncoderOptions::default()).unwrap();
    let direct_tagger = TokenTagger::load(&opened, "ner").unwrap();
    let feats = enc.features(&[sample_words.to_vec()], WordPooling::First).unwrap();
    let mut scores = Vec::new();
    let mut work = turbo::head::tagger::Work::default();
    let word_ix: Vec<u32> = (0..sample_words.len() as u32).collect();
    direct_tagger.scores_forms(&feats[0], &word_ix, Some(&sample_words), &mut scores, &mut work).unwrap();
    let mut direct_tags = Vec::new();
    direct_tagger.decode(&scores, sample_words.len(), &mut direct_tags).unwrap();
    let direct_spans = direct_tagger.spans(&direct_tags).unwrap();
    assert_eq!(entities.spans.len(), direct_spans.len());
    for (i, w) in direct_spans.iter().enumerate() {
        let got = entities.spans.get(Ix::new(i as u32));
        let (want_start, want_end) = (word_spans[w.start as usize].start, word_spans[w.end as usize - 1].end);
        assert_eq!((got.start, got.end), (want_start, want_end));
        assert_eq!(&sample_text[got.range()], sample_words[w.start as usize]);
    }
    assert!(ner_receipt.eval.model.as_ref().unwrap().score() >= 0.0);

    let (pos_run, split_run, sent_run, lang_run) =
        (scratch("e2e-pos-run"), scratch("e2e-split-run"), scratch("e2e-sent-run"), scratch("e2e-lang-run"));
    let cc_by = ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/");
    let split_label = "(m 0 1 () (s 'l' '')) (m 1 0 (s 'd' '') ())";
    for (dir, spec) in [
        (
            &pos_run,
            make_spec("pos", "turboembed:pos", HeadKind::TokenTagger, Scheme::Tags, Decode::Argmax, &["NOUN", "VERB"]),
        ),
        (
            &split_run,
            make_spec(
                "words",
                "turboembed:words",
                HeadKind::TokenSplit,
                Scheme::Tags,
                Decode::EditTree,
                &[KEEP_TOKEN, split_label],
            ),
        ),
        (
            &sent_run,
            make_spec(
                "sentences",
                "turboembed:sentences",
                HeadKind::SentenceBoundary,
                Scheme::Tags,
                Decode::Argmax,
                &["inside", "last"],
            ),
        ),
        (
            &lang_run,
            make_spec(
                "language",
                "turboembed:language-spans",
                HeadKind::TokenTagger,
                Scheme::Tags,
                Decode::Argmax,
                &["eng", "spa"],
            ),
        ),
    ] {
        synthesize_run(&ner_run, dir, spec, cc_by.0, cc_by.1, false);
    }

    let multi_dir = scratch("e2e-multi-bundle");
    let assembled =
        assemble_bundle(&tiny(), &[&ner_run, &pos_run, &split_run, &sent_run, &lang_run], &multi_dir).unwrap();
    assert_eq!(assembled.license_tier, LicenseClass::ShareAlike);

    let blob_root = scratch("e2e-blobs");
    let blobs = Blobs::new(&blob_root);
    seed_blobs(&assembled, &blobs).unwrap();
    let multi_art =
        zoo_artifact(&assembled, &zoo_opts("turbo-en-multi-test-1", Kind::Model, Component::Pos, "en")).unwrap();
    let multi_index = Index::parse(Index::to_canonical_json(&[multi_art]).as_bytes()).unwrap();
    let blob_cache = scratch("e2e-blob-cache");
    let zoo_blobs = Zoo::new(multi_index, blob_cache, Policy::default(), &blobs);

    let multi: MultiTaskTagger<(PosTags, Entities, Words, LanguageSpans)> = MultiTaskTagger::from_zoo(
        &zoo_blobs,
        "turbo-en-multi-test-1",
        &[None, None, None, None],
        &EncoderOptions::default(),
        &user,
        &acc,
    )
    .unwrap();
    let (pos_out, ent_out, words_out, lang_out) =
        multi.annotate((view, &toks, None::<&Subwords>, None::<&Sentences>)).unwrap();
    assert_eq!(pos_out.tags.len(), sample_words.len());
    assert_eq!(ent_out.spans.len(), entities.spans.len());
    assert_eq!(words_out.len(), sample_words.len());
    assert!(!lang_out.spans.is_empty());

    let seg =
        SentenceSegmenter::from_zoo(&zoo_blobs, "turbo-en-multi-test-1", None, &EncoderOptions::default(), &user, &acc)
            .unwrap();
    let (sents,) = seg.annotate((view, &toks, None::<&Subwords>)).unwrap();
    assert!(!sents.spans.is_empty());

    let word_tagger =
        WordSplitTagger::from_zoo(&zoo_blobs, "turbo-en-multi-test-1", None, &EncoderOptions::default(), &user, &acc)
            .unwrap();
    let (w2,) = word_tagger.annotate((view, &toks, None::<&Subwords>, None::<&Sentences>)).unwrap();
    assert_eq!(w2.len(), sample_words.len());

    let lang_tagger =
        LanguageTagger::from_zoo(&zoo_blobs, "turbo-en-multi-test-1", None, &EncoderOptions::default(), &user, &acc)
            .unwrap();
    let (l2,) = lang_tagger.annotate((view, &toks, None::<&Subwords>, None::<&Sentences>)).unwrap();
    assert_eq!(l2.spans.len(), lang_out.spans.len());
}

#[test]
fn licence_classes_and_bundle_filtering_in_zoo() {
    let data = scratch("lic-data");
    corpus_multiconer(&data);
    let base_run = scratch("lic-base-run");
    train_ner_run(&data, &base_run);

    let (pos_perm, ner_sa, local_run) = (scratch("lic-pos-perm"), scratch("lic-ner-sa"), scratch("lic-local-run"));
    let pos_spec =
        make_spec("pos", "turboembed:pos", HeadKind::TokenTagger, Scheme::Tags, Decode::Argmax, &["NOUN", "VERB"]);
    let ner_spec =
        make_spec("ner", "turboembed:entities", HeadKind::TokenTagger, Scheme::Bio, Decode::Argmax, &["O", "B-PER"]);
    let mut loc_spec = pos_spec.clone();
    loc_spec.name = "pos_local".into();
    loc_spec.tensor_prefix = "heads.pos_local".into();
    synthesize_run(&base_run, &pos_perm, pos_spec, "CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/", false);
    synthesize_run(
        &base_run,
        &ner_sa,
        ner_spec,
        "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        false,
    );
    synthesize_run(
        &base_run,
        &local_run,
        loc_spec,
        "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        true,
    );

    let sa_bundle_dir = scratch("lic-sa-bundle");
    let sa_bundle = assemble_bundle(&tiny(), &[&pos_perm, &ner_sa], &sa_bundle_dir).unwrap();
    assert_eq!(sa_bundle.license_tier, LicenseClass::ShareAlike);
    let sa_art = zoo_artifact(&sa_bundle, &zoo_opts("turbo-en-sa-multi-1", Kind::Model, Component::Pos, "en")).unwrap();
    assert_eq!(sa_art.class, LicenceClass::ShareAlike);

    let perm_bundle_dir = scratch("lic-perm-bundle");
    let perm_bundle = assemble_bundle(&tiny(), &[&pos_perm], &perm_bundle_dir).unwrap();
    let perm_art =
        zoo_artifact(&perm_bundle, &zoo_opts("turbo-en-perm-pos-1", Kind::Head, Component::Pos, "en")).unwrap();
    assert_eq!(perm_art.class, LicenceClass::Permissive);

    let local_bundle_dir = scratch("lic-local-bundle");
    let local_bundle = assemble_bundle(&tiny(), &[&local_run], &local_bundle_dir).unwrap();
    assert_eq!(local_bundle.license_tier, LicenseClass::LocalOnly);
    let local_art =
        zoo_artifact(&local_bundle, &zoo_opts("turbo-pt-porttinari-pos-1", Kind::Head, Component::Pos, "pt")).unwrap();
    assert_eq!(local_art.class, LicenceClass::LocalOnly);

    let mut pt_art = perm_art.clone();
    pt_art.id = "non-bundle-pt".parse().unwrap();
    pt_art.format = Format::Pytorch;
    let mut raw_st_art = perm_art.clone();
    raw_st_art.id = "non-bundle-raw-st".parse().unwrap();
    raw_st_art.files.retain(|f| f.path.as_str() != "manifest.json");

    let blob_root = scratch("lic-blobs");
    let blobs = Blobs::new(&blob_root);
    seed_blobs(&sa_bundle, &blobs).unwrap();
    seed_blobs(&perm_bundle, &blobs).unwrap();
    seed_blobs(&local_bundle, &blobs).unwrap();

    let index =
        Index::parse(Index::to_canonical_json(&[pt_art, raw_st_art, sa_art, perm_art, local_art]).as_bytes()).unwrap();
    let spy = SpyBlobs { blobs: &blobs, requested: RefCell::new(Vec::new()) };
    let cache = scratch("lic-zoo-cache");
    let zoo = Zoo::new(index, cache, Policy::default(), &spy);
    let user: UserId = "alice".parse().unwrap();
    let mut acc = Acceptances::default();

    assert!(matches!(
        PosTagger::from_zoo(&zoo, "non-bundle-pt", None, &EncoderOptions::default(), &user, &acc),
        Err(TaggerZooError::NotABundle { format: Format::Pytorch, .. })
    ));
    assert!(matches!(
        PosTagger::from_zoo(&zoo, "non-bundle-raw-st", None, &EncoderOptions::default(), &user, &acc),
        Err(TaggerZooError::NotABundle { format: Format::Safetensors, .. })
    ));
    assert!(spy.requested.borrow().is_empty(), "non-bundles must be refused before transport fetch");

    let _pos = PosTagger::from_zoo_language(&zoo, "en", None, &EncoderOptions::default(), &user, &acc).unwrap();
    assert!(
        spy.requested.borrow().iter().all(|r| r.starts_with("turbo-en-perm-pos-1:")),
        "from_zoo_language must skip non-bundles without fetching: {:?}",
        spy.requested.borrow()
    );

    assert!(matches!(
        zoo.ensure("turbo-en-sa-multi-1", &user, &acc),
        Err(ZooError::NotAccepted { class: LicenceClass::ShareAlike, .. })
    ));
    acc.accept(&user, Licence::CcBySa40, 1).unwrap();
    assert!(zoo.ensure("turbo-en-sa-multi-1", &user, &acc).is_ok());

    assert!(matches!(
        PosTagger::from_zoo(&zoo, "turbo-pt-porttinari-pos-1", None, &EncoderOptions::default(), &user, &acc),
        Err(TaggerZooError::Zoo(ZooError::LocalOnly { .. }))
    ));
}
