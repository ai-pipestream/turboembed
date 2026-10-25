use turbo_formats::masc::MascCorpus;
use turbo_zoo::RefusalReason;

use super::*;

#[test]
fn masc_keeps_its_released_partitions_when_ud_changes() {
    let masc = training_parameters();
    let ud = turbo_train_components::ud::training_parameters();
    assert_eq!(masc.get_int(Key::THREADS).unwrap(), Some(4));
    assert_eq!(ud.get_int(Key::THREADS).unwrap(), Some(12));
    for key in [Key::ITERATIONS, Key::CUTOFF] {
        assert_eq!(masc.get_int(key).unwrap(), ud.get_int(key).unwrap());
    }
    assert_eq!(turbo_train::GisParams::default().threads, 1);
}

#[test]
fn the_split_holds_out_one_in_ten() {
    let held: Vec<usize> = (0..35).filter(|&i| SPLIT.is_held_out(i)).collect();
    assert_eq!(held, [9, 19, 29]);
    assert_eq!(Split::new(0, 0), None);
    assert_eq!(Split::new(10, 10), None);
    assert_eq!(serde_json::to_string(&SPLIT).unwrap(), r#"{"every":10,"held_out_at":9}"#);
    assert_eq!(serde_json::from_str::<Split>(r#"{"every":10,"held_out_at":9}"#).unwrap(), SPLIT);
    for bad in
        [r#"{"every":0,"held_out_at":0}"#, r#"{"every":3,"held_out_at":3}"#, r#"{"every":3,"held_out_at":1,"x":1}"#]
    {
        assert!(serde_json::from_str::<Split>(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_twitter_genre_is_the_category_tw_of_a_header_s_text_class() {
    let header = |text_class: &str| {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<documentHeader \
             xmlns=\"http://www.xces.org/ns/GrAF/1.0/\"><profileDesc>{text_class}<primaryData \
             loc=\"a.txt\" f.id=\"f.text\"/></profileDesc></documentHeader>"
        )
    };
    let of = |text_class: &str| Categories::parse(&header(text_class)).unwrap();
    let tweets = of("<textClass catRef=\"WR TW\"><domain>Leisure</domain></textClass>");
    assert_eq!(tweets.codes(), ["WR", "TW"]);
    assert!(Exclusion::TwitterGenre.covers(&tweets));
    // Codes are compared whole, after whitespace is normalized.
    assert!(Exclusion::TwitterGenre.covers(&of("<textClass catRef=\" SP\tTW \"/>")));
    for other in ["<textClass catRef=\"WR BL\"/>", "<textClass catRef=\"WR TWX\"/>", "<textClass/>", ""] {
        assert!(!Exclusion::TwitterGenre.covers(&of(other)), "{other}");
    }
    assert_eq!(of(""), Categories::default());
    assert!(Categories::parse("<documentHeader>").is_err());
    assert_eq!(Exclusion::TwitterGenre.reason(), RefusalReason::PlatformTerms);
    assert_eq!(EXCLUSIONS, [Exclusion::TwitterGenre]);
    assert_eq!(serde_json::to_string(&Exclusion::TwitterGenre).unwrap(), r#""twitter_genre""#);
    let doc = ExcludedDocument { path: "data/written/twitter/tweets1.hdr".into(), exclusion: Exclusion::TwitterGenre };
    let json = serde_json::to_string(&doc).unwrap();
    assert_eq!(json, r#"{"path":"data/written/twitter/tweets1.hdr","exclusion":"twitter_genre"}"#);
    assert_eq!(serde_json::from_str::<ExcludedDocument>(&json).unwrap(), doc);
}

#[test]
fn a_document_outside_the_corpus_directory_is_refused() {
    let dir = std::env::temp_dir();
    let make_doc = |path: &Path| {
        turbo_formats::masc::MascDocument::parse(
            path,
            turbo_formats::masc::MascFiles {
                text: b"A.",
                segments: Some(b"<graph><region xml:id=\"seg-r0\" anchors=\"0 2\"/></graph>"),
                sentences: Some(b"<graph><region xml:id=\"s-r0\" anchors=\"0 2\"/></graph>"),
                penn: None,
                named_entities: None,
            },
        )
        .unwrap()
    };
    for bad in [Path::new("/elsewhere/doc.hdr").to_path_buf(), dir.join("../elsewhere/doc.hdr"), dir.clone()] {
        let corpus = MascCorpus::from_documents(vec![make_doc(&bad); 10]);
        assert!(matches!(SPLIT.apply(corpus, &dir), Err(Error::OutsideCorpus { .. })), "{}", bad.display());
    }
}

#[test]
fn an_unrecorded_release_is_neither_fetched_nor_taken_for_a_corpus() {
    let unrecorded = Release::NotRecorded;
    assert!(matches!(dataset(unrecorded), Err(Error::NotRecorded)));
    let home = std::env::temp_dir();
    let user: UserId = "tester".parse().unwrap();
    assert!(matches!(fetch_release(unrecorded, &home, &user), Err(Error::NotRecorded)));
    let any = CorpusDigest { sha256: String::new(), files: 0 };
    assert!(!unrecorded.is(&any));
}

#[test]
fn the_release_is_recorded_as_a_permissive_dataset_of_the_zoo() {
    let Release::Recorded(r) = RELEASE else { panic!("the release is recorded") };
    let a = dataset(RELEASE).unwrap();
    assert_eq!(Index::builtin().get(DATASET_ID).unwrap(), &a);
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&a)).as_bytes()).unwrap();
    let got = index.get(DATASET_ID).unwrap();
    assert_eq!(got.class, LicenceClass::Permissive);
    assert_eq!((got.files[0].size, got.files[0].sha256.to_string()), (r.size, r.sha256.to_owned()));
    assert_eq!(r.corpus_sha256.len(), 64);
    assert!(r.corpus_sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    assert!(RELEASE.is(&CorpusDigest { sha256: r.corpus_sha256.to_owned(), files: 4425 }));
}

const RECORDED: RecordedRelease = RecordedRelease {
    file: "MASC-3.0.0.zip",
    format: Format::Zip,
    url: "https://example.org/MASC-3.0.0.zip",
    size: 3,
    sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    sha512: "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    corpus_dir: "MASC-3.0.0",
    corpus_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
};

#[test]
fn a_recorded_release_is_a_permissive_dataset_the_index_takes() {
    let a = dataset(Release::Recorded(RECORDED)).unwrap();
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&a)).as_bytes()).unwrap();
    let got = index.get(DATASET_ID).unwrap();
    assert_eq!(got.class, LicenceClass::Permissive);
    assert_eq!(got.computed_class(), LicenceClass::Permissive);
    assert_eq!(index.shippable().count(), 1);
}

#[test]
fn the_corpus_digest_covers_every_file_by_path_and_content() {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-digest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("b")).unwrap();
    std::fs::write(dir.join("a.txt"), "abc").unwrap();
    std::fs::write(dir.join("b/c.txt"), "").unwrap();
    let d = corpus_digest(&dir).unwrap();
    // sha256sum's manifest of the two files.
    let manifest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a.txt\n\
                    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  b/c.txt\n";
    assert_eq!(d, CorpusDigest { sha256: Sha256::of(manifest.as_bytes()).to_string(), files: 2 });

    assert!(matches!(check_corpus(&dir, Release::NotRecorded, false), Err(Error::NotRecorded)));
    assert_eq!(check_corpus(&dir, Release::NotRecorded, true).unwrap(), d);
    // Another corpus than the recorded release: refused, unless asked
    // for, and then not taken for the release.
    let other = check_corpus(&dir, Release::Recorded(RECORDED), false).unwrap_err();
    assert!(matches!(other, Error::CorpusDigest { .. }), "{other}");
    assert!(other.to_string().contains("MASC-3.0.0"), "{other}");
    assert_eq!(check_corpus(&dir, Release::Recorded(RECORDED), true).unwrap(), d);
    assert!(!Release::Recorded(RECORDED).is(&d));
    assert!(matches!(check_corpus(&dir, RELEASE, false), Err(Error::CorpusDigest { .. })));
    let same = CorpusDigest { sha256: RECORDED.corpus_sha256.to_owned(), files: 2 };
    assert_eq!(check_digest(same.clone(), Release::Recorded(RECORDED), false).unwrap(), same);
    assert!(Release::Recorded(RECORDED).is(&same));

    std::fs::write(dir.join("b/c.txt"), "x").unwrap();
    assert_ne!(corpus_digest(&dir).unwrap().sha256, d.sha256);
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(dir.join("a.txt"), dir.join("link")).unwrap();
        assert!(corpus_digest(&dir).is_err());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
