//! The tagger's gate over the tags the reference can give, and its
//! samples from the checked-in MASC corpora.

use std::path::PathBuf;

use crate::bundle;
use crate::common::{GATED_DOCUMENTS, SENTENCES, rewrite_document, temp_dir, write_corpus};
use crate::{sentences_after, split_of};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::TagCount;
use turbo_train_recipes::postag::{self, POS, Scored};
use turbo_train_recipes::{Error, Recipe, held_out_name};

/// A sentence tagged as MASC tags brackets, a hyphen, an ellipsis, a
/// semicolon and an opening quotation mark, and the same sentence as the
/// reference tags them.
const MASC_SPELLED: &str =
    "Wait/VB .../... he/PRP said/VBD -/- no/DT ~;/; then/RB (/( yes/UH ~)/) and/CC \"/'' ~fine/JJ ~\"/'' ~./.";
const REFERENCE_SPELLED: &str =
    "Wait/VB .../: he/PRP said/VBD -/: no/DT ~;/: then/RB (/-LRB- yes/UH ~)/-RRB- and/CC \"/`` ~fine/JJ ~\"/'' ~./.";
/// A tag no Penn tagger gives, as MASC 3.0.0 has a few.
const ODD_TAG: &str = "Agh/UH` ~!/.";

/// The tagger's gate scores the held-out tokens whose tag the reference
/// can give, as written or through the mapping of its tags: a candidate
/// trained on MASC's tags is compared with the held-out tags as written,
/// and a reference trained on the same sentences in its own conventions
/// through the mapping, so both are right on every token; the reference's
/// tags are not counted right for the candidate. The token of a tag
/// neither can give is left out and counted, and more such tokens than
/// one in a thousand are an error that names their tags.
#[test]
fn the_tagger_is_scored_over_the_tags_the_reference_can_give_mapping_only_the_reference_s() {
    let pool = |extra: &'static str| {
        let mut pool = SENTENCES.to_vec();
        pool.extend([extra, ODD_TAG]);
        pool
    };
    let (masc_pool, reference_pool) = (pool(MASC_SPELLED), pool(REFERENCE_SPELLED));
    let root = temp_dir("mappings");
    write_corpus(&root, GATED_DOCUMENTS, &masc_pool[..=SENTENCES.len()]);
    let reference_root = temp_dir("mappings-reference");
    write_corpus(&reference_root, GATED_DOCUMENTS, &reference_pool[..=SENTENCES.len()]);
    let (split, digest) = split_of(&root);
    let (reference_split, _) = split_of(&reference_root);
    let candidate = POS.train(&split.train).unwrap();
    let reference = POS.train(&reference_split.train).unwrap();

    // One held-out sentence with a tag neither model gives.
    let odd = SENTENCES.len() + 1;
    rewrite_document(&root, &split.held_out_paths[0], &masc_pool, &sentences_after(&[odd]));
    let (split, _) = split_of(&root);
    let tags = POS.held_out(&split.held_out).unwrap();
    let run = Recipe::Pos(POS)
        .gate(&candidate.bytes, "reference", &reference.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    let coverage = run.held_out.tokens.unwrap();
    assert_eq!(coverage.tokens, tags.tokens());
    assert_eq!(coverage.left_out, [TagCount { tag: "UH`".to_owned(), tokens: 1 }]);
    assert_eq!(coverage.reference_mismatches, []);
    let scored: u64 = run.record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(scored, tags.tokens() - 1);
    // The reference gives `-LRB-`, `:` and ``` `` ``` where MASC has `(`, `-`
    // and `''`, and is right on every scored token, as the candidate is.
    assert_eq!(
        (run.record.result.candidate.score, run.record.result.reference.score),
        (100.0, 100.0),
        "{}",
        run.record
    );
    // Scored as the candidate, the reference's tags are compared as
    // written, and its `-LRB-`, `:` and ``` `` ``` are wrong.
    let reference_tags = POS.reference_tags(&reference.bytes).unwrap();
    let as_candidate: u64 = POS
        .score(&reference.bytes, Scored::Candidate, &tags, &reference_tags)
        .unwrap()
        .iter()
        .map(|c| c.true_positives)
        .sum();
    let mapped: u64 = POS
        .score(&reference.bytes, Scored::Reference, &tags, &reference_tags)
        .unwrap()
        .iter()
        .map(|c| c.true_positives)
        .sum();
    // Each held-out copy of the sentence has six such tokens: the
    // brackets, the hyphen, the ellipsis, the semicolon and the opening
    // quotation mark (the closing one is `''` in both).
    let copies = tags.samples().iter().filter(|s| s.tokens()[0].as_str() == "Wait").count() as u64;
    assert!(copies > 0);
    assert_eq!(mapped, scored);
    assert_eq!(mapped - as_candidate, 6 * copies);

    // Five such tokens are more than one in a thousand.
    rewrite_document(&root, &split.held_out_paths[0], &masc_pool, &sentences_after(&[odd; 5]));
    let (split, _) = split_of(&root);
    let tokens = POS.held_out(&split.held_out).unwrap().tokens();
    assert!((1000..5000).contains(&tokens), "{tokens}");
    let err = Recipe::Pos(POS)
        .gate(&candidate.bytes, "reference", &reference.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap_err();
    assert!(
        matches!(&err, Error::TagsOutsideReference { left_out, of, .. } if *of == tokens && left_out == &[TagCount { tag: "UH`".to_owned(), tokens: 5 }]),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        format!(
            "the reference reference cannot give the tags of 5 of {tokens} held-out tokens, more than the limit of \
             1 in 1000: UH` (5)"
        )
    );
    // The check made before training finds the same, given the
    // reference; without it, it checks only the samples.
    let checked = Recipe::Pos(POS).check_data(&split, false, "reference", Some(&reference.bytes)).unwrap_err();
    assert_eq!(checked.to_string(), err.to_string());
    Recipe::Pos(POS).check_data(&split, false, "reference", None).unwrap();
    for dir in [root, reference_root] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// What the part-of-speech recipe reads from the MASC corpora checked in
/// for the goldens (`analysis/goldens/data/inputs/formats/masc-*.txt`,
/// written for the tests in MASC's layout), against what OpenNLP's
/// `MascPOSSampleStream` read from them in the JVM (the `masc/pos`
/// goldens). The goldens' generator reads on after a read throws, so a
/// golden's first pass records every sample the stream read and every
/// exception it threw, each a token without a tag
/// (`IllegalArgumentException`): the recipe reads the same samples in the
/// same order and skips one sentence for each exception. A corpus the
/// stream could not open (`masc-aborts`) is refused when it is read, and
/// one whose documents have no Penn tags (`masc-unpenned`) when its
/// samples are read, each with the JVM's exception. The recipe's other
/// skip, a sentence of no tokens, is in none of these corpora;
/// `skipped_sentences_are_recorded_and_bounded` reads one.
#[test]
fn the_tagger_recipe_reads_the_jvm_s_samples_from_the_checked_in_masc_corpora() {
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
    // A layer's rows of the first pass, without the pass column, or every
    // row of a layer without one.
    let jvm = |name: &str, id: &str| -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(goldens.join(format!("masc/pos/formats/{name}.golden"))).unwrap();
        let start = text.find(&format!("layer\t{id}\t")).unwrap_or_else(|| panic!("{name}: no {id}"));
        let body = &text[start..];
        let end = body.find(&format!("\nend\t{id}")).unwrap();
        let rows = body[..end].lines().skip(1).map(|l| l.split('\t').map(str::to_owned).collect::<Vec<_>>());
        if id == "turbo:open-errors" {
            rows.collect()
        } else {
            rows.filter(|r| r[0] == "0").map(|r| r[1..].to_vec()).collect()
        }
    };
    let write = |name: &str| {
        let root = temp_dir(&format!("checked-in-{name}"));
        let bundle = std::fs::read_to_string(goldens.join(format!("inputs/formats/{name}.txt"))).unwrap();
        bundle::write_bundle(&bundle, &root);
        root
    };
    let names = [
        "masc-basic",
        "masc-names",
        "masc-opennlp",
        "masc-penn",
        "masc-sentences",
        "masc-skips",
        "masc-unicode",
        "masc-xml",
    ];
    let mut skipped_any = 0;
    for name in names {
        let expected: Vec<String> = jvm(name, "turbo:samples").into_iter().map(|r| r[0].clone()).collect();
        let thrown = jvm(name, "turbo:read-errors");
        assert!(jvm(name, "turbo:open-errors").is_empty(), "{name}");
        for row in &thrown {
            assert_eq!(row[1..], ["java.lang.IllegalArgumentException", "null elements are not allowed in tags!"]);
        }
        let root = write(name);
        let corpus = MascCorpus::read(&root).unwrap();
        let mut read = Vec::new();
        let skipped = postag::read_samples(&corpus, |s| read.push(s.to_string())).unwrap();
        assert_eq!(read, expected, "{name}");
        assert_eq!(skipped, thrown.len(), "{name}");
        println!(
            "{name}: {} samples read, {skipped} sentences skipped; the JVM threw {} times",
            read.len(),
            thrown.len()
        );
        skipped_any += skipped;
        std::fs::remove_dir_all(&root).unwrap();
    }
    // masc-penn and masc-xml have sentences with a token without a tag.
    assert_eq!(skipped_any, 3);

    let opened = jvm("masc-aborts", "turbo:open-errors");
    assert_eq!(opened.len(), 1);
    let root = write("masc-aborts");
    let err = MascCorpus::read(&root).unwrap_err();
    assert_eq!(err.java_class(), Some(opened[0][0].as_str()), "{err}");
    assert_eq!(err.to_string(), opened[0][1]);
    std::fs::remove_dir_all(&root).unwrap();

    let opened = jvm("masc-unpenned", "turbo:open-errors");
    assert_eq!(opened.len(), 1);
    let root = write("masc-unpenned");
    let corpus = MascCorpus::read(&root).unwrap();
    match postag::read_samples(&corpus, |s| panic!("masc-unpenned: a sample {s}")) {
        Err(Error::Masc(e)) => {
            assert_eq!(e.java_class(), Some(opened[0][0].as_str()), "{e}");
            assert_eq!(e.to_string(), opened[0][1]);
        }
        other => panic!("masc-unpenned: {other:?}"),
    }
    std::fs::remove_dir_all(&root).unwrap();
}
