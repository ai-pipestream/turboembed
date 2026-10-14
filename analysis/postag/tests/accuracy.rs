//! The English model's accuracy on the UD English EWT test split, and the
//! Rust tags against the JVM's over the same split, token by token.
//!
//! The treebank is CC BY-SA, so it is not in the repository: the test is
//! ignored unless `TURBO_UD_EWT_TEST` names a `en_ewt-ud-test.conllu`
//! downloaded for the run, as `cargo test -p turbo-postag --test accuracy
//! -- --ignored --nocapture` with the variable set. `TURBO_UD_EWT_JVM_TAGS`
//! may name the tags `PosAccuracy` (analysis/goldens-tool/java) wrote for
//! the same file, one sentence per line; every token's tag must then be
//! the JVM's. See analysis/postag/README.md for the commands and the
//! numbers they gave.
//!
//! The file is read as OpenNLP's `ConlluStream` and `ConlluPOSSampleStream`
//! read it for `POSTaggerEvaluator` (and as `PosAccuracy` reads it): a
//! sentence per blank-line block, comment lines skipped, every other line
//! ten tab-separated fields; a multiword token line keeps its form and
//! takes the UPOS tags of its parts joined with `+`, the part lines
//! dropped; an empty node stays a token tagged `_`. Both are tokens no
//! tagger gets right, so the accuracy is given over every token, as
//! `POSTaggerEvaluator` gives it, and over the plain word lines alone.

use std::path::{Path, PathBuf};

use turbo_postag::{PosModel, PosTagger};

/// One sentence: the forms, the reference tags and whether each token is
/// a plain word line (neither a multiword token nor an empty node).
#[derive(Debug, PartialEq)]
struct Sentence {
    forms: Vec<String>,
    tags: Vec<String>,
    plain: Vec<bool>,
}

fn read_conllu(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut block: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            if !block.is_empty() {
                sentences.push(sentence(&block));
                block.clear();
            }
            continue;
        }
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 10, "line {line:?} must have exactly 10 fields");
        block.push(fields);
    }
    if !block.is_empty() {
        sentences.push(sentence(&block));
    }
    sentences
}

/// One block's word lines, with every multiword range merged into its
/// line; the part lines must follow their range line, one per id.
fn sentence(lines: &[Vec<&str>]) -> Sentence {
    let mut s = Sentence { forms: Vec::new(), tags: Vec::new(), plain: Vec::new() };
    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];
        if let Some((start, end)) = line[0].split_once('-') {
            let (start, end): (usize, usize) = (start.parse().unwrap(), end.parse().unwrap());
            assert!(start <= end, "multiword token id runs backwards: {}", line[0]);
            let mut upos: Vec<&str> = Vec::new();
            for (k, id) in (start..=end).enumerate() {
                let part =
                    lines.get(i + 1 + k).unwrap_or_else(|| panic!("multiword token {} has no line {id}", line[0]));
                assert_eq!(part[0], id.to_string(), "multiword token {} is not followed by line {id}", line[0]);
                if part[3] != "_" {
                    upos.push(part[3]);
                }
            }
            s.forms.push(line[1].to_string());
            s.tags.push(upos.join("+"));
            s.plain.push(false);
            i += 1 + (end - start + 1);
            continue;
        }
        s.forms.push(line[1].to_string());
        s.tags.push(line[3].to_string());
        s.plain.push(!line[0].contains('.'));
        i += 1;
    }
    s
}

#[test]
fn conllu_reading_follows_opennlp() {
    let text = "# sent_id = 1\n# text = Don't go\n1-2\tDon't\t_\t_\t_\t_\t_\t_\t_\t_\n\
                1\tDo\tdo\tAUX\tVBP\t_\t3\taux\t_\t_\n2\tn't\tnot\tPART\tRB\t_\t3\tadvmod\t_\t_\n\
                3\tgo\tgo\tVERB\tVB\t_\t0\troot\t_\t_\n3.1\t_\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
                1\tOk\tok\tINTJ\tUH\t_\t0\troot\t_\t_\n";
    let got = read_conllu(text);
    assert_eq!(
        got,
        [
            Sentence {
                forms: ["Don't", "go", "_"].map(String::from).to_vec(),
                tags: ["AUX+PART", "VERB", "_"].map(String::from).to_vec(),
                plain: vec![false, true, false],
            },
            Sentence { forms: vec!["Ok".into()], tags: vec!["INTJ".into()], plain: vec![true] },
        ]
    );
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

#[test]
#[ignore = "needs TURBO_UD_EWT_TEST naming a downloaded en_ewt-ud-test.conllu"]
fn english_ewt_test_split() {
    let conllu = env_path("TURBO_UD_EWT_TEST").expect("TURBO_UD_EWT_TEST names the treebank's test split");
    let text = std::fs::read_to_string(&conllu).unwrap_or_else(|e| panic!("{}: {e}", conllu.display()));
    let sentences = read_conllu(&text);
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/opennlp-en-ud-ewt-pos-1.3-2.5.4.bin");
    let tagger = PosTagger::new(PosModel::read(&model).unwrap());
    let jvm_text = env_path("TURBO_UD_EWT_JVM_TAGS")
        .map(|p| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())));
    let jvm: Option<Vec<Vec<&str>>> =
        jvm_text.as_deref().map(|t| t.lines().map(|l| l.split(' ').filter(|t| !t.is_empty()).collect()).collect());
    if let Some(jvm) = &jvm {
        assert_eq!(jvm.len(), sentences.len(), "the JVM tagged another number of sentences");
    }
    let (mut tokens, mut correct, mut plain, mut plain_correct, mut same_as_jvm) = (0, 0, 0, 0, 0);
    let (mut tags, mut probs) = (Vec::new(), Vec::new());
    for (n, s) in sentences.iter().enumerate() {
        let words: Vec<&str> = s.forms.iter().map(String::as_str).collect();
        tags.clear();
        probs.clear();
        assert!(tagger.tag(&words, &mut tags, &mut probs).unwrap(), "sentence {n} has no tag sequence");
        assert_eq!(tags.len(), words.len());
        for (i, tag) in tags.iter().enumerate() {
            let label = tagger.pos_model().tagset().label(*tag);
            let right = label == s.tags[i];
            tokens += 1;
            correct += usize::from(right);
            if s.plain[i] {
                plain += 1;
                plain_correct += usize::from(right);
            }
            if let Some(jvm) = &jvm {
                assert_eq!(label, jvm[n][i], "sentence {n} token {i} {:?}: the JVM's tag", words[i]);
                same_as_jvm += 1;
            }
        }
    }
    println!("sentences\t{}", sentences.len());
    println!("tokens\t{tokens}\tcorrect\t{correct}\taccuracy\t{:.6}", correct as f64 / tokens as f64);
    println!(
        "plain word lines\t{plain}\tcorrect\t{plain_correct}\taccuracy\t{:.6}",
        plain_correct as f64 / plain as f64
    );
    if jvm.is_some() {
        println!("tags equal to the JVM's\t{same_as_jvm}\tof\t{tokens}");
    }
}
