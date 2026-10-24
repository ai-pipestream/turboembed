//! Case-sensitive accuracy and Italian error strata on pinned UD r2.16.
//! Usage: ud_case_eval <treebanks> <native-models> <legacy-models> <references> [language]
//! Emits aggregate counts and hashes only, never corpus text.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use turbo_formats::LemmaSample;
use turbo_lemmatizer::{LemmaEncoding, LemmatizerModel, MaxentLemmatizer, TokenTag, edit_script::lowercase_units};
use turbo_maxent::ModelFile;
use turbo_postag::{PosModel, PosTagger};
use turbo_train_components::ud::Component;
use turbo_train_recipes::ud::{self, UD_RECIPES};
use turbo_zoo::Sha256;

#[derive(Default)]
struct Accuracy {
    right: usize,
    total: usize,
}
impl Accuracy {
    fn add(&mut self, right: bool) {
        self.right += usize::from(right);
        self.total += 1;
    }
    fn percent(&self) -> f64 {
        if self.total == 0 { 0.0 } else { 100.0 * self.right as f64 / self.total as f64 }
    }
}

fn capital(lemma: &str) -> bool {
    lemma.encode_utf16().ne(lowercase_units(lemma))
}
fn model_path(root: &Path, id: &str) -> PathBuf {
    root.join(id).join(format!("{id}.bin"))
}

fn score(
    id: &str,
    side: &str,
    bytes: &[u8],
    samples: &[LemmaSample<'_>],
    vocab: &HashSet<&str>,
) -> Result<(), Box<dyn Error>> {
    let model = LemmatizerModel::open(ModelFile::open(bytes.to_vec())?)?;
    let expected = if side == "native" { LemmaEncoding::CaseSensitiveV1 } else { LemmaEncoding::OpenNlp };
    assert_eq!(model.encoding(), expected);
    let lemmatizer = MaxentLemmatizer::new(model);
    let mut counts: BTreeMap<String, Accuracy> = BTreeMap::new();
    for s in samples {
        let words: Vec<_> =
            s.tokens().iter().zip(s.tags()).map(|(w, t)| TokenTag { token: w.as_str(), tag: t.as_str() }).collect();
        let (mut classes, mut probs) = (Vec::new(), Vec::new());
        assert!(lemmatizer.classes(&words, &mut classes, &mut probs));
        assert_eq!(classes.len(), words.len());
        for ((word, class), gold) in words.iter().zip(classes).zip(s.lemmas()) {
            let lemma = lemmatizer.lemmatizer_model().encoding().decode(word.token, class)?;
            let right = lemma == gold.as_str();
            counts.entry("all".into()).or_default().add(right);
            counts.entry(if vocab.contains(word.token) { "seen" } else { "unseen" }.into()).or_default().add(right);
            counts.entry(format!("tag:{}", word.tag)).or_default().add(right);
            if capital(gold.as_str()) {
                counts.entry("capital".into()).or_default().add(right);
                if word.tag == "PROPN" {
                    counts.entry("capital-PROPN".into()).or_default().add(right);
                }
            }
        }
    }
    println!("model,{id},{side},{}", Sha256::of(bytes));
    for (stratum, c) in counts {
        println!("accuracy,{id},{side},{stratum},{},{},{:.8}", c.right, c.total, c.percent());
    }
    Ok(())
}

fn italian_pos(treebanks: &Path, native: &Path, references: &Path) -> Result<(), Box<dyn Error>> {
    let r = UD_RECIPES.iter().find(|r| r.treebank.language == "it" && r.component == Component::Pos).unwrap();
    let text = r.treebank.read(&treebanks.join(r.treebank.dataset()?.id.as_str()))?;
    let (train, _) = ud::pos_samples(&text.train)?;
    let (test, _) = ud::pos_samples(&text.test)?;
    let vocab: HashSet<&str> = train.iter().flat_map(|s| s.tokens().iter().map(|t| t.as_str())).collect();
    for (side, path) in
        [("native", model_path(native, r.model_id)), ("reference", references.join(r.reference.file()?))]
    {
        let bytes = std::fs::read(path)?;
        if side == "reference" {
            r.reference.check(&bytes)?;
        }
        let model = PosModel::open_with_format(ModelFile::open(bytes)?, ud::TAG_FORMAT)?;
        let tagger = PosTagger::new(model);
        let mut train_counts = Accuracy::default();
        for sample in &train {
            let words: Vec<_> = sample.tokens().iter().map(|t| t.as_str()).collect();
            let (mut tags, mut probs) = (Vec::new(), Vec::new());
            assert!(tagger.tag(&words, &mut tags, &mut probs)?);
            for (gold, given) in sample.tags().iter().zip(tags) {
                train_counts.add(gold.as_str() == tagger.pos_model().tagset().label(given));
            }
        }
        println!("pos-training,{side},{},{},{:.8}", train_counts.right, train_counts.total, train_counts.percent());
        let mut counts: BTreeMap<String, Accuracy> = BTreeMap::new();
        let mut errors: BTreeMap<(String, String), usize> = BTreeMap::new();
        for s in &test {
            let words: Vec<_> = s.tokens().iter().map(|t| t.as_str()).collect();
            let (mut tags, mut probs) = (Vec::new(), Vec::new());
            assert!(tagger.tag(&words, &mut tags, &mut probs)?);
            for ((word, gold), given) in words.iter().zip(s.tags()).zip(tags) {
                let given = tagger.pos_model().tagset().label(given);
                let right = given == gold.as_str();
                counts.entry("all".into()).or_default().add(right);
                counts.entry(if vocab.contains(word) { "seen" } else { "unseen" }.into()).or_default().add(right);
                counts.entry(format!("tag:{}", gold.as_str())).or_default().add(right);
                if !right {
                    *errors.entry((gold.as_str().to_owned(), given.to_owned())).or_default() += 1;
                }
            }
        }
        for (stratum, c) in counts {
            println!("pos,{side},{stratum},{},{},{:.8}", c.right, c.total, c.percent());
        }
        let mut errors: Vec<_> = errors.into_iter().collect();
        errors.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for ((gold, given), n) in errors.into_iter().take(10) {
            println!("pos-error,{side},{gold},{given},{n}");
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: ud_case_eval <treebanks> <native-models> <legacy-models> <references> [language]".into());
    }
    let (treebanks, native, legacy, references) =
        (Path::new(&args[0]), Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]));
    let language = args.get(4).map(|v| v.to_str().ok_or("language is not UTF-8")).transpose()?;
    let mut selected = 0;
    for r in UD_RECIPES
        .iter()
        .filter(|r| r.component == Component::Lemmas && language.is_none_or(|l| r.treebank.language == l))
    {
        selected += 1;
        let text = r.treebank.read(&treebanks.join(r.treebank.dataset()?.id.as_str()))?;
        let (train, train_skipped) = ud::lemma_samples(&text.train)?;
        let (test, test_skipped) = ud::lemma_samples(&text.test)?;
        assert_eq!((train_skipped, test_skipped), (0, 0));
        if r.treebank.language == "it" {
            for (split, samples) in [("train", &train), ("test", &test)] {
                let (mut punctuation, mut label) = (0, 0);
                for sample in samples {
                    for (tag, lemma) in sample.tags().iter().zip(sample.lemmas()) {
                        if tag.as_str() == "PUNCT" {
                            punctuation += 1;
                            label += usize::from(lemma.as_str() == "[PUNCT]");
                        }
                    }
                }
                println!("punctuation,it,{split},{label},{punctuation}");
            }
        }
        let mut roundtrips = 0;
        for s in train.iter().chain(&test) {
            for (word, lemma) in s.tokens().iter().zip(s.lemmas()) {
                let encoding = LemmaEncoding::CaseSensitiveV1;
                let class = encoding.encode(word.as_str(), lemma.as_str());
                assert_eq!(encoding.decode(word.as_str(), &class)?, lemma.as_str());
                roundtrips += 1;
            }
        }
        let vocab: HashSet<&str> = train.iter().flat_map(|s| s.tokens().iter().map(|t| t.as_str())).collect();
        let train_tokens: usize = train.iter().map(|s| s.len()).sum();
        let test_tokens: usize = test.iter().map(|s| s.len()).sum();
        println!(
            "data,{},{},{},{},{},{roundtrips},{},{}",
            r.model_id,
            train.len(),
            train_tokens,
            test.len(),
            test_tokens,
            r.treebank.train.sha256,
            r.treebank.test.sha256
        );
        for (side, path) in [
            ("native", model_path(native, r.model_id)),
            ("legacy", model_path(legacy, r.model_id)),
            ("reference", references.join(r.reference.file()?)),
        ] {
            let bytes = std::fs::read(path)?;
            if side == "reference" {
                r.reference.check(&bytes)?;
            }
            score(r.model_id, side, &bytes, &test, &vocab)?;
        }
    }
    assert!(selected > 0, "language selected no recipes");
    if language.is_none_or(|l| l == "it") {
        italian_pos(treebanks, native, references)?;
    }
    Ok(())
}
