//! Reproducible MASC fixture comparison using the repository's document
//! split, exclusions and corrected named-entity sample reader.
#[path = "crf_support/gradient.rs"]
mod gradient;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;
use turbo_formats::NameSample;
use turbo_formats::masc::MascCorpus;
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{CrfNameFinder, FoundName, MODEL_ENTRY_NAME, NameCodec, NameFinder};
use turbo_train::crf::CrfTrainer;
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::namefind::{TokenNameFinderFactory, train_crf_name_model, train_name_model};
use turbo_train_components::package::Stamp;
use turbo_train_recipes::{masc, namefind::read_samples, spans::SpanTypes};

#[derive(Default, Debug)]
struct Counts {
    correct: usize,
    predicted: usize,
    gold: usize,
}
impl Counts {
    fn update(&mut self, s: &NameSample<'_>, names: &[FoundName]) {
        let gold: HashSet<_> =
            s.names().iter().map(|n| (n.start as usize, n.end as usize, n.kind.unwrap_or("default"))).collect();
        let predicted: HashSet<_> =
            names.iter().map(|n| (n.span.start, n.span.end, n.span.kind.as_deref().unwrap_or("default"))).collect();
        self.correct += gold.intersection(&predicted).count();
        self.predicted += predicted.len();
        self.gold += gold.len();
    }
    fn f1(&self) -> f64 {
        if self.predicted + self.gold == 0 {
            0.0
        } else {
            2.0 * self.correct as f64 / (self.predicted + self.gold) as f64
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: crf_namefind MASC_DIRECTORY [bio|bilou]")?;
    let codec = match std::env::args().nth(2).as_deref() {
        None | Some("bio") => NameCodec::Bio,
        Some("bilou") => NameCodec::Bilou,
        _ => return Err("codec".into()),
    };
    let root = Path::new(&path);
    let digest = masc::corpus_digest(root)?;
    if !masc::RELEASE.is(&digest) {
        return Err("MASC corpus digest differs from recorded release".into());
    }
    let split = masc::SPLIT.apply(MascCorpus::read(root)?, root)?;
    let types = SpanTypes { types: &["person"], renamed: &[] };
    let (mut train, mut test) = (Vec::new(), Vec::new());
    read_samples(&split.train, &types, |s| {
        train.push(s);
        Ok(())
    })?;
    read_samples(&split.held_out, &types, |s| {
        test.push(s);
        Ok(())
    })?;
    // Spread the fixture across all corpus genres, without inspecting labels.
    let sample_evenly = |samples: Vec<_>, limit: usize| {
        let n = samples.len();
        samples
            .into_iter()
            .enumerate()
            .filter_map(|(i, s)| (n <= limit || i * limit / n != (i + 1) * limit / n).then_some(s))
            .collect::<Vec<_>>()
    };
    let train = sample_evenly(train, 2048);
    let test = sample_evenly(test, 512);
    println!(
        "training_entities={} held_out_entities={}",
        train.iter().map(|s| s.names().len()).sum::<usize>(),
        test.iter().map(|s| s.names().len()).sum::<usize>()
    );
    println!(
        "corpus_sha256={} train_samples={} test_samples={} codec={codec:?}",
        digest.sha256,
        train.len(),
        test.len()
    );
    gradient::check(&train)?;
    let factory = TokenNameFinderFactory::new("en", None, codec)?;
    let started = Instant::now();
    let crf = train_crf_name_model(
        &factory,
        &train,
        &CrfTrainer { codec, cutoff: 2, l2: 1.0, iterations: 100 },
        Stamp::at(0),
    )?;
    println!("CRF training_seconds={:.6}", started.elapsed().as_secs_f64());
    let file = ModelFile::open(crf.package.to_bytes())?;
    let mut finder = CrfNameFinder::from_file(&crf.model, &file, PatternMode::Utf16Units)?;
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 2);
    let started = Instant::now();
    let baseline =
        train_name_model(&train, &factory, None, &params, &mut TrainingConfiguration::default(), Stamp::at(0))?;
    println!("Maxent training_seconds={:.6}", started.elapsed().as_secs_f64());
    let file = ModelFile::open(baseline.package.to_bytes())?;
    let model = file.maxent(MODEL_ENTRY_NAME)?;
    let mut maxent = NameFinder::new(&file, &model, PatternMode::Utf16Units)?;
    let (mut c, mut m) = (Counts::default(), Counts::default());
    for s in &test {
        let words: Vec<_> = s.tokens().iter().map(|f| f.as_str()).collect();
        if s.clear_adaptive_data() {
            maxent.clear_adaptive_data();
        }
        c.update(s, &finder.find(&words)?);
        m.update(s, &maxent.find(&words));
    }
    println!("CRF {c:?} f1={:.9}", c.f1());
    println!("Maxent {m:?} f1={:.9}", m.f1());
    let words: Vec<Vec<_>> = test.iter().map(|s| s.tokens().iter().map(|t| t.as_str()).collect()).collect();
    for _ in 0..2 {
        for w in &words {
            std::hint::black_box(finder.find(w)?);
        }
    }
    for run in 0..5 {
        let start = Instant::now();
        for w in &words {
            std::hint::black_box(finder.find(w)?);
        }
        println!(
            "CRF decode_run={run} tokens={} seconds={:.9}",
            words.iter().map(Vec::len).sum::<usize>(),
            start.elapsed().as_secs_f64()
        );
    }
    if c.f1() + 1e-12 < m.f1() {
        return Err("CRF entity F1 is below Maxent on this fixture".into());
    }
    Ok(())
}
