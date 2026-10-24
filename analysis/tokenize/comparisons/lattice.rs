//! Standalone real-IPADIC comparison; dependencies and commands are documented
//! in docs/lattice-prebuilt.md, outside the workspace dependency graph.
use lindera::{dictionary::load_fs_dictionary, mode::Mode, segmenter::Segmenter};
use lindera_dictionary::{builder::DictionaryBuilder, dictionary::metadata::Metadata, viterbi::Lattice};
use std::{
    borrow::Cow,
    fs::{self, File},
    hint::black_box,
    io::{BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};
use turbo_tokenize::lattice::{PrebuiltImage, Scratch};
use turbo_tokenize::{LatticeTokenizer, MecabDictionary, WhitespaceMode};

fn mapped(path: &Path) -> MecabDictionary {
    // SAFETY: the benchmark creates this file before loading, retains the
    // mapping in the image owner, and never edits or truncates it while mapped.
    let mmap = unsafe { memmap2::Mmap::map(&File::open(path).unwrap()).unwrap() };
    let image = unsafe { PrebuiltImage::from_owner(Arc::new(mmap)) };
    MecabDictionary::from_prebuilt(image, Default::default()).unwrap()
}
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let now = Instant::now();
    let value = f();
    (value, now.elapsed().as_secs_f64() * 1000.0)
}
fn report(name: &str, mut samples: Vec<f64>) {
    println!("{name} samples_ms={samples:?}");
    samples.sort_by(f64::total_cmp);
    println!("{name} median_ms={:.6}", samples[samples.len() / 2]);
}
fn text(dir: &Path) -> String {
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut one = String::new();
    for p in files {
        one.push_str(&fs::read_to_string(p).unwrap());
        one.push('\n');
    }
    assert!(!one.is_empty());
    one.repeat((10usize << 20).div_ceil(one.len()))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "usage: lattice SOURCE WORK INPUTS_JA");
    let source = Path::new(&args[1]);
    let work = Path::new(&args[2]);
    fs::create_dir_all(work).unwrap();
    let corpus = text(Path::new(&args[3]));
    println!("text bytes={} sha256={}", corpus.len(), turbo_maxent::sha256(corpus.as_bytes()).to_hex());
    let path = work.join("turbo.bin");
    let (dictionary, source_ms) = timed(|| MecabDictionary::load(source, WhitespaceMode::Unicode).unwrap());
    println!(
        "source_sha256={} entries={} surfaces={}",
        dictionary.sha256().to_hex(),
        dictionary.lexicon_len(),
        dictionary.surfaces()
    );
    let (_, write_ms) = timed(|| dictionary.write_prebuilt(&path).unwrap());
    println!(
        "turbo_build_ms={:.6} image_bytes={} image_sha256={}",
        source_ms + write_ms,
        fs::metadata(&path).unwrap().len(),
        turbo_maxent::sha256(&fs::read(&path).unwrap()).to_hex()
    );

    let vibrato_path = work.join("vibrato.bin");
    let (_, build_ms) = timed(|| {
        let mut files: Vec<_> = fs::read_dir(source)
            .unwrap()
            .map(|p| p.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "csv"))
            .collect();
        files.sort();
        let mut lex = Vec::new();
        for p in files {
            lex.extend(fs::read(p).unwrap());
        }
        let d = vibrato::SystemDictionaryBuilder::from_readers(
            lex.as_slice(),
            File::open(source.join("matrix.def")).unwrap(),
            File::open(source.join("char.def")).unwrap(),
            File::open(source.join("unk.def")).unwrap(),
        )
        .unwrap();
        let mut out = BufWriter::new(File::create(&vibrato_path).unwrap());
        d.write(&mut out).unwrap();
        out.flush().unwrap();
    });
    println!("vibrato_build_ms={build_ms:.6}");
    let lindera_path = work.join("lindera");
    let (_, build_ms) =
        timed(|| DictionaryBuilder::new(Metadata::default()).build_dictionary(source, &lindera_path).unwrap());
    println!("lindera_build_ms={build_ms:.6}");

    // One warm load per engine, then nine interleaved loads. Destruction is
    // outside the timer; OS page cache remains warm. No cache-flush claims.
    let mut loads: [Vec<f64>; 5] = Default::default();
    for i in 0..10 {
        for j in 0..5 {
            let kind = (j + i) % 5;
            let ms = match kind {
                0 => timed(|| MecabDictionary::load(source, WhitespaceMode::Unicode).unwrap()).1,
                1 => timed(|| MecabDictionary::load_prebuilt(&path, Default::default()).unwrap()).1,
                2 => timed(|| mapped(&path)).1,
                3 => timed(|| load_fs_dictionary(&lindera_path).unwrap()).1,
                _ => timed(|| vibrato::Dictionary::read(BufReader::new(File::open(&vibrato_path).unwrap())).unwrap()).1,
            };
            if i != 0 {
                loads[kind].push(ms);
            }
        }
    }
    let names = ["source", "prebuilt-read", "prebuilt-mmap", "lindera", "vibrato"];
    for (name, samples) in names.iter().zip(loads) {
        report(&format!("load-{name}"), samples);
    }
    let tokenizers = [dictionary, MecabDictionary::load_prebuilt(&path, Default::default()).unwrap(), mapped(&path)]
        .map(|d| LatticeTokenizer::new(d, WhitespaceMode::Unicode));
    let mut expected = Vec::new();
    tokenizers[0].analyze(&corpus, &mut expected).unwrap();
    for tokenizer in &tokenizers[1..] {
        let mut found = Vec::new();
        tokenizer.analyze(&corpus, &mut found).unwrap();
        assert_eq!(found, expected);
        for token in &found {
            assert!(tokenizer.dictionary().features(token.entry).eq(tokenizers[0].dictionary().features(token.entry)));
        }
    }
    println!("turbo_morphemes={}", expected.len());
    drop(expected);
    let lindera = Segmenter::new(Mode::Normal, load_fs_dictionary(&lindera_path).unwrap(), None);
    let vibrato =
        vibrato::Tokenizer::new(vibrato::Dictionary::read(BufReader::new(File::open(&vibrato_path).unwrap())).unwrap())
            .ignore_space(true)
            .unwrap();
    let mut worker = vibrato.new_worker();
    let mut lattice = Lattice::default();
    let mut scratch = Scratch::new();
    let mut output = Vec::new();
    let mut runs: [Vec<f64>; 5] = Default::default();
    for i in 0..10 {
        for j in 0..5 {
            let kind = (j + i) % 5;
            let (count, ms) = timed(|| {
                if kind < 3 {
                    output.clear();
                    tokenizers[kind].analyze_with(black_box(&corpus), &mut scratch, &mut output).unwrap();
                    output.len()
                } else if kind == 3 {
                    lindera.segment_with_lattice(Cow::Borrowed(black_box(&corpus)), &mut lattice).unwrap().len()
                } else {
                    let mut count = 0;
                    for s in black_box(&corpus).split_whitespace() {
                        worker.reset_sentence(s);
                        worker.tokenize();
                        count += worker.num_tokens();
                    }
                    count
                }
            });
            if i == 0 {
                println!("{}_morphemes={count}", names[kind]);
            }
            if i != 0 {
                runs[kind].push(ms);
            }
        }
    }
    for (name, samples) in names.iter().zip(runs) {
        report(&format!("tokenize-{name}"), samples);
    }
}
