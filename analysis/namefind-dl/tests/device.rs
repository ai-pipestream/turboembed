//! The finder on a device against the finder on the CPU with the same
//! `dslim/bert-base-NER` bundle: TURBO_NER_BUNDLE names the bundle,
//! TURBO_TEST_DEVICE the device (a backend name or a device index) and
//! TURBO_TEST_PRECISION its tier (model by default); the CPU runs at
//! model. Every text must give the same names, with the same types, and
//! each name's probability within the tier's bound of the CPU's.

use std::path::{Path, PathBuf};

use turbo_analysis::Annotator;
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_namefind_dl::{Decoding, NameFinderDl, NameFinderDlOptions};

/// The largest probability difference allowed at f32: the core's largest
/// absolute difference for an f32 session's values
/// (`turbo::record::tolerance`).
const F32_BOUND: f64 = 1e-4;

/// The largest probability difference allowed at f16 and bf16, whose
/// record bounds only the cosine; as for SaT's device test
/// (analysis/sat/tests/device.rs).
const F16_BOUND: f64 = 1e-2;

const TEXTS: [&str; 6] = [
    "Angela Merkel met Emmanuel Macron and Kristalina Georgieva in Paris to discuss the European Union budget.",
    "She lives in (Paris),France.",
    "Apple opened a store on Fifth Avenue in New York, said Tim Cook.",
    "The United Nations met in Geneva on Monday; Antonio Guterres spoke first.",
    "Lionel Messi joined Inter Miami from Paris Saint-Germain in 2023.",
    "no names here at all",
];

fn device() -> Option<Device> {
    let d = std::env::var("TURBO_TEST_DEVICE").ok()?;
    Some(match d.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(d),
    })
}

fn precision() -> Precision {
    match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("fastest") => Precision::Fastest,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model, fastest or exact"),
    }
}

fn open(dir: &Path, encoder: EncoderOptions, decoding: Decoding) -> NameFinderDl {
    NameFinderDl::open(dir, &NameFinderDlOptions { encoder, decoding, ..NameFinderDlOptions::default() })
        .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
#[ignore = "needs TURBO_NER_BUNDLE naming the dslim/bert-base-NER bundle and TURBO_TEST_DEVICE a device"]
fn a_device_finds_the_names_the_cpu_finds() {
    let dir = PathBuf::from(std::env::var("TURBO_NER_BUNDLE").expect("TURBO_NER_BUNDLE"));
    let Some(device) = device() else { panic!("TURBO_TEST_DEVICE names no device") };
    let inputs: Vec<_> = TEXTS.iter().map(|t| (*t, vec![(0, t.len())])).collect();
    for decoding in [Decoding::Word, Decoding::Piece] {
        let cpu = open(&dir, EncoderOptions::default(), decoding);
        let dev = open(
            &dir,
            EncoderOptions { device: device.clone(), precision: precision(), ..Default::default() },
            decoding,
        );
        let execution = dev.execution().unwrap();
        assert_ne!(execution.backend, "cpu", "{execution:?}");
        let bound = if execution.precision.starts_with("f32") { F32_BOUND } else { F16_BOUND };
        let (want, got) = (cpu.find(&inputs).unwrap(), dev.find(&inputs).unwrap());
        let mut largest = 0f64;
        for ((text, w), g) in TEXTS.iter().zip(&want).zip(&got) {
            let names = |c: &[turbo_namefind_dl::decode::Candidate]| {
                c.iter().map(|c| (c.start, c.end, c.kind)).collect::<Vec<_>>()
            };
            assert_eq!(names(w), names(g), "{decoding:?} on {execution:?}: {text:?}");
            for (a, b) in w.iter().zip(g) {
                let diff = (a.probability - b.probability).abs();
                assert!(diff <= bound, "{decoding:?}: {text:?} {:?}: {diff:e} over {bound}", &text[a.start..a.end]);
                largest = largest.max(diff);
            }
        }
        let n: usize = want.iter().map(Vec::len).sum();
        eprintln!("{decoding:?} on {execution:?}: {n} names, largest probability difference {largest:.1e}");
    }
}
