//! Cross-revision release harness; the same file compiles on the main baseline.
use std::{hint::black_box, time::Instant};
use turbo_analysis::{Ix, Layer, Spans, ViewId};
use turbo_analysis_layers::Numbers;
use turbo_goldens::data_dir;
use turbo_numeric::{NumberNotation, NumberScanner};
fn reset(out: &mut Numbers) {
    out.spans = Spans::new(ViewId::SOURCE);
    out.numeral_start.clear();
    out.numeral_end.clear();
    out.negative.clear();
    out.coefficient = Default::default();
    out.scale.clear();
    out.multiplier.clear();
    out.marker.clear();
    out.notation.clear();
}
#[test]
#[ignore = "release timing receipt; run both revision binaries interleaved"]
fn paired_release() {
    let mut ascii_corpus = String::new();
    for name in ["prose-english.txt", "prose-german.txt", "decimals-us.txt", "eu-grouping.txt", "both-notations.txt"] {
        let s = std::fs::read_to_string(data_dir().join("inputs/num").join(name)).unwrap();
        for _ in 0..200 {
            ascii_corpus.push_str(&s);
            ascii_corpus.push('\n');
        }
    }

    let mut non_latin_corpus = String::new();
    let non_latin_sample = concat!(
        "भारत में कुल जनसंख्या और बजट अनुमान 125,000 करोड़ रुपये तथा 24.5 प्रतिशत वृद्धि दर्ज की गई।\n",
        "ประเทศไทยมีประชากร लगभग 66 ล้านคน และงบประมาณ 3,500.75 ล้านบาทในปีนี้\n",
        "بلغ إجمالي الإنتاج 12,345.67 طن متري في الربع الثالث من العام.\n",
        "東京都の統計データ（2026年度）：売上高は1,250.5億円、前年比＋12.4％を達成しました！😀🚀📦\n",
    );
    for _ in 0..2_500 {
        non_latin_corpus.push_str(non_latin_sample);
    }

    let mut non_latin_nd_corpus = String::new();
    let non_latin_nd_sample = concat!(
        "भारत में कुल जनसंख्या और बजट अनुमान १,२५,००० करोड़ रुपये तथा २४.५ प्रतिशत वृद्धि दर्ज की गई।\n",
        "ประเทศไทยมีประชากร लगभग 66 ล้านคน และงบประมาณ 3,500.75 ล้านบาทในปีนี้\n",
        "بلغ إجمالي الإنتاج ١٢٬٣٤٥٫٦٧ طن متري في الربع الثالث من العام.\n",
        "東京都の統計データ（２０２６年度）：売上高は１，２５０．５億円、前年比＋12.4％を達成しました！😀🚀📦\n",
    );
    for _ in 0..2_500 {
        non_latin_nd_corpus.push_str(non_latin_nd_sample);
    }

    let which = std::env::var("TURBO_BENCH_CORPUS").unwrap_or("ascii".into());
    let corpus = match which.as_str() {
        "ascii" => ascii_corpus,
        "non_latin" => non_latin_corpus,
        "non_latin_nd" => non_latin_nd_corpus,
        _ => panic!("unknown corpus"),
    };
    if let Ok(dir) = std::env::var("TURBO_BENCH_DUMP") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(format!("{which}.txt")), &corpus).unwrap();
    }
    let scanner = NumberScanner::new(NumberNotation::LatinUs);
    let mut out = Numbers::new(ViewId::SOURCE);
    for _ in 0..10 {
        reset(&mut out);
        scanner.scan(black_box(&corpus), &mut out);
        black_box(out.len());
    }
    let iters: usize = std::env::var("TURBO_BENCH_ITERS").map_or(100, |s| s.parse().unwrap());
    let t = Instant::now();
    for _ in 0..iters {
        reset(&mut out);
        scanner.scan(black_box(&corpus), &mut out);
        black_box(out.len());
    }
    let elapsed = t.elapsed();
    out.check().unwrap();
    let mut digest = turbo_analysis::Fnv1a::new();
    for i in 0..out.len() {
        let ix = Ix::new(i as u32);
        let row = format!(
            "{:?}",
            (
                out.spans.get(ix),
                out.numeral_start[i],
                out.numeral_end[i],
                out.negative[i],
                out.value(ix).to_plain_string(),
                out.scale[i],
                out.multiplier[i],
                out.marker[i],
                out.notation[i]
            )
        );
        digest.bytes(row.as_bytes());
    }
    let digest = digest.finish();
    println!(
        "corpus={which} bytes={} iters={iters} ms={:.6} mentions={} output_fnv64={digest:016x}",
        corpus.len(),
        elapsed.as_secs_f64() * 1000.,
        out.len()
    );
}
