//! Measured exact-span accuracy on licensed, hand-labelled PDF text.
//! See data/extracted-pdf/README.md for provenance and annotation policy.

use turbo_analysis::Span;
use turbo_goldens::{escape::unescape, hex::to_hex, sha256};
use turbo_normalize::{Artifact, ArtifactDetector, NoiseScorer};
use turbo_normalize_layers::ArtifactKind;

const SAMPLES: [(&str, &str, &str); 6] = [
    (
        "P19-1423",
        include_str!("../data/extracted-pdf/P19-1423.escaped"),
        "4a4c955ebcaac4f100a3eee0d9755b0febaa72394695e172f422f60715956f1c",
    ),
    (
        "2020.acl-main.1",
        include_str!("../data/extracted-pdf/2020.acl-main.1.escaped"),
        "61d8ea51cba932c21984731f870eb08efa34e280b91d20c81487577a5d8c86c8",
    ),
    (
        "2022.acl-long.14-mupdf",
        include_str!("../data/extracted-pdf/2022.acl-long.14-mupdf.escaped"),
        "ff84db26480ba60169e7e8c8a1803d2a297de852638a5a8f79609e540230c8b6",
    ),
    (
        "jmlr-25-0035",
        include_str!("../data/extracted-pdf/jmlr-25-0035.escaped"),
        "f902f5558cea9702dd1184e240f71109605b9fe9c1372be649e64c5d45becd83",
    ),
    (
        "leodar-111",
        include_str!("../data/extracted-pdf/leodar-111.escaped"),
        "51cde9a82c4603dcb3696521ce6f41221cfc026a4bb2b41c4ce4e8e022c3079b",
    ),
    (
        "leodar-116",
        include_str!("../data/extracted-pdf/leodar-116.escaped"),
        "116f31dc10e2ddc8075b02f548f5d2c39e226b9e98a7ae3b35b6692239fc43b5",
    ),
];

fn labels(id: &str, text: &str) -> Vec<Artifact> {
    let mut out = Vec::new();
    for line in include_str!("../data/extracted-pdf/labels.tsv").lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 4);
        assert!(SAMPLES.iter().any(|sample| sample.0 == fields[0]));
        if fields[0] != id {
            continue;
        }
        let start = fields[1].parse::<u32>().unwrap();
        let end = fields[2].parse::<u32>().unwrap();
        assert!(start < end);
        assert!(text.get(start as usize..end as usize).is_some());
        let artifact = Artifact { span: Span::new(start, end), kind: ArtifactKind::from_name(fields[3]).unwrap() };
        assert!(!out.contains(&artifact), "duplicate gold annotation");
        out.push(artifact);
    }
    assert!(out.windows(2).all(|w| w[0].span.end <= w[1].span.start));
    out
}

fn ratio(n: usize, d: usize) -> String {
    if d == 0 { "N/A".into() } else { format!("{:.4}", n as f64 / d as f64) }
}

#[test]
fn extracted_pdf_exact_span_report() {
    for (mode, detector) in [
        ("default", ArtifactDetector::default()),
        ("all", ArtifactDetector::builder().only(&ArtifactKind::ALL).build()),
    ] {
        let mut counts = [[0usize; 3]; 11];
        for (id, encoded, digest) in SAMPLES {
            let text = unescape(encoded.trim_end_matches('\n')).unwrap();
            assert_eq!(to_hex(&sha256(text.as_bytes())), digest);
            let gold = labels(id, &text);
            let found = detector.detect(&text);
            for (i, kind) in ArtifactKind::ALL.iter().enumerate() {
                let tp = found.iter().filter(|a| a.kind == *kind && gold.contains(a)).count();
                let fp = found.iter().filter(|a| a.kind == *kind && !gold.contains(a)).count();
                let missed: Vec<_> = gold.iter().filter(|a| a.kind == *kind && !found.contains(a)).collect();
                counts[i][0] += tp;
                counts[i][1] += fp;
                counts[i][2] += missed.len();
                for a in found.iter().filter(|a| a.kind == *kind && !gold.contains(a)) {
                    println!("{mode} {id} false positive: {a:?} {:?}", &text[a.span.range()]);
                }
                for a in missed {
                    println!("{mode} {id} false negative: {a:?}");
                }
            }
        }
        println!("mode\tkind\tTP\tFP\tFN\tprecision\trecall");
        for (kind, [tp, fp, missed]) in ArtifactKind::ALL.into_iter().zip(counts) {
            println!(
                "{mode}\t{}\t{tp}\t{fp}\t{missed}\t{}\t{}",
                kind.name(),
                ratio(tp, tp + fp),
                ratio(tp, tp + missed)
            );
        }
        // Measured against the fixed hand labels, including known false positives.
        let mut expected = [[0usize; 3]; 11];
        expected[0] = [1, 0, 0];
        expected[1] = [27, 0, 0];
        expected[4] = [4, 0, 0];
        expected[8] = [36, 0, 0];
        expected[9] = if mode == "default" { [0, 0, 2] } else { [2, 0, 0] };
        expected[10] = if mode == "default" { [0, 0, 6] } else { [6, 24, 0] };
        assert_eq!(counts, expected);
    }
}

#[test]
fn extracted_pdf_noise_negative_control() {
    // Neither page pair has a damaged ASCII word core or encoded binary payload.
    // Math font mapping damage is labelled as artifacts, not lexical gibberish.
    for (id, encoded, _) in SAMPLES[..2].iter().copied() {
        let text = unescape(encoded.trim_end_matches('\n')).unwrap();
        let found = NoiseScorer::default().score(&text, &[]).unwrap();
        for f in &found {
            println!("{id} noise false positive: {f:?} {:?}", &text[f.span.range()]);
        }
        println!("{id}\tnoise findings={}\tpositive support=0", found.len());
        let expected = if id == "P19-1423" { vec![Span::new(5548, 5553)] } else { vec![] };
        assert_eq!(found.iter().map(|f| f.span).collect::<Vec<_>>(), expected);
        for f in found {
            assert_eq!(&text[f.span.range()], "FFNNs");
            assert_eq!(f.severity, turbo_normalize_layers::NoiseSeverity::Damaged);
            assert_eq!(f.score, 0.5);
        }
    }
}

#[test]
fn corpus_gold_support_includes_replacement_and_recurring_headers() {
    let replacement = unescape(SAMPLES[2].1.trim_end_matches('\n')).unwrap();
    let gold = labels(SAMPLES[2].0, &replacement);
    assert_eq!(gold, vec![Artifact { span: Span::new(865, 868), kind: ArtifactKind::ReplacementChar }]);
    assert_eq!(&replacement[gold[0].span.range()], "\u{fffd}");
    let headers = unescape(SAMPLES[3].1.trim_end_matches('\n')).unwrap();
    let gold = labels(SAMPLES[3].0, &headers);
    assert_eq!(gold.iter().filter(|a| a.kind == ArtifactKind::Control).count(), 15);
    assert_eq!(gold.iter().filter(|a| a.kind == ArtifactKind::RepeatedLine).count(), 6);
    assert!(!gold.iter().any(|a| matches!(a.kind, ArtifactKind::Mojibake | ArtifactKind::ZeroWidth)));
}
