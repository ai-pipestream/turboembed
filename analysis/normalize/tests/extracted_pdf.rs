//! Measured exact-span accuracy on licensed, independently labelled PDF text.
//! See data/extracted-pdf/README.md for provenance and annotation policy.

use turbo_analysis::Span;
use turbo_goldens::{escape::unescape, hex::to_hex, sha256};
use turbo_normalize::{Artifact, ArtifactDetector, NoiseScorer};
use turbo_normalize_layers::ArtifactKind;

const SAMPLES: [(&str, &str, &str); 2] = [
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
        // Baseline measurements, not generated labels. Change only after review.
        assert_eq!(counts.iter().map(|c| c[0]).sum::<usize>(), 15);
        assert_eq!(counts.iter().map(|c| c[1]).sum::<usize>(), 0);
        assert_eq!(counts.iter().map(|c| c[2]).sum::<usize>(), 0);
    }
}

#[test]
fn extracted_pdf_noise_negative_control() {
    // Neither page pair has a damaged ASCII word core or encoded binary payload.
    // Math font mapping damage is labelled as artifacts, not lexical gibberish.
    for (id, encoded, _) in SAMPLES {
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
