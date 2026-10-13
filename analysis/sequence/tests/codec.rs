//! `BioCodec`, `BilouCodec` and their validators against the JVM
//! (`golden/src/CodecGolden.java`).

mod common;

use common::{body, escape, unescape};
use turbo_sequence::codec::{BilouValidator, BioValidator};
use turbo_sequence::{BilouCodec, BioCodec, SequenceCodec, SequenceValidator, Span};

fn span_text(s: &Span) -> String {
    format!("{} {} {}", s.start, s.end, escape(s.kind.as_deref().unwrap_or("null")))
}

fn parse_span(s: &str) -> Span {
    let mut parts = s.splitn(3, ' ');
    let start = parts.next().unwrap().parse().unwrap();
    let end = parts.next().unwrap().parse().unwrap();
    let kind = unescape(parts.next().unwrap());
    Span::new(start, end, (kind != "null").then_some(kind.as_str()))
}

fn check(file: &str, component: &str, codec: &dyn SequenceCodec, validator: &dyn SequenceValidator<&str>) {
    let text = common::read(file);
    let mut failures = Vec::new();
    let mut counts = [0usize; 4];
    for line in body(&text, component) {
        let (left, right) = line.split_once("\t=>").unwrap();
        let right = right.strip_prefix('\t').unwrap_or(right);
        let mut fields = left.split('\t');
        let op = fields.next().unwrap();
        let args: Vec<String> = fields.map(unescape).collect();
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let actual = match op {
            "decode" => {
                counts[0] += 1;
                codec.decode(&args).iter().map(span_text).collect::<Vec<_>>().join("\t")
            }
            "valid" => {
                counts[1] += 1;
                let (outcome, _) = right.split_once('\t').unwrap();
                let outcome = unescape(outcome);
                let tokens = vec!["w"; args.len() + 1];
                let ok = validator.valid_sequence(args.len(), &tokens, &args, &outcome);
                format!("{}\t{}", escape(&outcome), u8::from(ok))
            }
            "encode" => {
                counts[2] += 1;
                let length: usize = args[0].parse().unwrap();
                let spans: Vec<Span> = args[1..].iter().map(|s| parse_span(s)).collect();
                codec.encode(&spans, length).iter().map(|o| escape(o)).collect::<Vec<_>>().join("\t")
            }
            "compatible" => {
                counts[3] += 1;
                u8::from(codec.are_outcomes_compatible(&args)).to_string()
            }
            other => panic!("unknown line {other}"),
        };
        if actual != right {
            failures.push(format!("{line}\n  got {actual}"));
        }
    }
    assert!(counts.iter().all(|&c| c > 0), "{counts:?}");
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn bio() {
    check("expected/codec-bio.txt", "codec-bio", &BioCodec, &BioValidator);
}

#[test]
fn bilou() {
    check("expected/codec-bilou.txt", "codec-bilou", &BilouCodec, &BilouValidator);
}
