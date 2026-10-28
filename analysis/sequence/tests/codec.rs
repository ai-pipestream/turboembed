//! `BioCodec`, `BilouCodec` and their validators against the JVM (the
//! goldens of `CodecGolden.java` in `analysis/goldens-tool`).

mod common;

use common::{grouped, layer, strs, u32s};
use turbo_goldens::Golden;
use turbo_sequence::codec::{BilouValidator, BioValidator};
use turbo_sequence::{BilouCodec, BioCodec, SequenceCodec, SequenceValidator, Span};

/// The outcomes of each case of a layer pair: the case layer numbers the
/// cases, the outcome layer holds each case's outcomes.
fn case_outcomes(g: &Golden, cases: &str, outcomes: &str) -> Vec<Vec<String>> {
    let n = layer(g, cases).len();
    let l = layer(g, outcomes);
    let values = strs(l, "outcome");
    grouped(l, "case", n).into_iter().map(|rows| rows.into_iter().map(|r| values[r].clone()).collect()).collect()
}

/// The spans of each case.
fn case_spans(g: &Golden, cases: &str, spans: &str) -> Vec<Vec<Span>> {
    let n = layer(g, cases).len();
    let l = layer(g, spans);
    let (start, end, typed) = (u32s(l, "start"), u32s(l, "end"), u32s(l, "typed"));
    let kind = strs(l, "type");
    grouped(l, "case", n)
        .into_iter()
        .map(|rows| {
            rows.into_iter()
                .map(|r| Span::new(start[r] as usize, end[r] as usize, (typed[r] == 1).then_some(kind[r].as_str())))
                .collect()
        })
        .collect()
}

fn check(run: &str, codec: &dyn SequenceCodec, validator: &dyn SequenceValidator<&str>) {
    let g = common::read("codec", run, "codec", run);
    let mut failures = Vec::new();

    let decode = case_outcomes(&g, "turbo:decode", "turbo:decode-outcomes");
    let decoded = case_spans(&g, "turbo:decode", "turbo:decode-spans");
    for (c, outcomes) in decode.iter().enumerate() {
        let outcomes: Vec<&str> = outcomes.iter().map(String::as_str).collect();
        let actual = codec.decode(&outcomes);
        if actual != decoded[c] {
            failures.push(format!("decode {outcomes:?}: expected {:?}, got {actual:?}", decoded[c]));
        }
    }

    let priors = case_outcomes(&g, "turbo:valid", "turbo:valid-priors");
    let valid = layer(&g, "turbo:valid");
    let (prior_count, outcome, answer) = (u32s(valid, "priors"), strs(valid, "outcome"), u32s(valid, "valid"));
    for (c, prior) in priors.iter().enumerate() {
        assert_eq!(prior.len(), prior_count[c] as usize);
        let prior: Vec<&str> = prior.iter().map(String::as_str).collect();
        let tokens = vec!["w"; prior.len() + 1];
        let ok = validator.valid_sequence(prior.len(), &tokens, &prior, &outcome[c]);
        if ok != (answer[c] == 1) {
            failures.push(format!("valid {prior:?} then {}: expected {}, got {ok}", outcome[c], answer[c]));
        }
    }

    let lengths = u32s(layer(&g, "turbo:encode"), "length");
    let spans = case_spans(&g, "turbo:encode", "turbo:encode-spans");
    let encoded = case_outcomes(&g, "turbo:encode", "turbo:encode-outcomes");
    for (c, spans) in spans.iter().enumerate() {
        let actual: Vec<String> = codec.encode(spans, lengths[c] as usize).iter().map(|o| o.to_string()).collect();
        if actual != encoded[c] {
            failures.push(format!("encode {spans:?} over {}: expected {:?}, got {actual:?}", lengths[c], encoded[c]));
        }
    }

    let subsets = case_outcomes(&g, "turbo:compatible", "turbo:compatible-outcomes");
    let compatible = u32s(layer(&g, "turbo:compatible"), "compatible");
    for (c, subset) in subsets.iter().enumerate() {
        let subset: Vec<&str> = subset.iter().map(String::as_str).collect();
        let ok = codec.are_outcomes_compatible(&subset);
        if ok != (compatible[c] == 1) {
            failures.push(format!("compatible {subset:?}: expected {}, got {ok}", compatible[c]));
        }
    }

    for (what, n) in
        [("decode", decode.len()), ("valid", priors.len()), ("encode", spans.len()), ("compatible", subsets.len())]
    {
        assert!(n > 0, "no {what} cases");
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn bio() {
    check("bio", &BioCodec, &BioValidator);
}

#[test]
fn bilou() {
    check("bilou", &BilouCodec, &BilouValidator);
}
