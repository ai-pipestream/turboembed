mod support;
use turbo_analysis::{Ix, ViewId};
use turbo_analysis_layers::{Tokens, Words};
use turbo_depparse::*;
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;

fn fields<'a>(lines: &mut impl Iterator<Item = &'a str>, kind: &str) -> Vec<&'a str> {
    let mut row = lines.next().unwrap().split('\t');
    assert_eq!(row.next(), Some(kind));
    row.collect()
}
fn heads(row: Vec<&str>) -> Vec<Option<Ix<Tokens>>> {
    row.into_iter()
        .map(|v| {
            let i = v.parse::<i32>().unwrap();
            if i < 0 { None } else { Some(Ix::new(i as u32)) }
        })
        .collect()
}
#[test]
fn pinned_jvm_oracle_context_and_predictions_match() {
    let mut lines = include_str!("data/reference.tsv").lines().peekable();
    let parser = support::parser();
    let ties = DependencyParserME::new(
        DependencyModel::from_raw(&support::raw(&["RIGHT_ARC:dep", "SHIFT"], "never", &[0.0, 0.0])).unwrap(),
    );
    let mut samples = 0;
    let mut contexts = 0;
    while lines.peek().is_some() {
        let words = fields(&mut lines, "W");
        let tags = fields(&mut lines, "P");
        let h = heads(fields(&mut lines, "GH"));
        let rel = fields(&mut lines, "GR").into_iter().map(str::to_owned).collect();
        let gold = DependencyGraph::<Tokens>::new(h, rel).unwrap();
        let count = fields(&mut lines, "O")[0].parse::<usize>().unwrap();
        if count == 0 {
            assert!(!gold.is_projective());
            assert!(ArcStandardOracle::transitions(&gold).is_err());
        } else {
            let steps = ArcStandardOracle::transitions(&gold).unwrap();
            assert_eq!(steps.len(), count);
            let mut state = ArcStandardState::<Tokens>::new(words.len()).unwrap();
            for step in steps {
                let expected = fields(&mut lines, "T");
                assert_eq!(step.encode(), expected[0]);
                assert_eq!(DependencyContextGenerator::context(&state, &words, &tags).unwrap(), expected[1..]);
                state.apply(&step).unwrap();
                contexts += 1;
            }
        }
        for (p, prefix) in [(&parser, "M"), (&ties, "Z")] {
            let expected = heads(fields(&mut lines, &format!("{prefix}H")));
            let rels = fields(&mut lines, &format!("{prefix}R"));
            let g = p.parse::<Tokens>(&words, &tags).unwrap();
            assert_eq!(g.to_layer().head, expected);
            for (i, r) in rels.iter().enumerate() {
                assert_eq!(g.arc(Ix::new(i as u32)).unwrap().relation, *r);
            }
        }
        samples += 1;
    }
    assert_eq!(samples, 7);
    assert_eq!(contexts, 152);
}
#[test]
fn ud_samples_keep_offsets_ranges_and_fields_through_graph_and_layer() {
    let mut multiwords = 0;
    for source in [include_str!("data/ud-en.conllu"), include_str!("data/ud-es.conllu")] {
        for block in source.trim_end().split("\n\n") {
            let data = ConlluDependencySample::read_sentence(block, Tagset::U, WhitespaceMode::Unicode, ViewId::SOURCE)
                .unwrap();
            multiwords += data.multiword_tokens.len();
            for (word, span) in data.sample.tokens().iter().zip(&data.word_spans) {
                assert_eq!(&block[span.range()], word.as_str());
            }
            let sample = DependencySample::<Words>::from_conllu(data.clone()).unwrap();
            let layer = sample.graph().to_layer();
            assert_eq!(sample.with_layer(layer).unwrap().to_conllu().unwrap(), data);
        }
    }
    assert_eq!(multiwords, 3);
}
