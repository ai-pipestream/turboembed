use turbo_analysis::{Span, ViewId};
use turbo_analysis_layers::Words;
use turbo_depparse::DependencySample;
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;
const SOURCE: &str = "# text = Voy al mar.\n1\tVoy\t_\tVERB\tV\t_\t0\troot\t_\t_\n2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_\n2\ta\t_\tADP\tA\t_\t4\tcase\t_\t_\n3\tel\t_\tDET\tD\t_\t4\tdet\t_\t_\n4\tmar\t_\tNOUN\tN\t_\t1\tobl\t_\t_\n";
#[test]
fn conllu_graph_layer_round_trip_preserves_mwt_fields_and_view_offsets() {
    let mut input =
        ConlluDependencySample::read_sentence(SOURCE, Tagset::U, WhitespaceMode::Unicode, ViewId(7)).unwrap();
    assert_eq!(input.multiword_tokens[0].words, 1..3);
    assert_eq!(input.multiword_tokens[0].form.as_str(), "al");
    assert_eq!(&SOURCE[input.multiword_tokens[0].form.span.range()], "al");
    input.word_spans = vec![Span::new(10, 13), Span::new(14, 16), Span::new(14, 16), Span::new(17, 20)];
    let sample = DependencySample::<Words>::from_conllu(input.clone()).unwrap();
    let layer = sample.graph().to_layer();
    let restored = sample.with_layer(layer).unwrap().to_conllu().unwrap();
    assert_eq!(restored, input);
}
#[test]
fn malformed_alignment_is_rejected_without_panics() {
    let input =
        ConlluDependencySample::read_sentence(SOURCE, Tagset::X, WhitespaceMode::Unicode, ViewId::SOURCE).unwrap();
    let mut bad = input.clone();
    bad.multiword_tokens[0].words = u32::MAX..u32::MAX;
    assert!(bad.check().is_err());
    let mut bad = input.clone();
    bad.word_spans.pop();
    assert!(DependencySample::<Words>::from_conllu(bad).is_err());
    let mut bad = input;
    bad.word_spans[0] = Span { start: 9, end: 2 };
    assert!(bad.check().is_err());
    assert!(
        ConlluDependencySample::read_sentence(
            &format!("{SOURCE}\n{SOURCE}"),
            Tagset::U,
            WhitespaceMode::Unicode,
            ViewId::SOURCE
        )
        .is_err()
    );
}
