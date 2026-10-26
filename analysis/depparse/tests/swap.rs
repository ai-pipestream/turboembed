use turbo_analysis::{Ix, ViewId};
use turbo_analysis_layers::{Tokens, Words};
use turbo_depparse::{ArcStandardOracle, ArcStandardState, DependencyGraph, DependencySample, SwapOracle, Transition};
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;

#[test]
fn swap_oracle_preserves_a_real_crossing_tree_and_source_fields() {
    let text = include_str!("data/ud-en-nonprojective.conllu");
    let sample =
        ConlluDependencySample::read_sentence(text, Tagset::U, WhitespaceMode::Unicode, ViewId::SOURCE).unwrap();
    let sample = DependencySample::<Words>::from_conllu(sample).unwrap();
    assert!(!sample.graph().is_projective());
    let steps = SwapOracle::transitions(sample.graph()).unwrap();
    assert!(steps.contains(&Transition::Swap));
    let mut state = ArcStandardState::<Words>::new(sample.tokens().len()).unwrap();
    for t in &steps {
        state.apply(t).unwrap();
    }
    assert_eq!(state.to_graph().unwrap().to_layer().head, sample.graph().to_layer().head);
    let gold = sample.graph().to_layer();
    let got = state.to_graph().unwrap().to_layer();
    for i in 0..gold.head.len() {
        assert_eq!(gold.relations.label(gold.relation[i]), got.relations.label(got.relation[i]));
    }
    assert!(state.action_count() <= state.action_limit());
    assert_eq!(sample.clone().with_layer(got).unwrap().to_conllu().unwrap(), sample.to_conllu().unwrap());
}

#[test]
fn all_valid_trees_through_five_nodes_replay_and_terminate() {
    let mut crossing = 0;
    for n in 1usize..=5 {
        for encoding in 0..(n + 1).pow(n as u32) {
            let mut value = encoding;
            let heads = (0..n)
                .map(|_| {
                    let h = value % (n + 1);
                    value /= n + 1;
                    (h != n).then(|| Ix::<Tokens>::new(h as u32))
                })
                .collect();
            let Ok(gold) = DependencyGraph::new(heads, vec!["dep".into(); n]) else { continue };
            let steps = SwapOracle::transitions(&gold).unwrap();
            let mut state = ArcStandardState::<Tokens>::new(n).unwrap();
            if gold.is_projective() {
                assert_eq!(steps, ArcStandardOracle::transitions(&gold).unwrap());
            } else {
                crossing += 1;
                assert!(steps.contains(&Transition::Swap));
            }
            for t in &steps {
                state.apply(t).unwrap();
            }
            assert!(state.is_terminal());
            assert_eq!(state.to_graph().unwrap().to_layer().head, gold.to_layer().head, "{encoding} / {n}");
            assert!(state.action_count() <= state.action_limit());
            assert!(!state.can_apply(&Transition::Swap));
        }
    }
    assert!(crossing > 100);
}

#[test]
fn swap_refuses_root_and_reverse_pair_and_preserves_buffer_order() {
    let mut state = ArcStandardState::<Tokens>::new(3).unwrap();
    assert!(!state.can_apply(&Transition::Swap));
    state.apply(&Transition::Shift).unwrap();
    assert!(!state.can_apply(&Transition::Swap));
    state.apply(&Transition::Shift).unwrap();
    state.apply(&Transition::Swap).unwrap();
    assert_eq!(state.buffer_size(), 2);
    assert!(state.buffer(0) == turbo_depparse::Position::Node(Ix::new(0)));
    assert!(state.buffer(1) == turbo_depparse::Position::Node(Ix::new(2)));
    state.apply(&Transition::Shift).unwrap();
    assert!(!state.can_apply(&Transition::Swap));
    assert!(state.apply(&Transition::Swap).is_err());
    assert!(DependencyGraph::<Tokens>::new(vec![Some(Ix::new(1)), Some(Ix::new(0))], vec!["dep".into(); 2]).is_err());
    assert!(DependencyGraph::<Tokens>::new(vec![Some(Ix::new(9))], vec!["dep".into()]).is_err());
}

#[test]
fn maximum_pair_swapping_terminates_at_the_quadratic_bound() {
    for n in 1..=16 {
        let mut state = ArcStandardState::<Tokens>::new(n).unwrap();
        let forms: Vec<_> = (0..n).map(|i| i.to_string()).collect();
        let forms: Vec<_> = forms.iter().map(String::as_str).collect();
        let tags = vec!["NOUN"; n];
        while !state.is_terminal() {
            // Exercise feature extraction at reversed buffer/stack indexes too.
            turbo_depparse::DependencyContextGenerator::context(&state, &forms, &tags).unwrap();
            let action = [Transition::Swap, Transition::Shift, Transition::RightArc("dep".into())]
                .into_iter()
                .find(|action| state.can_apply(action))
                .unwrap();
            state.apply(&action).unwrap();
            assert!(state.action_count() <= state.action_limit());
        }
        assert_eq!(state.action_count(), state.action_limit());
        state.to_graph().unwrap().check_tree().unwrap();
        assert!(state.apply(&Transition::Shift).is_err());
    }
    assert!(ArcStandardState::<Tokens>::new(0).is_err());
}
