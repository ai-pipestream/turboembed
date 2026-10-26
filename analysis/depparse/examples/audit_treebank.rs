//! Structural audit only: oracle replay is not a learned-parser accuracy test.
use turbo_analysis::ViewId;
use turbo_analysis_layers::Words;
use turbo_depparse::{ArcStandardState, DependencySample, SwapOracle, Transition};
use turbo_formats::conllu::{ConlluDependencySample, Tagset, words::ConlluWordReader};
use turbo_unicode::WhitespaceMode;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let input = std::fs::read_to_string(&path)?;
        let (mut sentences, mut words, mut crossing, mut crossing_words, mut swaps, mut max_actions) =
            (0, 0, 0, 0, 0, 0);
        for sentence in ConlluWordReader::new(&input)? {
            let sentence = sentence?;
            let span = sentence.source;
            let sample = DependencySample::<Words>::from_conllu(ConlluDependencySample::read_sentence(
                &input[span.start as usize..span.end as usize],
                Tagset::U,
                WhitespaceMode::Unicode,
                ViewId::SOURCE,
            )?)?;
            sentences += 1;
            words += sample.tokens().len();
            if !sample.graph().is_projective() {
                crossing += 1;
                crossing_words += sample.tokens().len();
            }
            let mut state = ArcStandardState::<Words>::new(sample.tokens().len())?;
            for action in SwapOracle::transitions(sample.graph())? {
                swaps += usize::from(action == Transition::Swap);
                state.apply(&action)?;
            }
            max_actions = max_actions.max(state.action_count());
            let predicted = state.to_graph()?;
            for i in 0..sample.tokens().len() {
                let ix = turbo_analysis::Ix::new(i as u32);
                let (gold, got) = (sample.graph().arc(ix)?, predicted.arc(ix)?);
                if gold.head != got.head || gold.relation != got.relation {
                    return Err("oracle replay differs".into());
                }
            }
        }
        println!(
            "sha256={} sentences={sentences} words={words} nonprojective_sentences={crossing} nonprojective_words={crossing_words} swaps={swaps} max_actions={max_actions}",
            turbo_maxent::sha256(input.as_bytes()).to_hex()
        );
    }
    Ok(())
}
