//! The comparisons of one golden with the Rust taggers.

use turbo_goldens::Golden;
use turbo_postag::neural::feedforward::ABSENT;
use turbo_postag::neural::{
    BilstmPosModel, BilstmPosTagger, CrfScorer, FeedforwardPosContext, FeedforwardPosModel, FeedforwardPosTagger,
    NeuralSequence,
};

use crate::golden::*;
use crate::run::Run;

/// Runs `check` on every run, one thread each (the runs are independent),
/// and returns the results in run order. A panic in a thread is raised
/// again here.
pub(crate) fn each_run<T: Send>(runs: &[Run], check: impl Fn(&Run) -> T + Sync) -> Vec<T> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = runs.iter().map(|run| scope.spawn(|| check(run))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    })
}

/// The tagger's two entry points, as the recipe calls them.
pub(crate) struct Tagging<'a> {
    pub(crate) tag: &'a dyn Fn(&[String]) -> Vec<String>,
    pub(crate) best: &'a dyn Fn(&[String]) -> Vec<NeuralSequence>,
}

/// Compares the tagger's best sequence with `turbo:neural-tagged` and
/// `turbo:neural-sequences`.
pub(crate) fn check_sequences(
    what: &str,
    g: &Golden,
    sentences: &[Vec<String>],
    tagging: Tagging<'_>,
    tolerance: f64,
    totals: &mut Totals,
) {
    let tagged = layer(g, "turbo:neural-tagged");
    let (s, i, t, p) =
        (u32s(tagged, "sentence"), u32s(tagged, "index"), strs(tagged, "tag"), f64s(tagged, "probability"));
    let scores = f64s(layer(g, "turbo:neural-sequences"), "score");
    assert_eq!(scores.len(), sentences.len(), "{what}: sequence scores");
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let sequences = (tagging.best)(sentence);
        assert_eq!(sequences.len(), 1, "{what}: sentence {k}: sequences");
        let sequence = &sequences[0];
        assert_eq!((tagging.tag)(sentence), sequence.outcomes(), "{what}: sentence {k}: tag and topKSequences");
        for index in 0..sentence.len() {
            assert_eq!((s[row], i[row]), (k as u32, index as u32), "{what}: turbo:neural-tagged row {row}");
            assert_eq!(sequence.outcomes()[index], t[row], "{what}: sentence {k} token {index}: tag");
            totals.probabilities.compare(sequence.probs()[index], p[row], tolerance, &|| {
                format!("{what}: sentence {k} token {index}: probability")
            });
            totals.tags += 1;
            row += 1;
        }
        totals
            .sequence_scores
            .compare(sequence.score(), scores[k], tolerance, &|| format!("{what}: sentence {k}: sequence score"));
    }
    assert_eq!(row, tagged.len(), "{what}: turbo:neural-tagged rows past the last token");
    totals.sentences += sentences.len();
}

pub(crate) fn check_bilstm(
    what: &str,
    g: &Golden,
    model: &BilstmPosModel,
    tagger: &BilstmPosTagger,
    totals: &mut Totals,
) {
    let sentences = sentences(g);
    let tags = model.tags().len();
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, tags);
    let crf = model.is_crf();
    let viterbi = crf.then(|| u32s(layer(g, "turbo:crf-viterbi"), "tag").to_vec());
    let marginals = crf.then(|| table(g, "turbo:crf-marginals", "probability", &sentences, tags));
    let mut path_row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let rust = model.score(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        assert!(rust.iter().all(|row| row.len() == tags), "{what}: sentence {k}: scores per token");
        for (index, row) in rust.iter().enumerate() {
            for (tag, &score) in row.iter().enumerate() {
                totals.bilstm_emissions.compare(score, emissions[k][index][tag], BILSTM_TOLERANCE, &|| {
                    format!("{what}: sentence {k} token {index} tag {tag}: emission")
                });
            }
        }
        if let (Some(viterbi), Some(marginals)) = (&viterbi, &marginals) {
            let (transitions, start, end) =
                (model.transition_weights().unwrap(), model.start_weights().unwrap(), model.end_weights().unwrap());
            let path = CrfScorer::viterbi(&rust, transitions, start, end).unwrap();
            for (index, &tag) in path.iter().enumerate() {
                assert_eq!(tag as u32, viterbi[path_row], "{what}: sentence {k} token {index}: Viterbi tag");
                path_row += 1;
            }
            let of_jvm = CrfScorer::marginals(&emissions[k], transitions, start, end).unwrap();
            let of_rust = CrfScorer::marginals(&rust, transitions, start, end).unwrap();
            for index in 0..sentence.len() {
                for tag in 0..tags {
                    let jvm = marginals[k][index][tag];
                    let at = || format!("{what}: sentence {k} token {index} tag {tag}: marginal");
                    totals.marginals_of_jvm_emissions.compare(of_jvm[index][tag], jvm, BILSTM_TOLERANCE, &at);
                    totals.marginals.compare(of_rust[index][tag], jvm, BILSTM_TOLERANCE, &at);
                }
            }
        }
    }
    if let Some(viterbi) = &viterbi {
        assert_eq!(path_row, viterbi.len(), "{what}: Viterbi rows past the last token");
    } else {
        assert!(g.layer("turbo:crf-viterbi").is_none() && g.layer("turbo:crf-marginals").is_none(), "{what}");
    }
    let tagging = Tagging { tag: &|s| tagger.tag(s).unwrap(), best: &|s| tagger.top_k_sequences(s).unwrap() };
    check_sequences(what, g, &sentences, tagging, BILSTM_TOLERANCE, totals);
}

pub(crate) fn check_feedforward(what: &str, g: &Golden, model: &FeedforwardPosModel, totals: &mut Totals) {
    let sentences = sentences(g);
    let tags = model.tags();
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, tags.len());
    let context = layer(g, "turbo:neural-context");
    let (c_sentence, c_index, c_slot) = (u32s(context, "sentence"), u32s(context, "index"), u32s(context, "slot"));
    let (c_present, c_symbol, c_row) = (u32s(context, "present"), strs(context, "symbol"), u32s(context, "row"));
    let vector_rows = g.layer("turbo:neural-vector-rows");
    assert_eq!(vector_rows.is_some(), model.uses_pretrained_vectors(), "{what}: turbo:neural-vector-rows");
    let mut row = 0;
    let mut vector_row = 0;
    for (k, (sentence, jvm_emissions)) in sentences.iter().zip(&emissions).enumerate() {
        let mut assigned: Vec<&str> = Vec::new();
        for (index, jvm_scores) in jvm_emissions.iter().enumerate() {
            let previous = index.checked_sub(1).map(|i| assigned[i]);
            let before_previous = index.checked_sub(2).map(|i| assigned[i]);
            let symbols = FeedforwardPosContext::extract(sentence, index, previous, before_previous);
            let ids = model.feature_ids(&symbols);
            for (slot, (symbol, &id)) in symbols.iter().zip(&ids).enumerate() {
                let at = format!("{what}: sentence {k} token {index} slot {slot}");
                assert_eq!((c_sentence[row], c_index[row], c_slot[row]), (k as u32, index as u32, slot as u32), "{at}");
                // Java has null where the Rust context has the padding symbol.
                let (present, jvm_symbol) = (c_present[row] == 1, c_symbol[row].as_str());
                assert_eq!(if present { jvm_symbol } else { ABSENT }, symbol, "{at}: symbol");
                assert!(present || jvm_symbol.is_empty(), "{at}: an absent symbol is empty");
                assert_eq!(id as u32, c_row[row], "{at}: embedding row");
                totals.contexts += 1;
                row += 1;
            }
            let rows = model.pretrained_rows(sentence, index);
            if let Some(l) = vector_rows {
                let rows = rows.expect("vector rows");
                let (s, i, slot, present, r) =
                    (u32s(l, "sentence"), u32s(l, "index"), u32s(l, "slot"), u32s(l, "present"), u32s(l, "row"));
                for (p, &rust) in rows.iter().enumerate() {
                    let at = format!("{what}: sentence {k} token {index} vector slot {p}");
                    let key = (s[vector_row], i[vector_row], slot[vector_row]);
                    assert_eq!(key, (k as u32, index as u32, p as u32), "{at}");
                    let jvm = if present[vector_row] == 1 { r[vector_row] as isize } else { -1 };
                    assert_eq!(rust, jvm, "{at}");
                    vector_row += 1;
                }
            }
            let scores = model
                .score_with_pretrained(&ids, rows.as_ref().map(|r| &r[..]))
                .unwrap_or_else(|e| panic!("{what}: sentence {k} token {index}: {e}"));
            assert_eq!(scores.len(), jvm_scores.len(), "{what}: sentence {k} token {index}: scores");
            for (tag, (&score, &jvm)) in scores.iter().zip(jvm_scores).enumerate() {
                totals
                    .ff_emissions
                    .compare(score, jvm, 0.0, &|| format!("{what}: sentence {k} token {index} tag {tag}: emission"));
            }
            let mut best = 0;
            for o in 1..scores.len() {
                if scores[o] > scores[best] {
                    best = o;
                }
            }
            assigned.push(&tags[best]);
        }
    }
    assert_eq!(row, context.len(), "{what}: context rows past the last token");
    if let Some(l) = vector_rows {
        assert_eq!(vector_row, l.len(), "{what}: vector rows past the last token");
    }
    let tagger = FeedforwardPosTagger::new(model.clone());
    let tagging = Tagging { tag: &|s| tagger.tag(s).unwrap(), best: &|s| tagger.top_k_sequences(s).unwrap() };
    check_sequences(what, g, &sentences, tagging, 0.0, totals);
}
