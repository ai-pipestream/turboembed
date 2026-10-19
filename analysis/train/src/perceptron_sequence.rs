//! `SimplePerceptronSequenceTrainer` behind `Algorithm=PERCEPTRON_SEQUENCE`.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::event::Event;
use crate::gis::TrainingReport;
use crate::index::{DataIndex, IndexParams, IndexerKind};
use crate::java_string::JavaString;
use crate::params::{DEFAULT_CUTOFF, DEFAULT_ITERATIONS, Key, TrainerType, TrainingParameters};
use crate::perceptron::{Observer, PerceptronIteration, PerceptronModel};

/// `Sequence<T>` without its source: the events of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub events: Vec<Event>,
}

/// `SequenceStream<Event>`: the sequences, read again every iteration, and
/// `updateContext`, which tags a sequence's events with a model and
/// returns them with the model's outcomes, one per event in order.
pub trait SequenceSource {
    fn sequences(&self) -> &[Sequence];

    /// `updateContext(sequence, model)`: the events as the model tags
    /// them. Each must carry an outcome of the model's.
    fn update_context(&self, sequence: &Sequence, model: &PerceptronModel) -> Vec<Event>;
}

/// `SimplePerceptronSequenceTrainer`'s record of a parameter's last change,
/// for the lazy average: `updates[pi][oi]`.
#[derive(Debug, Clone, Copy, Default)]
struct LastChange {
    /// The parameter after the change, truncated to an `int`.
    value: i32,
    /// The 0-based iteration of the change.
    iteration: i32,
    /// The sequence of the change.
    sequence: i32,
}

/// `TrainerFactory.getEventModelSequenceTrainer(...).train(stream)` for
/// `PERCEPTRON_SEQUENCE`: `Iterations`, `Cutoff` and `UseAverage` from the
/// parameters, the events indexed by `OnePassDataIndexer` unsorted, then
/// the sequence perceptron. There is no outcome check here: a stream with
/// one outcome trains.
pub fn train_sequences(
    source: &dyn SequenceSource,
    params: &TrainingParameters,
) -> Result<(PerceptronModel, TrainingReport)> {
    let iterations = params.get_int(Key::ITERATIONS)?.unwrap_or(DEFAULT_ITERATIONS);
    let cutoff = params.get_int(Key::CUTOFF)?.unwrap_or(DEFAULT_CUTOFF);
    let use_average = params.get_bool(Key::USE_AVERAGE)?.unwrap_or(true);
    let model = train_sequence_model(source, iterations, cutoff, use_average, None)?;
    Ok((model, TrainingReport { event_hash: None, trainer_type: TrainerType::EventModelSequence }))
}

/// `SimplePerceptronSequenceTrainer.trainModel(iterations, stream, cutoff,
/// useAverage)`. After every iteration an observer, when given, hears the
/// iteration (with the events tagged right over the events, as the trainer
/// logs them) and the parameters a run stopping there would return. A
/// source whose `update_context` returns another number of events than the
/// sequence has, or an outcome the index does not know, is refused
/// ([`Error::SequenceLength`], [`Error::UnknownOutcome`]).
pub fn train_sequence_model(
    source: &dyn SequenceSource,
    iterations: i32,
    cutoff: i32,
    use_average: bool,
    mut observe: Option<Observer<'_>>,
) -> Result<PerceptronModel> {
    let sequences = source.sequences();
    let events: Vec<Event> = sequences.iter().flat_map(|s| s.events.iter().cloned()).collect();
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, cutoff, false))?;
    // trainingStats, after the last iteration, reads outcomeList[oei] for
    // every event of the stream; the list has one entry per event kept.
    if (index.num_events() as usize) < events.len() {
        return Err(Error::SequenceEventsDropped { indexed: index.num_events(), streamed: events.len() });
    }
    let num_sequences = sequences.len() as i32;
    let total = index.num_events() as i32;
    let mut model = PerceptronModel::zeros(index.outcome_labels().to_vec(), index.pred_labels().to_vec());
    let num_outcomes = model.num_outcomes();
    let omap: HashMap<JavaString, usize> =
        index.outcome_labels().iter().enumerate().map(|(i, o)| (o.clone(), i)).collect();
    let mut average = vec![0.0; model.parameters.len()];
    let mut changes = vec![LastChange::default(); model.parameters.len()];
    // `featureCounts`: one map per outcome, cleared for each sequence that
    // updates. Keyed by predicate id, since a feature the index does not
    // know never reaches the parameters.
    let mut counts: Vec<HashMap<u32, f32>> = (0..num_outcomes).map(|_| HashMap::new()).collect();
    let mut mean = vec![0.0; if observe.is_some() && use_average { model.parameters.len() } else { 0 }];

    for iteration in 0..iterations {
        let mut correct = 0;
        for (si, sequence) in (0i32..).zip(sequences) {
            let tagged = source.update_context(sequence, &model);
            if tagged.len() != sequence.events.len() {
                return Err(Error::SequenceLength { expected: sequence.events.len(), got: tagged.len() });
            }
            let mut update = false;
            for (t, e) in tagged.iter().zip(&sequence.events) {
                if t.outcome() != e.outcome() {
                    update = true;
                } else {
                    correct += 1;
                }
            }
            if !update {
                continue;
            }
            for c in counts.iter_mut() {
                c.clear();
            }
            for e in &sequence.events {
                // The sequence's own outcomes are the index's.
                let oi = omap[e.outcome()];
                for (ci, feature) in e.context().iter().enumerate() {
                    let Some(pi) = model.predicate_index(feature) else { continue };
                    let value = e.values().map_or(1.0f32, |v| v[ci]);
                    counts[oi].entry(pi).and_modify(|c| *c += value).or_insert(value);
                }
            }
            for t in &tagged {
                let Some(&oi) = omap.get(t.outcome()) else {
                    return Err(Error::UnknownOutcome(t.outcome().to_string()));
                };
                for (ci, feature) in t.context().iter().enumerate() {
                    let Some(pi) = model.predicate_index(feature) else { continue };
                    let value = t.values().map_or(1.0f32, |v| v[ci]);
                    let c = match counts[oi].get(&pi) {
                        None => -value,
                        Some(&c) => c - value,
                    };
                    if c == 0.0 {
                        counts[oi].remove(&pi);
                    } else {
                        counts[oi].insert(pi, c);
                    }
                }
            }
            for (oi, c) in counts.iter().enumerate() {
                for (&pi, &count) in c {
                    *model.cell(pi, oi) += f64::from(count);
                    if use_average {
                        let at = pi as usize * num_outcomes + oi;
                        let last = changes[at];
                        if last.value != 0 {
                            let held = num_sequences
                                .wrapping_mul(iteration.wrapping_sub(last.iteration))
                                .wrapping_add(si.wrapping_sub(last.sequence));
                            average[at] += f64::from(last.value.wrapping_mul(held));
                        }
                        changes[at] = LastChange { value: *model.cell(pi, oi) as i32, iteration, sequence: si };
                    }
                }
            }
        }
        let accuracy = f64::from(correct) / f64::from(total);
        if let Some(observe) = observe.as_deref_mut() {
            let output: &[f64] = if use_average {
                finish_average(&average, &changes, num_sequences, iteration + 1, &mut mean);
                &mean
            } else {
                &model.parameters
            };
            observe(&PerceptronIteration { iteration: iteration + 1, correct, total, accuracy }, output);
        }
    }
    if use_average {
        let mut out = vec![0.0; model.parameters.len()];
        finish_average(&average, &changes, num_sequences, iterations, &mut out);
        model.parameters = out;
    }
    Ok(model)
}

/// The end of `nextIteration` on the last iteration, into `out`: each
/// parameter's pending weight added and the sum divided by the iterations
/// times the sequences. `ran` is the iterations run so far, which the
/// trainer only ever reaches with `ran == iterations`; the observer sees
/// the mean a run of `ran` iterations would end with.
fn finish_average(average: &[f64], changes: &[LastChange], num_sequences: i32, ran: i32, out: &mut [f64]) {
    let total = f64::from(ran) * f64::from(num_sequences);
    out.copy_from_slice(average);
    for (at, p) in out.iter_mut().enumerate() {
        let last = changes[at];
        if last.value != 0 {
            let held = num_sequences.wrapping_mul(ran.wrapping_sub(last.iteration)).wrapping_sub(last.sequence);
            *p += f64::from(last.value.wrapping_mul(held));
        }
        if *p != 0.0 {
            *p /= total;
        }
    }
}
