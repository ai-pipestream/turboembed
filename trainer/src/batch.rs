//! A minibatch's gradients across threads, with bytes that do not depend
//! on how many: each example's gradients go to a buffer of its own, and
//! the buffers are summed element by element in example order, so one
//! thread and sixteen give the same sum. A batch runs in rounds of
//! `lanes` examples, each round's buffers added to the sum in turn, so
//! the sum is the same whatever the rounds and memory is bounded by the
//! lanes, not the batch (gradient accumulation). The buffers are kept
//! from step to step.

use std::thread;

use turbo::status::Result;

use crate::fit::Model;

/// One example's gradients, its working memory, and its loss and words.
pub(crate) struct Slot<M: Model> {
    grads: Vec<Vec<f32>>,
    work: M::Work,
    out: Option<Result<(f64, usize)>>,
}

/// The buffers a run reuses, and the threads it spreads a batch over.
pub(crate) struct Batch<M: Model> {
    slots: Vec<Slot<M>>,
    threads: usize,
    lanes: usize,
}

impl<M: Model> Batch<M> {
    /// For `threads` threads (0 takes every core the machine offers) and
    /// `lanes` examples at once (0 takes the whole batch).
    pub(crate) fn new(threads: usize, lanes: usize) -> Batch<M> {
        Batch { slots: Vec::new(), threads: resolve(threads), lanes }
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    /// The summed gradients of `examples` written to `grads` (shaped as
    /// the model's weights), and their summed loss and words; the first
    /// refused example's error, in example order, with the sum left
    /// unfinished.
    pub(crate) fn gradients(
        &mut self,
        model: &M,
        examples: &[&M::Example],
        grads: &mut [Vec<f32>],
    ) -> Result<(f64, usize)> {
        let lanes = if self.lanes == 0 { examples.len() } else { self.lanes }.max(1);
        while self.slots.len() < lanes.min(examples.len()) {
            self.slots.push(Slot { grads: model.zero_grads(), work: M::Work::default(), out: None });
        }
        let (mut loss, mut words) = (0.0, 0);
        let mut first = true;
        for round in examples.chunks(lanes) {
            let k = round.len();
            let slots = &mut self.slots[..k];
            let per = k.div_ceil(self.threads.min(k).max(1));
            let run = |slots: &mut [Slot<M>], examples: &[&M::Example]| {
                for (s, e) in slots.iter_mut().zip(examples) {
                    s.grads.iter_mut().for_each(|g| g.fill(0.0));
                    s.out = Some(model.loss(e, &mut s.grads, &mut s.work));
                }
            };
            if per >= k {
                run(slots, round);
            } else {
                thread::scope(|scope| {
                    for (s, e) in slots.chunks_mut(per).zip(round.chunks(per)) {
                        scope.spawn(|| run(s, e));
                    }
                });
            }
            for s in slots.iter_mut() {
                let (l, n) = s.out.take().expect("every slot ran")?;
                loss += l;
                words += n;
            }
            let parts: Vec<&[Vec<f32>]> = slots.iter().map(|s| &s.grads[..]).collect();
            reduce(&parts, grads, self.threads, first);
            first = false;
        }
        if first {
            grads.iter_mut().for_each(|g| g.fill(0.0));
        }
        Ok((loss, words))
    }
}

/// `threads`, or the machine's cores when 0.
fn resolve(threads: usize) -> usize {
    match threads {
        0 => thread::available_parallelism().map_or(1, usize::from),
        n => n,
    }
}

/// grads = `parts` summed in order, each element on its own, onto zero
/// (`first`) or onto what grads holds, with the tensors split across
/// `threads` threads by element.
fn reduce(parts: &[&[Vec<f32>]], grads: &mut [Vec<f32>], threads: usize, first: bool) {
    let sum = |i: usize, start: usize, out: &mut [f32]| {
        if first {
            out.fill(0.0);
        }
        for p in parts {
            for (a, &b) in out.iter_mut().zip(&p[i][start..]) {
                *a += b;
            }
        }
    };
    let total: usize = grads.iter().map(Vec::len).sum();
    if threads <= 1 || parts.len() <= 1 || total < 1 << 14 {
        for (i, g) in grads.iter_mut().enumerate() {
            sum(i, 0, g);
        }
        return;
    }
    let per = total.div_ceil(threads);
    let mut work: Vec<Vec<(usize, usize, &mut [f32])>> = (0..threads).map(|_| Vec::new()).collect();
    let (mut t, mut room) = (0, per);
    for (i, g) in grads.iter_mut().enumerate() {
        let mut rest: &mut [f32] = g;
        let mut start = 0;
        while !rest.is_empty() {
            let take = room.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            work[t].push((i, start, head));
            (rest, start, room) = (tail, start + take, room - take);
            if room == 0 {
                (t, room) = ((t + 1).min(threads - 1), per);
            }
        }
    }
    thread::scope(|scope| {
        for pieces in work {
            scope.spawn(move || pieces.into_iter().for_each(|(i, start, out)| sum(i, start, out)));
        }
    });
}
