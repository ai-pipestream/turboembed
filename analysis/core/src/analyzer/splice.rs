//! `Analyzer::run_spliced`: the pipeline over an edited text, with the
//! outputs of partition-local steps copied from the base document on the
//! partitions the edit did not touch and computed again only on the
//! windows it did. Document-wide steps run as they do in `run`.

#[path = "splice/plan.rs"]
mod plan;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::annotate::Locality;
use crate::bytes::SourceError;
use crate::doc::{Document, Status, View, check_view_alignment};
use crate::error::Error;
use crate::layer::{Key, KeyId, Provenance};
use crate::span::Span;
use crate::splice::{LayerSplicer, Shift, SpliceCx};
use crate::text::Alignment;

use super::Analyzer;
use plan::Plan;

pub(super) type Splicers = HashMap<TypeId, Box<dyn LayerSplicer>>;

/// How one step ran in [`Analyzer::run_spliced`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepRun {
    /// A document-wide step: it ran over the whole new document.
    Document,
    /// A partition-local step: its outputs were copied on the clean
    /// partitions and computed on this many windows.
    Spliced { windows: usize },
    /// A partition-local step that had to run over the whole new
    /// document anyway, and why: its declaration was not borne out by
    /// the pipeline, an output has no splicer, the base document lacks
    /// the output, an item or an index left its partition, or a window
    /// run failed.
    Rerun(String),
}

/// What [`Analyzer::run_spliced`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpliceReport {
    /// Per partition layer, the windows of the new text its local steps
    /// ran on, in text order.
    pub windows: Vec<(KeyId, Vec<Span>)>,
    /// Per step, in run order, how it ran.
    pub steps: Vec<(&'static str, StepRun)>,
}

impl SpliceReport {
    pub fn step(&self, id: &str) -> Option<&StepRun> {
        self.steps.iter().find(|(s, _)| *s == id).map(|(_, r)| r)
    }
}

impl Analyzer {
    /// Runs the pipeline over `new_text`, an edited copy of `base`'s
    /// source text, reusing `base`'s layers where it can. `alignment`
    /// maps `base`'s source text (parent) to `new_text` (child); several
    /// edits are one alignment, composed with `Alignment::compose` when
    /// they were made one after another. `base` is not changed.
    ///
    /// Every step with an effective locality of `Partition(P)` has its
    /// outputs copied, item by item with spans and indexes moved, from
    /// the partitions of `base` that no edit touched and that map to
    /// exactly one partition of the new text, and computed again only on
    /// windows of the dirty partitions. Everything else runs as in
    /// [`Analyzer::run`]. The result is what `run` would make on
    /// `new_text`, layer for layer, with the same provenance, and the
    /// report says what was copied and what ran. A step's failure on the
    /// new document is the error.
    pub fn run_spliced(
        &self,
        base: &Document,
        new_text: impl Into<Arc<str>>,
        alignment: &Alignment,
    ) -> Result<(Document, SpliceReport), Error> {
        let new_text: Arc<str> = new_text.into();
        let source = Key::<View>::DEFAULT.id();
        let base_text = base.require(&Key::<View>::DEFAULT)?.text();
        check_view_alignment(base_text, &new_text, alignment)
            .map_err(|detail| Error::Source(SourceError::Misaligned(detail)))?;
        if self.given.iter().any(|&g| g != source) {
            return Err(Error::Pipeline("run_spliced needs a pipeline given only the source text".into()));
        }
        let splicers: Splicers = self
            .steps
            .iter()
            .flat_map(|s| s.writes().into_iter().zip(s.splicers()))
            .filter_map(|(w, sp)| sp.map(|sp| (w.ty, sp)))
            .collect();
        let mut doc = Document::new(new_text);
        let mut status = [Ok(())];
        let mut report = SpliceReport::default();
        let mut plans: HashMap<KeyId, Plan> = HashMap::new();
        for stage in &self.stages {
            let whole: Vec<usize> = stage.iter().copied().filter(|&i| !self.is_local(i)).collect();
            let backedges: Vec<KeyId> = whole.iter().flat_map(|&i| self.backedge_reads[i].iter().copied()).collect();
            let taken = doc.take_entries(&backedges);
            let mut outputs = self.compute_steps(&whole, std::slice::from_ref(&doc)).into_iter();
            for &i in stage {
                let id = self.steps[i].id();
                if self.is_local(i) {
                    let run = match self.splice_step(i, base, &mut doc, alignment, &mut plans, &splicers) {
                        Ok(windows) => StepRun::Spliced { windows },
                        Err(reason) => {
                            self.run_steps(&[i], std::slice::from_mut(&mut doc), &mut status);
                            StepRun::Rerun(reason)
                        }
                    };
                    report.steps.push((id, run));
                    continue;
                }
                let out = outputs.next().expect("an output per whole step");
                self.steps[i].store(out, std::slice::from_mut(&mut doc), &mut status);
                for w in self.steps[i].writes() {
                    doc.mark_pipeline_stored(w);
                }
                let run = match self.locality_reason(i) {
                    Some(reason) => StepRun::Rerun(reason),
                    None => StepRun::Document,
                };
                report.steps.push((id, run));
            }
            doc.restore_entries(taken);
        }
        let [status] = status;
        status?;
        for (p, plan) in plans {
            report.windows.push((p, plan.windows.iter().map(|w| w.span).collect()));
        }
        report.windows.sort_by_key(|(_, w)| w.first().map_or(0, |s| s.start));
        Ok((doc, report))
    }

    /// Whether step `i` is spliced: partition-local by the pipeline and
    /// not on an optional cycle.
    fn is_local(&self, i: usize) -> bool {
        matches!(self.localities[i], Locality::Partition(_)) && self.backedge_reads[i].is_empty()
    }

    /// Why step `i`, declared partition-local, runs whole.
    fn locality_reason(&self, i: usize) -> Option<String> {
        if !self.backedge_reads[i].is_empty() && matches!(self.localities[i], Locality::Partition(_)) {
            return Some("it reads a later step's output on an optional cycle".into());
        }
        super::locality::downgrade_reason(i, &self.given, &self.steps, &self.localities)
    }

    /// Splices step `i`'s outputs into `doc`: `Ok` with the number of
    /// windows it ran on, or `Err` with why it has to run whole instead,
    /// with nothing added to `doc`.
    fn splice_step(
        &self,
        i: usize,
        base: &Document,
        doc: &mut Document,
        alignment: &Alignment,
        plans: &mut HashMap<KeyId, Plan>,
        splicers: &Splicers,
    ) -> Result<usize, String> {
        let step = &self.steps[i];
        let Locality::Partition(p) = self.localities[i] else { return Err("not partition-local".into()) };
        let writes = step.writes();
        let mut sps = step.splicers();
        if sps.len() != writes.len() {
            return Err("the step cannot say how its outputs are spliced".into());
        }
        for (w, sp) in writes.iter().zip(&sps) {
            if sp.is_none() {
                return Err(format!("{w} has no splicer"));
            }
            if !base.present(*w) || base.status(*w) != Some(&Status::Complete) {
                return Err(format!("{w} is not complete on the base document"));
            }
        }
        let plan = match plans.entry(p) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut plan = Plan::build(p, base, doc, alignment, splicers)?;
                self.seed_windows(p, &mut plan, doc, splicers)?;
                e.insert(plan)
            }
        };
        for w in &mut plan.windows {
            let mut st = [Ok(())];
            self.run_steps(&[i], std::slice::from_mut(&mut w.doc), &mut st);
            let [st] = st;
            st.map_err(|e| format!("the window at {}..{} failed: {e}", w.span.start, w.span.end))?;
            if let Some(k) = writes.iter().find(|&&k| !w.doc.present(k)) {
                return Err(format!("{k} was skipped on the window at {}..{}", w.span.start, w.span.end));
            }
        }
        let provenance = step.expected_provenance(doc);
        let mut added: Vec<KeyId> = Vec::with_capacity(writes.len());
        for (k, w) in writes.iter().enumerate() {
            let sp = sps[k].take().expect("checked above");
            let result = assemble(&*sp, *w, base, doc, plan, splicers)
                .and_then(|out| sp.add(out, doc, *w, &added, &provenance).map_err(|e| e.to_string()));
            if let Err(reason) = result {
                for &a in &added {
                    doc.remove(a);
                }
                return Err(reason);
            }
            doc.mark_pipeline_stored(*w);
            added.push(*w);
        }
        Ok(plan.windows.len())
    }

    /// Seeds each window's document with the slices of the partition
    /// layer and of every other spliceable output of its producer, so the
    /// local steps find their inputs there.
    fn seed_windows(&self, p: KeyId, plan: &mut Plan, doc: &Document, splicers: &Splicers) -> Result<(), String> {
        let producer =
            self.steps.iter().find(|s| s.writes().contains(&p)).ok_or_else(|| format!("nothing writes {p}"))?;
        let provenance = Provenance::by("turbo:splice");
        let mut seeded: Vec<KeyId> = Vec::new();
        for o in producer.writes() {
            let Some(sp) = splicers.get(&o.ty) else { continue };
            let Some(value) = doc.erased(o) else { continue };
            let on: Vec<KeyId> = doc.made_on(o).to_vec();
            let ranges = plan.new_ranges(o, doc, splicers).ok_or_else(|| format!("{o} does not follow {p}"))?.to_vec();
            for wi in 0..plan.windows.len() {
                let (parts, span) = (plan.windows[wi].parts.clone(), plan.windows[wi].span);
                let items = ranges[parts.start].start..ranges[parts.end - 1].end;
                let mut shifts = vec![Shift { key: o, from: items.clone(), to: 0 }];
                for &k in &on {
                    let r = plan.new_ranges(k, doc, splicers).ok_or_else(|| format!("{k} does not follow {p}"))?;
                    shifts.push(Shift { key: k, from: r[parts.start].start..r[parts.end - 1].end, to: 0 });
                }
                let mut out = sp.start(value, crate::text::ViewId::SOURCE).ok_or("another type")?;
                let mut map = |s: Span| Span::new(s.start - span.start, s.end - span.start);
                let mut cx = SpliceCx::new(&mut map, &shifts, o);
                sp.append(&mut *out, value, items.clone(), &mut cx)?;
                sp.add(out, &mut plan.windows[wi].doc, o, &seeded, &provenance).map_err(|e| e.to_string())?;
            }
            seeded.push(o);
        }
        Ok(())
    }
}

/// Builds output `w` of a local step on the new document: the base's
/// items on the clean partitions, moved by each partition's offset, and
/// the windows' items where the partitions are dirty, in text order.
fn assemble(
    sp: &dyn LayerSplicer,
    w: KeyId,
    base: &Document,
    doc: &Document,
    plan: &mut Plan,
    splicers: &Splicers,
) -> Result<Box<dyn Any + Send + Sync>, String> {
    let base_value = base.erased(w).ok_or_else(|| format!("{w} is not on the base document"))?;
    let base_on: Vec<KeyId> = base.made_on(w).to_vec();
    let base_items =
        plan.base_ranges(w, base, splicers).ok_or_else(|| format!("{w} does not follow its partitions"))?.to_vec();
    let mut base_shifts: Vec<(KeyId, Vec<std::ops::Range<usize>>)> = Vec::with_capacity(base_on.len());
    for &k in &base_on {
        let r = plan.base_ranges(k, base, splicers).ok_or_else(|| format!("{k} does not follow its partitions"))?;
        base_shifts.push((k, r.to_vec()));
    }
    let mut new_starts: Vec<(KeyId, Vec<usize>)> = Vec::with_capacity(base_on.len());
    for &k in &base_on {
        let r = plan.new_ranges(k, doc, splicers).ok_or_else(|| format!("{k} does not follow its partitions"))?;
        new_starts.push((k, r.iter().map(|r| r.start).collect()));
    }
    let mut out = sp.start(base_value, crate::text::ViewId::SOURCE).ok_or("another type")?;
    let mut appended = 0usize;
    for j in 0..plan.new.len() {
        if let Some(i) = plan.clean[j] {
            let items = base_items[i].clone();
            let mut shifts = vec![Shift { key: w, from: items.clone(), to: appended }];
            for ((k, from), (_, to)) in base_shifts.iter().zip(&new_starts) {
                shifts.push(Shift { key: *k, from: from[i].clone(), to: to[j] });
            }
            let delta = plan.new[j].start as i64 - plan.base[i].start as i64;
            let mut map = |s: Span| Span::new((s.start as i64 + delta) as u32, (s.end as i64 + delta) as u32);
            let mut cx = SpliceCx::new(&mut map, &shifts, w);
            sp.append(&mut *out, base_value, items.clone(), &mut cx)?;
            appended += items.len();
            continue;
        }
        let wi = plan.window_of[j].expect("a dirty partition is in a window");
        let win = &plan.windows[wi];
        if win.parts.start != j {
            continue;
        }
        let value = win.doc.erased(w).ok_or_else(|| format!("{w} is not on the window"))?;
        let len = win.doc.layer_len(w);
        let mut shifts = vec![Shift { key: w, from: 0..len, to: appended }];
        for &k in win.doc.made_on(w) {
            let to = new_starts.iter().find(|(key, _)| *key == k).map(|(_, s)| s[j]);
            let to = to.ok_or_else(|| format!("{w} was made on {k} in the window but not on the base"))?;
            shifts.push(Shift { key: k, from: 0..win.doc.layer_len(k), to });
        }
        let off = win.span.start;
        let mut map = |s: Span| Span::new(s.start + off, s.end + off);
        let mut cx = SpliceCx::new(&mut map, &shifts, w);
        sp.append(&mut *out, value, 0..len, &mut cx)?;
        appended += len;
    }
    Ok(out)
}
