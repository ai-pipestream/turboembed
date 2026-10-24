//! Boilerplate detection: blocks repeated across a batch of documents
//! (headers, footers, navigation, disclaimers) and spans repeated inside
//! one document (templates).

use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Fnv1a, Ix, Layer, Refs};
use turbo_analysis_layers::{Block, Boilerplate, BoilerplateKind, Chunks, Sentences, Structure};

use crate::sketch::Sketches;

/// A level whose items can be boilerplate. `hint` is what the layer
/// itself says an item is, before any repetition is counted: a structure
/// block marked as a page header, or inside a list.
pub trait BoilerplateUnit: Layer {
    fn hint(&self, _i: Ix<Self>) -> Option<BoilerplateKind> {
        None
    }
}

impl BoilerplateUnit for Sentences {}
impl BoilerplateUnit for Chunks {}

impl BoilerplateUnit for Structure {
    /// The kind the block or its nearest marked ancestor says: a page
    /// header is a header, a page footer a footer, and a list or list
    /// item navigation.
    fn hint(&self, i: Ix<Structure>) -> Option<BoilerplateKind> {
        let mut at = Some(i);
        while let Some(b) = at {
            match self.kind[b.get()] {
                Block::PageHeader => return Some(BoilerplateKind::Header),
                Block::PageFooter => return Some(BoilerplateKind::Footer),
                Block::List | Block::ListItem => return Some(BoilerplateKind::Navigation),
                _ => at = self.parent[b.get()],
            }
        }
        None
    }
}

/// Reads the items of `L` and their sketches; writes which are
/// boilerplate (`Boilerplate<L>`).
///
/// Across a batch (`Analyzer::run` hands the step every document of the
/// batch at once), an item is boilerplate when its shape (its words with
/// numbers folded, so page numbers and dates do not tell copies apart)
/// is held by at least `min_docs` documents and by at least `min_share`
/// of the batch. Its score is the share of the batch's documents that
/// hold it. Its kind is the layer's hint when there is one; otherwise a
/// disclaimer when it holds at least `disclaimer_words` words and is in
/// the second half of its document's items by index, else a header in the
/// first half and a footer in the second.
///
/// Inside one document, a shape held by at least two items that do not
/// overlap is a template, score `1 - 1/n` for `n` such copies, unless the
/// batch already made it boilerplate of another kind. A document alone is
/// a batch of one, so only templates are found in it. Items with fewer
/// than `min_words` words are never boilerplate.
pub struct BoilerplateDetector<L> {
    pub min_docs: u32,
    pub min_share: f64,
    pub min_words: u32,
    pub disclaimer_words: u32,
    _of: PhantomData<fn() -> L>,
}

impl<L> Default for BoilerplateDetector<L> {
    /// Two documents and half the batch; two words; disclaimers of 25.
    fn default() -> BoilerplateDetector<L> {
        BoilerplateDetector { min_docs: 2, min_share: 0.5, min_words: 2, disclaimer_words: 25, _of: PhantomData }
    }
}

impl<L> BoilerplateDetector<L> {
    pub fn new(min_docs: u32, min_share: f64, min_words: u32, disclaimer_words: u32) -> BoilerplateDetector<L> {
        BoilerplateDetector { min_docs, min_share, min_words, disclaimer_words, _of: PhantomData }
    }
}

/// One document's items as the detector reads them.
struct Doc<'d, L> {
    units: &'d L,
    start: &'d [u32],
    end: &'d [u32],
    sketches: &'d Sketches<L>,
}

impl<L: BoilerplateUnit> BoilerplateDetector<L> {
    fn doc<'d>(&self, units: &'d L, sketches: &'d Sketches<L>) -> Result<Doc<'d, L>, String> {
        let ts = units.text_spans().ok_or_else(|| format!("{} has no spans", L::NAME))?;
        if !ts.is_start_ordered() {
            return Err(format!("the items of {} are not in start order", L::NAME));
        }
        if sketches.words.len() != ts.start.len() {
            return Err(format!("{} sketches for {} items", sketches.words.len(), ts.start.len()));
        }
        sketches.check()?;
        Ok(Doc { units, start: ts.start, end: ts.end, sketches })
    }

    fn eligible(&self, d: &Doc<'_, L>, i: usize) -> bool {
        d.sketches.words[i] >= self.min_words.max(1)
    }

    /// The flags of one document, given how many documents of a batch
    /// of `n` hold each shape (`held`, sorted by shape).
    fn flag(&self, d: &Doc<'_, L>, held: &[(u64, u32)], n: u32) -> Boilerplate<L> {
        let len = d.start.len();
        let (mut kind, mut score) = (vec![None; len], vec![0f32; len]);
        let need = f64::from(self.min_docs.max(1)).max((self.min_share * f64::from(n)).ceil());
        for i in (0..len).filter(|&i| self.eligible(d, i)) {
            let count = held.binary_search_by_key(&d.sketches.shape[i], |h| h.0).map_or(0, |k| held[k].1);
            if n < self.min_docs || f64::from(count) < need {
                continue;
            }
            let second_half = 2 * i >= len;
            kind[i] = Some(d.units.hint(Ix::new(i as u32)).unwrap_or(
                if second_half && d.sketches.words[i] >= self.disclaimer_words {
                    BoilerplateKind::Disclaimer
                } else if second_half {
                    BoilerplateKind::Footer
                } else {
                    BoilerplateKind::Header
                },
            ));
            score[i] = (f64::from(count) / f64::from(n)) as f32;
        }
        // Templates: shapes held by two items that do not overlap.
        let mut order: Vec<usize> = (0..len).filter(|&i| self.eligible(d, i)).collect();
        order.sort_unstable_by_key(|&i| (d.sketches.shape[i], i));
        for run in order.chunk_by(|&a, &b| d.sketches.shape[a] == d.sketches.shape[b]) {
            let (mut copies, mut end) = (0u32, 0u32);
            for &i in run {
                if copies == 0 || d.start[i] >= end {
                    copies += 1;
                    end = d.end[i];
                }
            }
            if copies < 2 {
                continue;
            }
            for &i in run {
                if kind[i].is_none() {
                    kind[i] = Some(BoilerplateKind::Template);
                    score[i] = 1.0 - 1.0 / copies as f32;
                }
            }
        }
        Boilerplate::new(kind, score)
    }

    fn check(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.min_share) {
            return Err(format!("a minimum share of {} outside [0, 1]", self.min_share));
        }
        Ok(())
    }
}

impl<L: BoilerplateUnit> Annotator for BoilerplateDetector<L> {
    const ID: &'static str = "turbo:boilerplate-detector";
    type Reads = (L, Sketches<L>);
    type Writes = (Boilerplate<L>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Boilerplate<L>,), Error> {
        self.annotate_batch(vec![input]).pop().expect("one result for one document")
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<(Boilerplate<L>,), Error>> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if let Err(e) = self.check() {
            return inputs.iter().map(|_| Err(fail(e.clone()))).collect();
        }
        let docs: Vec<Result<Doc<'d, L>, String>> = inputs.iter().map(|(u, s)| self.doc(u, s)).collect();
        // How many documents hold each shape: each document's distinct
        // shapes, counted.
        let mut shapes: Vec<u64> = Vec::new();
        let mut one: Vec<u64> = Vec::new();
        let mut n = 0u32;
        for d in docs.iter().flatten() {
            one.clear();
            one.extend((0..d.start.len()).filter(|&i| self.eligible(d, i)).map(|i| d.sketches.shape[i]));
            one.sort_unstable();
            one.dedup();
            shapes.extend_from_slice(&one);
            n += 1;
        }
        shapes.sort_unstable();
        let held: Vec<(u64, u32)> = shapes.chunk_by(|a, b| a == b).map(|r| (r[0], r.len() as u32)).collect();
        docs.into_iter().map(|d| d.map(|d| (self.flag(&d, &held, n),)).map_err(fail)).collect()
    }

    fn config_hash(&self) -> u64 {
        [
            u64::from(self.min_docs),
            self.min_share.to_bits(),
            u64::from(self.min_words),
            u64::from(self.disclaimer_words),
        ]
        .iter()
        .fold(Fnv1a::new(), |mut h, &x| {
            h.u64(x);
            h
        })
        .finish()
    }
}
