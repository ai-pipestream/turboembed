//! Matching over an image: the in-heap matcher's walk, over the double
//! array.

use turbo_analysis::Span;
use turbo_glossary_layers::EntryIx;

use super::{MappedGlossaryMatcher, NO};
use crate::boundaries::Boundaries;
use crate::select::{Candidate, GlossaryMatch, select};

/// The buffers a scan works in, kept by the caller so that matching text
/// after text allocates nothing once they have grown to fit.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    ring: Vec<u32>,
    candidates: Vec<Candidate>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

const ROOT: u32 = 0;

impl MappedGlossaryMatcher<'_> {
    /// The code of a (folded) code point, 0 for one in no pattern.
    #[inline]
    fn code(&self, cp: u32) -> u32 {
        if cp < 128 {
            return self.ascii[cp as usize];
        }
        match self.page_index[(cp >> 8) as usize] {
            0 => 0,
            p => self.pages[((p as usize - 1) << 8) | (cp & 255) as usize],
        }
    }

    /// The cell after `code` from `s`, following failure links on a miss.
    /// A code in no pattern labels no edge, so it leads to the root.
    #[inline]
    fn step(&self, mut s: u32, code: u32) -> u32 {
        if code == 0 {
            return ROOT;
        }
        loop {
            let c = &self.cells[s as usize];
            let t = c.base.wrapping_add(code);
            if self.cells.get(t as usize).is_some_and(|x| x.check == s) {
                return t;
            }
            if s == ROOT {
                return ROOT;
            }
            s = c.fail;
        }
    }

    /// Calls `hit` with every occurrence of every pattern in `text`, in the
    /// order their ends are reached: the same hits, in the same order, as
    /// [`TextMatcher::scan`](crate::TextMatcher::scan) on the matcher the
    /// image was written from.
    pub fn scan(&self, text: &str, scratch: &mut Scratch, mut hit: impl FnMut(Span, EntryIx)) {
        let size = ((self.counts.longest as usize).min(text.len()) + 1).next_power_of_two();
        let mask = size - 1;
        let ring = &mut scratch.ring;
        if ring.len() < size {
            ring.resize(size, 0);
        }
        let case = self.counts.case;
        let mut seen = 0usize;
        let mut state = ROOT;
        for (i, c) in text.char_indices() {
            ring[seen & mask] = i as u32;
            seen += 1;
            state = self.step(state, self.code(case.fold(c)));
            let end = (i + c.len_utf8()) as u32;
            let cell = &self.cells[state as usize];
            let mut s = if cell.entry != NO { state } else { cell.dict };
            while s != NO {
                let x = &self.cells[s as usize];
                let start = ring[(seen - x.depth as usize) & mask];
                hit(Span::new(start, end), EntryIx(x.entry));
                s = x.dict;
            }
        }
    }

    /// Appends to `out` the matches in `text` whose both ends are in
    /// `boundaries`, in text order and without overlaps: the matches of
    /// [`TextMatcher::find`](crate::TextMatcher::find).
    pub fn find(&self, text: &str, boundaries: &Boundaries, scratch: &mut Scratch, out: &mut Vec<GlossaryMatch>) {
        let mut candidates = std::mem::take(&mut scratch.candidates);
        candidates.clear();
        self.scan(text, scratch, |span, entry| {
            if boundaries.around(span) {
                candidates.push(Candidate { span, tokens: 0, entry });
            }
        });
        select(&mut candidates, out);
        scratch.candidates = candidates;
    }
}
