//! Place names found in running text with a gazetteer's match keys as one
//! mapped automaton, for candidate mentions before geocoding or linking.
//!
//! [`name_image`] writes the distinct match keys of a gazetteer's index as
//! a `turbo-glossary` automaton image ([`MappedGlossaryWriter`], the keys
//! as terms, case sensitive, since keys are folded already).
//! [`NameFinder`] reads the image in place and finds, in a text, every run
//! of word tokens whose match key is one of the image's keys: the text's
//! UAX #29 word tokens are folded one by one as [`Folder`] folds a name,
//! joined by one space, and the automaton matches over that at token
//! edges. A match is a byte span of the text and the key it folds to, and
//! [`Gazetteer::lookup`](crate::Gazetteer::lookup) of the span's text
//! gives the same records as a lookup of the key.
//!
//! The image adds to the gazetteer rather than replacing it: a match names
//! a key, and the gazetteer gives its records ([`lookup_key`]). Nothing in
//! the image ties it to one gazetteer; one written from another table, or
//! from an older version of this one, misses the names it lacks and
//! reports keys the gazetteer does not hold, so write it again when the
//! table changes.
//!
//! A word longer than 255 UTF-16 units is cut into pieces, as the match
//! key's tokenizer cuts it, and a name that starts partway into such a
//! word is not found, though a lookup of its text alone would find it: the
//! finder sees the piece as the text's tokenizer made it. Names next to
//! ordinary words are not affected.
//!
//! [`lookup_key`]: crate::InMemoryGazetteer::lookup_key
//!
//! The image needs no heap structure per state and loads in place from a
//! byte slice or a mapped file, so a catalog of millions of names starts
//! without rebuilding its automaton; [`NameAnnotator`] writes the matches
//! as location entities for [`GeocodeAnnotator`](crate::GeocodeAnnotator)
//! and [`GeoEntityLinker`](crate::GeoEntityLinker).

mod annotate;

pub use annotate::NameAnnotator;

use std::fmt;

use turbo_analysis::Span;
use turbo_glossary::{
    BuildError, Case, GlossaryMatch, ImageError, MappedGlossaryMatcher, MappedGlossaryWriter, Scratch,
};
use turbo_glossary_layers::{EntryIx, Glossary};

use crate::fold::Folder;
use crate::index::GazetteerIndex;

/// The image of `index`'s match keys, key `k` as entry `k`.
pub(crate) fn name_image(index: &GazetteerIndex) -> Result<Vec<u8>, BuildError> {
    let names = &index.names;
    let mut glossary = Glossary::new();
    let mut id = String::new();
    for k in 0..names.len() as u32 {
        id.clear();
        id.push_str(itoa(k, &mut [0; 10]));
        glossary.push(&id, names.key(k)).map_err(|_| BuildError::BlankTerm { entry: EntryIx(k) })?;
    }
    MappedGlossaryWriter::write(&glossary, Case::Sensitive)
}

fn itoa(mut n: u32, buf: &mut [u8; 10]) -> &str {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    std::str::from_utf8(&buf[i..]).unwrap()
}

/// One place name found in a text: its bytes and the key it folds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NameMatch {
    pub span: Span,
    pub key: EntryIx,
}

/// Why a name image was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameImageError {
    /// The bytes are not a valid automaton image.
    Image(ImageError),
    /// An image that folds case itself: match keys are folded already, and
    /// a name image matches them as they are.
    Insensitive,
}

impl fmt::Display for NameImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameImageError::Image(e) => write!(f, "name image: {e}"),
            NameImageError::Insensitive => f.write_str("a name image matches case sensitively"),
        }
    }
}

impl std::error::Error for NameImageError {}

/// The buffers a [`NameFinder`] works in, kept by the caller between texts.
#[derive(Default)]
pub struct NameScratch {
    folder: Folder,
    folded: String,
    spans: Vec<Span>,
    starts: Vec<u32>,
    ends: Vec<u32>,
    /// Per byte of the folded text, one more than the token that starts
    /// there, and one more than the token that ends there; 0 for none.
    start_at: Vec<u32>,
    end_at: Vec<u32>,
    glossary: Scratch,
    matches: Vec<GlossaryMatch>,
}

impl NameScratch {
    pub fn new() -> NameScratch {
        NameScratch::default()
    }
}

/// Finds a gazetteer's names in text with its key image, read in place.
#[derive(Clone, Copy, Debug)]
pub struct NameFinder<'a> {
    matcher: MappedGlossaryMatcher<'a>,
}

impl<'a> NameFinder<'a> {
    /// Checks `bytes` as a name image (see [`MappedGlossaryMatcher::from_bytes`])
    /// and reads it in place.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<NameFinder<'a>, NameImageError> {
        let matcher = MappedGlossaryMatcher::from_bytes(bytes).map_err(NameImageError::Image)?;
        if matcher.case() != Case::Sensitive {
            return Err(NameImageError::Insensitive);
        }
        Ok(NameFinder { matcher })
    }

    /// How many keys the image holds.
    pub fn keys(&self) -> usize {
        self.matcher.entries()
    }

    /// The match key of `key`, or `None` past the last one.
    pub fn key(&self, key: EntryIx) -> Option<&'a str> {
        self.matcher.term(key)
    }

    /// Appends to `out` every run of word tokens in `text` whose match key
    /// is in the image, nested and overlapping runs included, in text order
    /// (by start, then end).
    pub fn find_all(&self, text: &str, scratch: &mut NameScratch, out: &mut Vec<NameMatch>) {
        let first = out.len();
        let s = &mut *scratch;
        s.folder.fold_text(text, &mut s.folded, &mut s.spans, &mut s.starts, &mut s.ends);
        let n = s.folded.len() + 1;
        for (at, edges) in [(&mut s.start_at, &s.starts), (&mut s.end_at, &s.ends)] {
            at.clear();
            at.resize(n, 0);
            for (t, &e) in edges.iter().enumerate() {
                at[e as usize] = t as u32 + 1;
            }
        }
        let (spans, start_at, end_at) = (&s.spans, &s.start_at, &s.end_at);
        self.matcher.scan(&s.folded, &mut s.glossary, |hit, key| {
            let (i, j) = (start_at[hit.start as usize], end_at[hit.end as usize]);
            if i > 0 && j > 0 {
                out.push(NameMatch { span: Span::new(spans[i as usize - 1].start, spans[j as usize - 1].end), key });
            }
        });
        out[first..].sort_unstable();
    }

    /// Appends to `out` the names in `text` without overlaps: leftmost
    /// first, then the longest, as the glossary matcher selects.
    pub fn find(&self, text: &str, scratch: &mut NameScratch, out: &mut Vec<NameMatch>) {
        let s = &mut *scratch;
        s.folder.fold_text(text, &mut s.folded, &mut s.spans, &mut s.starts, &mut s.ends);
        let edges = s.starts.iter().chain(s.ends.iter()).copied();
        let boundaries = turbo_glossary::Boundaries::from_offsets(s.folded.len() as u32, edges);
        s.matches.clear();
        self.matcher.find(&s.folded, &boundaries, &mut s.glossary, &mut s.matches);
        for m in &s.matches {
            // Every key of an image this crate wrote starts and ends with a
            // token; a key of another image that does not is left out.
            if let (Ok(i), Ok(j)) = (s.starts.binary_search(&m.span.start), s.ends.binary_search(&m.span.end)) {
                out.push(NameMatch { span: Span::new(s.spans[i].start, s.spans[j].end), key: m.entry });
            }
        }
    }
}
