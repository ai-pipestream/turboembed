use std::fmt;
use std::path::{Component, Path};

use turbo_formats::masc::{MascCorpus, MascDocument};
use turbo_maxent::xml::{self, Attribute, Handler};
use turbo_zoo::RefusalReason;

use crate::error::{Error, Result, io};

/// A part of MASC the recipes leave out of both the training and the
/// held-out documents, wherever the split puts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// MASC's Twitter genre: in MASC 3.0.0, the two documents of
    /// `data/written/twitter` (`tweets1` and `tweets2`, 24,180 words),
    /// tweets from 2008 whose headers give their source as unknown. The
    /// licence policy refuses data under a platform's terms, and neither
    /// the release nor anc.org says whether Twitter's terms reach the
    /// tweets' text (docs/licences.md, "Datasets"); Kristian Rickert
    /// decided on 2026-10-15 to leave the genre out.
    TwitterGenre,
}

/// The exclusions the recipes apply ([`Split::apply`]).
pub const EXCLUSIONS: [Exclusion; 1] = [Exclusion::TwitterGenre];

impl Exclusion {
    /// The category of MASC's taxonomy (`resource-header.xml`) whose
    /// documents are left out, as a document header's `textClass` names
    /// it in its `catRef`: `TW`, "tweets", a genre of the written part
    /// (`WR`). In MASC 3.0.0 the two headers of `data/written/twitter`,
    /// and no other, name it.
    pub fn category(self) -> &'static str {
        match self {
            Exclusion::TwitterGenre => "TW",
        }
    }

    /// Why the documents are left out, in the licence policy's terms.
    pub fn reason(self) -> RefusalReason {
        match self {
            Exclusion::TwitterGenre => RefusalReason::PlatformTerms,
        }
    }

    /// Whether a document of `categories` is left out.
    pub fn covers(self, categories: &Categories) -> bool {
        categories.contains(self.category())
    }
}

impl fmt::Display for Exclusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Exclusion::TwitterGenre => "the Twitter genre",
        })
    }
}

/// The categories of MASC's taxonomy a document header's `textClass`
/// elements name in their `catRef` attributes, a list of codes separated
/// by whitespace (`WR TW`: written, tweets). A header without one has
/// none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Categories(Vec<String>);

impl Categories {
    /// The categories of the header at `path`.
    ///
    /// # Errors
    ///
    /// A header that cannot be read, or is not XML ([`Error::Header`]).
    pub fn of_header(path: &Path) -> Result<Categories> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let header = |e: turbo_maxent::Error| Error::Header { path: path.to_owned(), message: e.to_string() };
        Categories::parse(&xml::decode(&bytes).map_err(header)?).map_err(header)
    }

    /// The categories of a header's decoded text.
    ///
    /// # Errors
    ///
    /// Text that is not XML.
    pub fn parse(text: &str) -> std::result::Result<Categories, turbo_maxent::Error> {
        struct TextClasses(Vec<String>);
        impl Handler for TextClasses {
            fn start_element(&mut self, name: &str, a: &[Attribute<'_>], _: usize) -> turbo_maxent::Result<()> {
                if name == "textClass"
                    && let Some(codes) = a.iter().find(|a| a.name == "catRef")
                {
                    self.0.extend(codes.value.split_whitespace().map(str::to_owned));
                }
                Ok(())
            }
            fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
                Ok(())
            }
            fn characters(&mut self, _: &str) {}
        }
        let mut found = TextClasses(Vec::new());
        xml::parse(text, &mut found)?;
        Ok(Categories(found.0))
    }

    /// Whether `code` is one of the categories.
    pub fn contains(&self, code: &str) -> bool {
        self.0.iter().any(|c| c == code)
    }

    /// The codes, in the order of the header.
    pub fn codes(&self) -> &[String] {
        &self.0
    }
}

/// A document left out by an exclusion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedDocument {
    /// The document's header, relative to the corpus directory.
    pub path: String,
    pub exclusion: Exclusion,
}

/// Which documents are held out: those whose place in the corpus, in the
/// order [`MascCorpus::read`] reads them (headers sorted by name in each
/// directory), leaves `held_out_at` when divided by `every`. The rule
/// uses no randomness, so no seed; with `every` 10 it holds out one
/// document in ten, spread over the corpus's genre directories.
/// `held_out_at` is below `every`, so `every` is not 0: [`Split::new`]
/// and reading a card check it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SplitFields", into = "SplitFields")]
pub struct Split {
    every: usize,
    held_out_at: usize,
}

/// A split as written in a card, checked by `Split::try_from`.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitFields {
    every: usize,
    held_out_at: usize,
}

impl TryFrom<SplitFields> for Split {
    type Error = String;
    fn try_from(f: SplitFields) -> std::result::Result<Split, String> {
        Split::new(f.every, f.held_out_at)
            .ok_or_else(|| format!("a split of every {} at {}: the place must be below every", f.every, f.held_out_at))
    }
}

impl From<Split> for SplitFields {
    fn from(s: Split) -> SplitFields {
        SplitFields { every: s.every, held_out_at: s.held_out_at }
    }
}

/// The split of the recipes: every tenth document, from the tenth.
pub const SPLIT: Split = match Split::new(10, 9) {
    Some(s) => s,
    None => panic!("a valid split"),
};

/// A corpus split into the documents trained on and those held out.
#[derive(Debug)]
pub struct SplitCorpus {
    pub train: MascCorpus,
    pub held_out: MascCorpus,
    /// The held-out documents' paths relative to the corpus directory.
    pub held_out_paths: Vec<String>,
    /// The documents left out of both sides by an exclusion, in the order
    /// read.
    pub excluded: Vec<ExcludedDocument>,
}

impl Split {
    /// The split that holds out the documents at `held_out_at` modulo
    /// `every`, or `None` unless `held_out_at < every`.
    pub const fn new(every: usize, held_out_at: usize) -> Option<Split> {
        if held_out_at < every { Some(Split { every, held_out_at }) } else { None }
    }

    pub fn every(&self) -> usize {
        self.every
    }

    pub fn held_out_at(&self) -> usize {
        self.held_out_at
    }

    pub fn is_held_out(&self, position: usize) -> bool {
        position % self.every == self.held_out_at
    }

    /// Splits `corpus`, read from `root`, by the documents' places, and
    /// leaves out of both sides the documents [`EXCLUSIONS`] covers
    /// ([`Split::apply_excluding`]).
    ///
    /// # Errors
    ///
    /// As [`Split::apply_excluding`].
    pub fn apply(&self, corpus: MascCorpus, root: &Path) -> Result<SplitCorpus> {
        self.apply_excluding(corpus, root, &EXCLUSIONS)
    }

    /// Splits `corpus`, read from `root`, by the documents' places, and
    /// leaves out of both sides each document an exclusion of
    /// `exclusions` covers, by the categories its header names
    /// ([`Categories`]). The places are those of every document read,
    /// excluded or not, so an exclusion takes its documents out of the
    /// side they fall on and moves no other document from one side to
    /// the other. A document the corpus read skipped (it logged its error
    /// and left it out, as `MascDocumentStream` does) has no place, so it
    /// would move every document read after it to the next place and
    /// change which are held out: a corpus with one is refused. MASC 3.0.0
    /// has none.
    ///
    /// # Errors
    ///
    /// [`Error::SkippedDocuments`] for a corpus whose read skipped a
    /// document, [`Error::OutsideCorpus`] for a document whose path is not
    /// under `root`, which a corpus read from `root` does not have, and
    /// [`Error::Header`] for a header that cannot be read again for its
    /// categories.
    pub fn apply_excluding(&self, corpus: MascCorpus, root: &Path, exclusions: &[Exclusion]) -> Result<SplitCorpus> {
        let root = std::path::absolute(root).map_err(io(root))?;
        let rel = |p: &Path| -> Result<String> {
            let inside = p.strip_prefix(&root).map_err(|_| Error::OutsideCorpus { path: p.to_owned() })?;
            if inside.as_os_str().is_empty() || !inside.components().all(|c| matches!(c, Component::Normal(_))) {
                return Err(Error::OutsideCorpus { path: p.to_owned() });
            }
            // Joined with `/` on every OS, so a card reads the same wherever it was made.
            Ok(inside.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"))
        };
        if !corpus.skipped().is_empty() {
            let documents = corpus
                .skipped()
                .iter()
                .map(|s| Ok(format!("{} ({})", rel(&s.path)?, s.error)))
                .collect::<Result<_>>()?;
            return Err(Error::SkippedDocuments { documents });
        }
        let (mut train, mut held_out): (Vec<MascDocument>, Vec<MascDocument>) = (Vec::new(), Vec::new());
        let (mut held_out_paths, mut excluded) = (Vec::new(), Vec::new());
        for (i, doc) in corpus.into_documents().into_iter().enumerate() {
            let path = rel(doc.path())?;
            if !exclusions.is_empty() {
                let categories = Categories::of_header(doc.path())?;
                if let Some(&exclusion) = exclusions.iter().find(|e| e.covers(&categories)) {
                    excluded.push(ExcludedDocument { path, exclusion });
                    continue;
                }
            }
            if self.is_held_out(i) {
                held_out.push(doc);
                held_out_paths.push(path);
            } else {
                train.push(doc);
            }
        }
        Ok(SplitCorpus {
            train: MascCorpus::from_documents(train),
            held_out: MascCorpus::from_documents(held_out),
            held_out_paths,
            excluded,
        })
    }
}
