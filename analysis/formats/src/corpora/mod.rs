//! The sentence corpora OpenNLP trains and evaluates its sentence
//! detector on (Moses, LetsMT, NKJP and the Irish Sentence Bank, which
//! also gives token samples), the Leipzig corpora its language detector
//! is trained on, and the 20 Newsgroups corpus for its document
//! categorizer.
//!
//! | Corpus | Reader | OpenNLP | Registry name |
//! |---|---|---|---|
//! | Moses | [`MosesReader`] | `MosesSentenceSampleStream` | `moses` ([`MosesSentences`]) |
//! | LetsMT | [`LetsmtDocument`], [`LetsmtReader`] | `LetsmtDocument`, `LetsmtSentenceStream`, `DetokenizeSentenceSampleStream` | `letsmt` ([`LetsmtSentences`]) |
//! | NKJP | [`NkjpSegmentation`], [`NkjpText`], [`NkjpReader`] | `NKJPSegmentationDocument`, `NKJPTextDocument`, `NKJPSentenceSampleStream` | `nkjp` ([`NkjpSentences`]) |
//! | Irish Sentence Bank | [`IrishSentenceBank`] | `IrishSentenceBankDocument`, `IrishSentenceBankSentenceStream`, `IrishSentenceBankTokenSampleStream` | `irishsentencebank` ([`IrishSentences`], [`IrishTokens`]) |
//! | Leipzig | [`LeipzigCorpus`] | `LeipzigLanguageSampleStream`, `SampleShuffleStream`, `SampleSkipStream` | none: a corpus is a directory of files |
//! | 20 Newsgroups | [`newsgroup_listing`], [`newsgroup_order`], [`NewsgroupDocument`] | `TwentyNewsgroupSampleStream` | none: a corpus is a directory of directories |
//!
//! The readers take decoded text ([`crate::decode`]). OpenNLP reads Moses
//! and Leipzig files as UTF-8 and the XML files in the encoding their
//! declaration names; these readers take the decoded text whatever the
//! declaration says. Sentence and token samples have texts of their own,
//! and each sentence or token keeps the bytes of the source it came from:
//! its line, its `s`, `sentence` or `token` element, or its text.
//!
//! The XML formats are read with the XML reader of `turbo-maxent`, which
//! refuses a document type declaration as OpenNLP does; a document that
//! is not well-formed gives the one message OpenNLP wraps every parse
//! error in.
//!
//! # What OpenNLP does here
//!
//! - Moses: an empty line is an empty sentence, not a break, and a run of
//!   them is one; 25 sentences make a sample whatever the lines hold.
//! - LetsMT: one buffer of character data is emptied only at the end of
//!   a `w` or `s` element, so text before a sentence (a header, say) is
//!   part of its text when it has no words, and text between words is
//!   part of the next word.
//! - NKJP: all the sentences of a file are one sample. The pointers are
//!   UTF-16 offsets into the paragraphs; a pointer with three numbers
//!   gives its offset as `a * 1000 + b`. A sentence whose segments leave a
//!   paragraph and come back makes one sentence per stretch. A paragraph
//!   id given twice keeps the paragraph that comes last in `HashMap`
//!   order. The `nkjp:nps` attribute is read and not used.
//! - Irish Sentence Bank: all the sentences are one sentence sample. The
//!   text between two tokens is a token too unless it is exactly one
//!   space, trimmed of spaces only, so a line break or tab in it is part of
//!   a token, and two or more spaces alone throw.
//! - Leipzig: every file shares one `Random(23)`, so a file's samples
//!   depend on the files before it, and all the samples are shuffled again
//!   with a new `Random(23)` before the skip.
//! - 20 Newsgroups: a message is its whole file, headers and all, decoded
//!   as UTF-8 with replacement, and the order of the samples follows the
//!   hash codes of the files' paths, so moving the corpus changes it.
//!
//! Where this port differs, deliberately:
//!
//! - A cut between the halves of a surrogate pair ([`CorpusError::SplitsSurrogatePair`],
//!   [`CorpusError::IrishTokenSplitsSurrogatePair`]) is refused, as a Rust
//!   string cannot hold half a pair.
//! - Leipzig counts that overflow a Java `int` are refused
//!   ([`CorpusError::LeipzigParameters`]).
//! - Namespace prefixes are not checked for a declaration, which the
//!   JDK's namespace aware LetsMT parser does.
//! - Java allocates an Irish Sentence Bank sentence's flex array at its
//!   largest slot, and runs out of memory on a slot near
//!   `Integer.MAX_VALUE`; this port keeps the flexes in a list.

mod dom;
mod error;
mod formats;
mod irish;
mod leipzig;
mod letsmt;
mod moses;
mod newsgroups;
mod nkjp;

use std::fmt;
use std::path::PathBuf;

pub use error::CorpusError;
pub use formats::{IrishSentences, IrishTokens, LetsmtSentences, MosesSentences, NkjpSentences};
pub use irish::{IrishFlex, IrishSentence, IrishSentenceBank};
pub use leipzig::{LeipzigCorpus, LeipzigParams, LeipzigSample};
pub use letsmt::{LetsmtDocument, LetsmtReader, LetsmtSentence};
pub use moses::{MosesReader, SENTENCES_PER_SAMPLE};
pub use newsgroups::{NewsgroupDocument, NewsgroupFile, newsgroup_listing, newsgroup_order};
pub use nkjp::{NkjpParagraph, NkjpPointer, NkjpReader, NkjpSegmentation, NkjpSentence, NkjpText};

use crate::error::FormatError;

/// `MosesSentenceSampleStreamFactory`'s format name.
pub const MOSES_FORMAT: &str = "moses";
/// `LetsmtSentenceStreamFactory`'s format name.
pub const LETSMT_FORMAT: &str = "letsmt";
/// `NKJPSentenceSampleStreamFactory`'s format name.
pub const NKJP_FORMAT: &str = "nkjp";
/// The format name of both Irish Sentence Bank factories.
pub const IRISH_SENTENCE_BANK_FORMAT: &str = "irishsentencebank";
/// `LeipzigLanguageSampleStreamFactory`'s format name.
pub const LEIPZIG_FORMAT: &str = "leipzig";
/// `TwentyNewsgroupSampleStreamFactory`'s format name.
pub const NEWSGROUPS_FORMAT: &str = "20newsgroup";

/// An error of the readers of a corpus directory.
#[derive(Debug)]
pub enum DirectoryError {
    /// The corpus path is not a directory.
    NotADirectory { path: PathBuf },
    /// A directory could not be listed or a file read.
    Io { path: PathBuf, error: std::io::Error },
    /// What the format refuses.
    Format(FormatError),
}

impl fmt::Display for DirectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirectoryError::NotADirectory { path } => write!(f, "{} is not a directory", path.display()),
            DirectoryError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            DirectoryError::Format(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for DirectoryError {}
