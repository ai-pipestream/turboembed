//! FineWeb-2 and FineWeb, the web text the language detector recipe
//! trains on beside Tatoeba's sentences (issue #429), read by range from
//! the Hugging Face hub without the rest of each file.
//!
//! FineWeb-2 (`HuggingFaceFW/fineweb-2`) is CommonCrawl text in over a
//! thousand language and script configurations, each with a `train` and a
//! `test` split. It has no English, which FineWeb (`HuggingFaceFW/fineweb`)
//! holds. Both are published under ODC-By 1.0, their use also subject to
//! CommonCrawl's terms of use. Both label each document's language with a
//! classifier (GlotLID for FineWeb-2, fastText's language identifier for
//! FineWeb), so the labels the recipe trains on and scores against are
//! those classifiers' decisions, not people's.
//!
//! Each language of [`WEB_LANGUAGES`] reads one or two sources: the
//! FineWeb-2 configuration of the language in the script Tatoeba writes it
//! in (two for Serbian and Uzbek, which Tatoeba writes in two scripts), or
//! FineWeb's 10BT sample for English. A source's training documents come
//! from the first file of its `train` split and its held-out ones from the
//! first file of its `test` split; FineWeb has no test split, so its
//! held-out documents come from the sample's second file.
//!
//! A file is read from three ranges ([`FileRanges`]): its last 8 bytes,
//! the first [`WebSplit::footer_prefix`] bytes of its footer, and its
//! first bytes up to the end of the fewest leading row groups that hold
//! [`WebSplit::rows_per_document`] rows for each document wanted, or all
//! of its row groups when they hold fewer. The files are listed with their
//! size and SHA-256 as the hub's API gives them at the datasets'
//! revisions ([`FILES`]), and the ranges with theirs ([`RANGES`]). The
//! binary checks ranges, not whole files, so a file's SHA-256 records
//! which file it is and is not checked; a run checks every range it
//! reads against its SHA-256, and that the ranges listed are the ones the
//! files give.
//!
//! From those rows, in order, each document gives its first
//! [`WebSplit::window`] characters, or all of them when it has fewer. A
//! document with fewer than [`WebSplit::min_chars`] characters is passed
//! over, and so is one from a host of [`EXCLUDED_HOSTS`], a window its
//! language already has or an earlier row of its file gave, and a window
//! that holds a Tatoeba sentence of its language of at least
//! [`WebSplit::overlap_min_chars`] characters from the other side of the
//! split, byte for byte ([`Avoid`]). A file's
//! documents are taken in row order, at first at most one in
//! [`WebSplit::host_share`] from one host; when that leaves the file short,
//! the documents passed over for their host are taken next, in row order.
//! A held-out window that is a training window of any language is passed
//! over. A file gives at most the documents wanted of it: one with too few
//! rows left gives fewer, and a language's other source does not make up
//! the difference.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo_eval::read::parquet::{Leading, Parquet, footer_len};
use turbo_zoo::{Licence, Sha256};

/// The files the recipe reads, as `fineweb-files.txt` lists them.
pub const FILES: &str = include_str!("../fineweb-files.txt");

/// The ranges of those files the recipe reads, as `fineweb-ranges.txt`
/// lists them.
pub const RANGES: &str = include_str!("../fineweb-ranges.txt");

/// The name [`FILES`] is read under in errors.
pub const FILES_NAME: &str = "fineweb-files.txt";

/// The name [`RANGES`] is read under in errors.
pub const RANGES_NAME: &str = "fineweb-ranges.txt";

/// A dataset on the Hugging Face hub, at the revision the recipe reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// Its name, as its card gives it.
    pub name: &'static str,
    pub repo: &'static str,
    /// The commit of the dataset's repository the files are read at.
    pub revision: &'static str,
    pub licence: Licence,
    /// Where the dataset's card states the licence.
    pub licence_terms: &'static str,
}

impl Dataset {
    /// The address of `path` at the dataset's revision.
    pub fn url(&self, path: &str) -> String {
        format!("https://huggingface.co/datasets/{}/resolve/{}/{path}", self.repo, self.revision)
    }
}

pub const FINEWEB_2: Dataset = Dataset {
    name: "FineWeb-2",
    repo: "HuggingFaceFW/fineweb-2",
    revision: "af9c13333eb981300149d5ca60a8e9d659b276b9",
    licence: Licence::OdcBy10,
    licence_terms: "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2#licensing-information",
};

pub const FINEWEB: Dataset = Dataset {
    name: "FineWeb",
    repo: "HuggingFaceFW/fineweb",
    revision: "9bb295ddab0e05d785b879661af7260fed5140fc",
    licence: Licence::OdcBy10,
    licence_terms: "https://huggingface.co/datasets/HuggingFaceFW/fineweb#licensing-information",
};

/// The terms of use both datasets' cards say their use is also subject to.
pub const COMMONCRAWL_TERMS: &str = "https://commoncrawl.org/terms-of-use";

/// Where a language's web documents come from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A FineWeb-2 configuration: an ISO 639-3 code and an ISO 15924
    /// script.
    FineWeb2(&'static str),
    /// FineWeb's 10BT sample, which is English.
    FineWeb,
}

/// Which side of the split a file's documents are on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Training,
    HeldOut,
}

impl Source {
    pub fn dataset(self) -> Dataset {
        match self {
            Source::FineWeb2(_) => FINEWEB_2,
            Source::FineWeb => FINEWEB,
        }
    }

    /// The file `side`'s documents come from.
    pub fn path(self, side: Side) -> String {
        match (self, side) {
            (Source::FineWeb2(c), Side::Training) => format!("data/{c}/train/000_00000.parquet"),
            (Source::FineWeb2(c), Side::HeldOut) => format!("data/{c}/test/000_00000.parquet"),
            (Source::FineWeb, Side::Training) => "sample/10BT/000_00000.parquet".to_owned(),
            (Source::FineWeb, Side::HeldOut) => "sample/10BT/001_00000.parquet".to_owned(),
        }
    }
}

/// A language of the recipe and where its web documents come from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WebLanguage {
    /// Tatoeba's code, the trained model's outcome.
    pub tatoeba: &'static str,
    pub sources: &'static [Source],
}

macro_rules! sources {
    (fineweb) => {
        &[Source::FineWeb]
    };
    ($($config:literal),+) => {
        &[$(Source::FineWeb2($config)),+]
    };
}

macro_rules! web {
    ($($tatoeba:literal => [$($s:tt)+]),* $(,)?) => {
        [$(WebLanguage { tatoeba: $tatoeba, sources: sources!($($s)+) }),*]
    };
}

/// The languages of [`crate::langdetect::LANGUAGES`], in its order, and
/// their sources. Most FineWeb-2 configurations share Tatoeba's code;
/// where they do not, the configuration is the language Tatoeba's code
/// names or, for a macrolanguage, the individual language its written
/// standard is: Standard Arabic (`arb`) for `ara`, North Azerbaijani
/// (`azj`) for `aze`, Standard Estonian (`ekk`) for `est`, Plateau
/// Malagasy (`plt`) for `mlg`, Halh Mongolian (`khk`) for `mon`, Southern
/// Pashto (`pbt`) for `pus`, Tosk Albanian (`als`) for `sqi` and Northern
/// Uzbek (`uzn`) for `uzb`. FineWeb-2 has no `pes`, `tgl` or `ara`
/// configuration: Persian is `fas`, Tagalog `fil`. Where FineWeb-2 also
/// has a romanized configuration of a language (`hin_Latn` and the like),
/// it is left out: Tatoeba writes those languages in their own scripts.
pub const WEB_LANGUAGES: [WebLanguage; 100] = web![
    "afr" => ["afr_Latn"], "ara" => ["arb_Arab"], "ast" => ["ast_Latn"], "aze" => ["azj_Latn"],
    "bak" => ["bak_Cyrl"], "bel" => ["bel_Cyrl"], "ben" => ["ben_Beng"], "bos" => ["bos_Latn"],
    "bre" => ["bre_Latn"], "bul" => ["bul_Cyrl"], "cat" => ["cat_Latn"], "ceb" => ["ceb_Latn"],
    "ces" => ["ces_Latn"], "che" => ["che_Cyrl"], "cmn" => ["cmn_Hani"], "cym" => ["cym_Latn"],
    "dan" => ["dan_Latn"], "deu" => ["deu_Latn"], "ell" => ["ell_Grek"], "eng" => [fineweb],
    "epo" => ["epo_Latn"], "est" => ["ekk_Latn"], "eus" => ["eus_Latn"], "fao" => ["fao_Latn"],
    "fin" => ["fin_Latn"], "fra" => ["fra_Latn"], "fry" => ["fry_Latn"], "gle" => ["gle_Latn"],
    "glg" => ["glg_Latn"], "gsw" => ["gsw_Latn"], "guj" => ["guj_Gujr"], "heb" => ["heb_Hebr"],
    "hin" => ["hin_Deva"], "hrv" => ["hrv_Latn"], "hun" => ["hun_Latn"], "hye" => ["hye_Armn"],
    "ind" => ["ind_Latn"], "isl" => ["isl_Latn"], "ita" => ["ita_Latn"], "jav" => ["jav_Latn"],
    "jpn" => ["jpn_Jpan"], "kan" => ["kan_Knda"], "kat" => ["kat_Geor"], "kaz" => ["kaz_Cyrl"],
    "kir" => ["kir_Cyrl"], "kor" => ["kor_Hang"], "lat" => ["lat_Latn"], "lim" => ["lim_Latn"],
    "lit" => ["lit_Latn"], "ltz" => ["ltz_Latn"], "lvs" => ["lvs_Latn"], "mal" => ["mal_Mlym"],
    "mar" => ["mar_Deva"], "min" => ["min_Latn"], "mkd" => ["mkd_Cyrl"], "mlg" => ["plt_Latn"],
    "mlt" => ["mlt_Latn"], "mon" => ["khk_Cyrl"], "mri" => ["mri_Latn"], "nan" => ["nan_Latn"],
    "nds" => ["nds_Latn"], "nld" => ["nld_Latn"], "nno" => ["nno_Latn"], "nob" => ["nob_Latn"],
    "npi" => ["npi_Deva"], "oci" => ["oci_Latn"], "pan" => ["pan_Guru"], "pes" => ["fas_Arab"],
    "pnb" => ["pnb_Arab"], "pol" => ["pol_Latn"], "por" => ["por_Latn"], "pus" => ["pbt_Arab"],
    "ron" => ["ron_Latn"], "rus" => ["rus_Cyrl"], "san" => ["san_Deva"], "sin" => ["sin_Sinh"],
    "slk" => ["slk_Latn"], "slv" => ["slv_Latn"], "som" => ["som_Latn"], "spa" => ["spa_Latn"],
    "sqi" => ["als_Latn"], "srp" => ["srp_Cyrl", "srp_Latn"], "sun" => ["sun_Latn"],
    "swe" => ["swe_Latn"], "swh" => ["swh_Latn"], "tam" => ["tam_Taml"], "tat" => ["tat_Cyrl"],
    "tel" => ["tel_Telu"], "tgk" => ["tgk_Cyrl"], "tgl" => ["fil_Latn"], "tha" => ["tha_Thai"],
    "tur" => ["tur_Latn"], "ukr" => ["ukr_Cyrl"], "urd" => ["urd_Arab"], "uzb" => ["uzn_Latn", "uzn_Cyrl"],
    "vie" => ["vie_Latn"], "vol" => ["vol_Latn"], "war" => ["war_Latn"], "zsm" => ["zsm_Latn"],
    "zul" => ["zul_Latn"],
];

/// How many web documents each language gives, and how they are read and
/// chosen: see the module comment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSplit {
    /// Training windows per language, shared among its sources.
    pub training: usize,
    /// Held-out windows per language, shared the same way.
    pub held_out: usize,
    /// A window's most characters (Unicode scalar values).
    pub window: usize,
    /// The fewest characters a document needs to give a window.
    pub min_chars: usize,
    /// At first, at most one in `host_share` of a file's documents comes
    /// from one host.
    pub host_share: usize,
    /// Rows read for each document a file gives, so that documents passed
    /// over leave enough.
    pub rows_per_document: u64,
    /// The bytes read of each footer.
    pub footer_prefix: u64,
    /// The fewest characters a Tatoeba sentence needs to be looked for in
    /// the windows ([`Overlap`]). Shorter ones are common phrases, found in
    /// web text whatever its source.
    pub overlap_min_chars: usize,
}

/// The recipe's web split: per language at most 1,000 training windows
/// of at most 300 characters, a fifth of the Tatoeba sentences' count and
/// about twice their text, which keeps the events of 100 languages within
/// the runner's memory beside Tatoeba's; at most 500 held out, as many as
/// Tatoeba's. 64 KiB of a footer holds its schema and the leading row
/// groups the recipe reads of each of its files; a file whose footer's
/// first bytes hold too few is refused.
pub const WEB_SPLIT: WebSplit = WebSplit {
    training: 1_000,
    held_out: 500,
    window: 300,
    min_chars: 100,
    host_share: 20,
    rows_per_document: 2,
    footer_prefix: 65_536,
    overlap_min_chars: 20,
};

/// Hosts whose documents are passed over, with their subdomains:
/// Tatoeba's own pages, whose sentences are on the other side of the
/// recipe.
pub const EXCLUDED_HOSTS: &[&str] = &["tatoeba.org"];

/// Whether `host` ([`host`]) is one of [`EXCLUDED_HOSTS`] or a subdomain
/// of one.
pub fn excluded_host(host: &str) -> bool {
    EXCLUDED_HOSTS.iter().any(|x| host == *x || host.strip_suffix(*x).is_some_and(|h| h.ends_with('.')))
}

/// The bytes of a sentence's start that [`Overlap`] looks up.
pub const OVERLAP_KEY: usize = 16;

/// Texts looked for in windows, per language of the recipe: each kept by
/// its first [`OVERLAP_KEY`] bytes, so a window is searched in one pass.
pub struct Overlap<'a> {
    by_language: Vec<HashMap<&'a [u8], Vec<&'a str>>>,
}

impl<'a> Overlap<'a> {
    /// The texts of `texts`, each with its language's index, that have at
    /// least `min_chars` characters and [`OVERLAP_KEY`] bytes.
    pub fn new(languages: usize, min_chars: usize, texts: impl IntoIterator<Item = (usize, &'a str)>) -> Overlap<'a> {
        let mut by_language: Vec<HashMap<&'a [u8], Vec<&'a str>>> = vec![HashMap::new(); languages];
        for (language, text) in texts {
            if text.len() >= OVERLAP_KEY && text.chars().nth(min_chars.saturating_sub(1)).is_some() {
                by_language[language].entry(&text.as_bytes()[..OVERLAP_KEY]).or_default().push(text);
            }
        }
        Overlap { by_language }
    }

    /// No texts.
    pub fn none(languages: usize) -> Overlap<'static> {
        Overlap { by_language: vec![HashMap::new(); languages] }
    }

    /// Whether `window` holds a text of language `language`, byte for
    /// byte.
    pub fn found(&self, language: usize, window: &str) -> bool {
        let texts = &self.by_language[language];
        let b = window.as_bytes();
        if texts.is_empty() || b.len() < OVERLAP_KEY {
            return false;
        }
        // A text starts with a character's first byte, so a match found
        // at any byte starts at a character.
        (0..=b.len() - OVERLAP_KEY).any(|i| {
            texts.get(&b[i..i + OVERLAP_KEY]).is_some_and(|ts| ts.iter().any(|t| b[i..].starts_with(t.as_bytes())))
        })
    }
}

/// The Tatoeba sentences the windows are checked against: a training
/// window may not hold a held-out sentence of its language, which would
/// favour the trained model on the Tatoeba gate, nor a held-out window a
/// training sentence of its language. Only sentences of at least
/// [`WebSplit::overlap_min_chars`] characters are looked for, byte for
/// byte.
pub struct Avoid<'a> {
    /// What a training window may not hold.
    pub training: Overlap<'a>,
    /// What a held-out window may not hold.
    pub held_out: Overlap<'a>,
}

impl Avoid<'static> {
    /// Nothing to avoid.
    pub fn none(languages: usize) -> Avoid<'static> {
        Avoid { training: Overlap::none(languages), held_out: Overlap::none(languages) }
    }
}

/// The languages whose web text is read and how: the web side of a
/// recipe.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WebRecipe {
    pub languages: &'static [WebLanguage],
    pub split: WebSplit,
}

/// [`WEB_LANGUAGES`] split by [`WEB_SPLIT`].
pub const WEB: WebRecipe = WebRecipe { languages: &WEB_LANGUAGES, split: WEB_SPLIT };

/// One file the recipe reads: a source's side of a language, and how many
/// documents it gives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Want {
    /// The language's index in [`WebRecipe::languages`].
    pub language: usize,
    pub source: Source,
    pub side: Side,
    pub documents: usize,
}

impl Want {
    pub fn repo(&self) -> &'static str {
        self.source.dataset().repo
    }

    pub fn path(&self) -> String {
        self.source.path(self.side)
    }
}

impl WebRecipe {
    /// Every file the recipe reads: the training files of every language
    /// in order, each language's sources in order, then the held-out files
    /// the same way. A side's documents are shared among a language's
    /// sources, the first ones taking one more when they do not divide
    /// evenly.
    pub fn wants(&self) -> Vec<Want> {
        let mut out = Vec::new();
        for (side, total) in [(Side::Training, self.split.training), (Side::HeldOut, self.split.held_out)] {
            for (language, l) in self.languages.iter().enumerate() {
                let n = l.sources.len();
                for (k, &source) in l.sources.iter().enumerate() {
                    let documents = total / n + usize::from(k < total % n);
                    out.push(Want { language, source, side, documents });
                }
            }
        }
        out
    }
}

impl WebSplit {
    /// The rows read of a file that gives `documents` documents.
    pub fn rows(&self, documents: usize) -> u64 {
        documents as u64 * self.rows_per_document
    }

    /// The most documents of one file one host gives at first.
    pub fn host_limit(&self, documents: usize) -> usize {
        (documents / self.host_share).max(1)
    }
}

/// Why the web files could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebError {
    /// A line of [`FILES`] or [`RANGES`] that is not as the recipe writes
    /// them.
    Line {
        list: String,
        line: usize,
        message: String,
    },
    /// The files list does not list exactly the files the recipe reads.
    Files {
        message: String,
    },
    /// The ranges list does not list exactly the ranges the files give.
    Ranges {
        message: String,
    },
    /// A range a run needs that the directory does not hold.
    Missing {
        path: PathBuf,
    },
    /// A range whose bytes are not the ones recorded.
    Digest {
        path: PathBuf,
        expected: String,
        found: String,
    },
    /// A file whose ranges are not read as the recipe needs.
    File {
        repo: String,
        path: String,
        message: String,
    },
    /// What the Parquet reader refused, naming the file by its repository
    /// and path.
    Read {
        message: String,
    },
    Io {
        path: PathBuf,
        message: String,
    },
}

impl fmt::Display for WebError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebError::Line { list, line, message } => write!(f, "{list}:{line}: {message}"),
            WebError::Files { message } => write!(f, "{FILES_NAME}: {message}"),
            WebError::Ranges { message } => write!(f, "{RANGES_NAME}: {message}"),
            WebError::Missing { path } => write!(f, "{}: not fetched", path.display()),
            WebError::Digest { path, expected, found } => {
                write!(f, "{}: SHA-256 {found}, the list records {expected}", path.display())
            }
            WebError::File { repo, path, message } => write!(f, "{repo} {path}: {message}"),
            WebError::Read { message } => write!(f, "{message}"),
            WebError::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for WebError {}

type Result<T> = std::result::Result<T, WebError>;

/// One line of [`FILES`]: a file at a dataset's revision, its size and
/// its SHA-256 as the hub's API gives them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFile {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Bytes `start` up to, not including, `end` of a file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: u64,
    pub end: u64,
}

impl Span {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

/// One line of [`RANGES`]: a range of a listed file and its SHA-256.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebRange {
    pub repo: String,
    pub path: String,
    pub span: Span,
    pub sha256: String,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The dataset of `repo`, one of the two the recipe reads.
fn dataset(repo: &str) -> Option<Dataset> {
    [FINEWEB_2, FINEWEB].into_iter().find(|d| d.repo == repo)
}

/// The tab separated fields of each line that is neither empty nor a
/// comment, with its line number.
fn lines<'a>(list: &str, text: &'a str, fields: usize) -> Result<Vec<(usize, Vec<&'a str>)>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != fields {
            return Err(WebError::Line {
                list: list.to_owned(),
                line: i + 1,
                message: format!("{} fields, not {fields}", f.len()),
            });
        }
        out.push((i + 1, f));
    }
    Ok(out)
}

fn number(list: &str, line: usize, what: &str, s: &str) -> Result<u64> {
    let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
    // Decimal digits only, so each number has one spelling.
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0')) {
        return Err(err(format!("the {what} {s:?} is not a number")));
    }
    s.parse().map_err(|_| err(format!("the {what} {s} is too large")))
}

/// Reads a files list: `repo`, `revision`, `path`, `size` and `sha256`,
/// tab separated. Each repository must be one of the two datasets at its
/// revision, and no file may be listed twice. `list` names the list in
/// errors.
pub fn parse_files(list: &str, text: &str) -> Result<Vec<WebFile>> {
    let mut out: Vec<WebFile> = Vec::new();
    let mut seen = HashSet::new();
    for (line, f) in lines(list, text, 5)? {
        let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
        let d = dataset(f[0]).ok_or_else(|| err(format!("{} is not a dataset the recipe reads", f[0])))?;
        if f[1] != d.revision {
            return Err(err(format!("{} at {}, not the recipe's revision {}", d.repo, f[1], d.revision)));
        }
        if !is_sha256(f[4]) {
            return Err(err(format!("{:?} is not a SHA-256 in lowercase hex", f[4])));
        }
        if !seen.insert((f[0], f[2])) {
            return Err(err(format!("{} {} is listed twice", f[0], f[2])));
        }
        out.push(WebFile {
            repo: f[0].to_owned(),
            revision: f[1].to_owned(),
            path: f[2].to_owned(),
            size: number(list, line, "size", f[3])?,
            sha256: f[4].to_owned(),
        });
    }
    Ok(out)
}

/// Reads a ranges list: `repo`, `path`, `start`, `end` and `sha256`, tab
/// separated, `end` not included. No range may be listed twice or be
/// empty. `list` names the list in errors.
pub fn parse_ranges(list: &str, text: &str) -> Result<Vec<WebRange>> {
    let mut out: Vec<WebRange> = Vec::new();
    let mut seen = HashSet::new();
    for (line, f) in lines(list, text, 5)? {
        let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
        let span = Span { start: number(list, line, "start", f[2])?, end: number(list, line, "end", f[3])? };
        if span.end <= span.start {
            return Err(err(format!("the range {span} is empty")));
        }
        if !is_sha256(f[4]) {
            return Err(err(format!("{:?} is not a SHA-256 in lowercase hex", f[4])));
        }
        if !seen.insert((f[0], f[1], span)) {
            return Err(err(format!("{} {} {span} is listed twice", f[0], f[1])));
        }
        out.push(WebRange { repo: f[0].to_owned(), path: f[1].to_owned(), span, sha256: f[4].to_owned() });
    }
    Ok(out)
}

/// A files list as [`parse_files`] reads it, with a header.
pub fn write_files(files: &[WebFile]) -> String {
    let mut out = String::from(
        "# The FineWeb-2 and FineWeb files the language detector recipe reads (turbo_train_recipes::fineweb),\n\
         # with each file's size and SHA-256 as the Hugging Face hub's API gives them at the revision.\n\
         # repo\trevision\tpath\tsize\tsha256\n",
    );
    for f in files {
        out.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", f.repo, f.revision, f.path, f.size, f.sha256));
    }
    out
}

/// A ranges list as [`parse_ranges`] reads it, with a header.
pub fn write_ranges(ranges: &[WebRange]) -> String {
    let mut out = String::from(
        "# The ranges of the files in fineweb-files.txt the language detector recipe reads\n\
         # (turbo_train_recipes::fineweb), each with its SHA-256; end is not included.\n\
         # repo\tpath\tstart\tend\tsha256\n",
    );
    for r in ranges {
        out.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", r.repo, r.path, r.span.start, r.span.end, r.sha256));
    }
    out
}

/// The files of `wants`, in their order, from `files`, which must list
/// exactly those.
pub fn files_of<'a>(wants: &[Want], files: &'a [WebFile]) -> Result<Vec<&'a WebFile>> {
    let by_path: HashMap<(&str, &str), &WebFile> =
        files.iter().map(|f| ((f.repo.as_str(), f.path.as_str()), f)).collect();
    let mut out = Vec::with_capacity(wants.len());
    let mut used = HashSet::new();
    for w in wants {
        let path = w.path();
        let f = by_path.get(&(w.repo(), path.as_str())).ok_or_else(|| WebError::Files {
            message: format!("{} {path}, which the recipe reads, is not listed", w.repo()),
        })?;
        used.insert((f.repo.as_str(), f.path.as_str()));
        out.push(*f);
    }
    if let Some(f) = files.iter().find(|f| !used.contains(&(f.repo.as_str(), f.path.as_str()))) {
        return Err(WebError::Files {
            message: format!("{} {} is listed, and the recipe does not read it", f.repo, f.path),
        });
    }
    Ok(out)
}

/// Fetched ranges in a directory: range `span` of a file at `path` in
/// `repo` at `revision` is the file
/// `<dir>/<repo>/<revision>/<path>.<start>-<end>`, so a range kept from
/// one revision is never read as another's.
pub struct Store<'a> {
    dir: &'a Path,
    /// The SHA-256 each recorded range must have; `None` when no list is
    /// checked, as when the ranges are first fetched.
    recorded: Option<HashMap<(String, String, Span), String>>,
}

impl<'a> Store<'a> {
    pub fn new(dir: &'a Path, recorded: Option<&[WebRange]>) -> Store<'a> {
        let recorded = recorded
            .map(|rs| rs.iter().map(|r| ((r.repo.clone(), r.path.clone(), r.span), r.sha256.clone())).collect());
        Store { dir, recorded }
    }

    /// Where range `span` of `file` is kept.
    pub fn file(&self, file: &WebFile, span: Span) -> PathBuf {
        self.dir.join(&file.repo).join(&file.revision).join(format!("{}.{span}", file.path))
    }

    /// The bytes of range `span` of `file`, `None` when the directory does
    /// not hold it. They must be `span`'s length, and when a list is
    /// checked, the range must be in it with these bytes' SHA-256.
    pub fn get(&self, file: &WebFile, span: Span) -> Result<Option<Vec<u8>>> {
        let at = self.file(file, span);
        let expected = match &self.recorded {
            None => None,
            Some(r) => Some(r.get(&(file.repo.clone(), file.path.clone(), span)).ok_or_else(|| WebError::Ranges {
                message: format!("{} {} {span}, which the recipe reads, is not listed", file.repo, file.path),
            })?),
        };
        let bytes = match std::fs::read(&at) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(WebError::Io { path: at, message: e.to_string() }),
        };
        if bytes.len() as u64 != span.len() {
            return Err(WebError::Io {
                path: at,
                message: format!("{} bytes, not the range's {}", bytes.len(), span.len()),
            });
        }
        if let Some(expected) = expected {
            let found = Sha256::of(&bytes).to_string();
            if found != *expected {
                return Err(WebError::Digest { path: at, expected: expected.clone(), found });
            }
        }
        Ok(Some(bytes))
    }
}

/// The ranges of one file the recipe reads, as far as the bytes at hand
/// give them: the last 8 bytes always; the footer's first bytes once the
/// last 8 are at hand; the head, and the row groups it holds, once those
/// are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRanges {
    pub tail: Span,
    pub prefix: Option<Span>,
    pub head: Option<(Span, usize)>,
}

impl FileRanges {
    /// The ranges known, in the order they are fetched.
    pub fn known(&self) -> Vec<Span> {
        let mut out = vec![self.tail];
        out.extend(self.prefix);
        out.extend(self.head.map(|(s, _)| s));
        out
    }
}

/// A file's ranges read as far as the bytes at hand go, and its leading
/// row groups once its footer's first bytes are read.
struct Planned {
    ranges: FileRanges,
    leading: Option<Leading>,
}

fn file_err(file: &WebFile, message: String) -> WebError {
    WebError::File { repo: file.repo.clone(), path: file.path.clone(), message }
}

/// The name the Parquet reader gives `file` in its errors.
fn reader_name(file: &WebFile) -> String {
    format!("{} {}", file.repo, file.path)
}

fn read_err(message: String) -> WebError {
    WebError::Read { message }
}

/// The ranges of `file` the recipe reads for `rows` rows.
fn plan_file(file: &WebFile, rows: u64, split: &WebSplit, store: &Store) -> Result<Planned> {
    if file.size < 12 {
        return Err(file_err(file, format!("{} bytes is too short for a Parquet file", file.size)));
    }
    let tail = Span { start: file.size - 8, end: file.size };
    let mut ranges = FileRanges { tail, prefix: None, head: None };
    let Some(t) = store.get(file, tail)? else {
        return Ok(Planned { ranges, leading: None });
    };
    let name = reader_name(file);
    let len = footer_len(&name, file.size, &t).map_err(read_err)?;
    let start = file.size - 8 - len;
    let prefix = Span { start, end: start + len.min(split.footer_prefix) };
    ranges.prefix = Some(prefix);
    let Some(p) = store.get(file, prefix)? else {
        return Ok(Planned { ranges, leading: None });
    };
    let leading = Leading::read(&name, file.size, len, &p).map_err(read_err)?;
    let spans = leading.spans();
    let mut held = 0u64;
    let mut groups = None;
    for (k, s) in spans.iter().enumerate() {
        held += s.rows;
        if held >= rows {
            groups = Some(k + 1);
            break;
        }
    }
    let groups = match groups {
        Some(k) => k,
        None if spans.len() == leading.row_groups() => spans.len(),
        None => {
            return Err(file_err(
                file,
                format!(
                    "the first {} bytes of the footer hold {} of {} row groups, {held} rows, and {rows} are wanted",
                    prefix.len(),
                    spans.len(),
                    leading.row_groups()
                ),
            ));
        }
    };
    if groups == 0 {
        return Err(file_err(file, "the file has no row groups".to_owned()));
    }
    let end = spans[..groups].iter().map(|s| s.end).max().expect("at least one group");
    ranges.head = Some((Span { start: 0, end }, groups));
    Ok(Planned { ranges, leading: Some(leading) })
}

/// The ranges the recipe reads of `files` that the bytes at hand give and
/// the directory does not hold, each with its file, in the order of
/// `split`'s wants.
pub fn needed<'a>(files: &'a [WebFile], web: &WebRecipe, store: &Store) -> Result<Vec<(&'a WebFile, Span)>> {
    let (wants, split) = (web.wants(), &web.split);
    let mut out = Vec::new();
    for (w, f) in wants.iter().zip(files_of(&wants, files)?) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        for span in planned.ranges.known() {
            if store.get(f, span)?.is_none() {
                out.push((f, span));
                break;
            }
        }
    }
    Ok(out)
}

/// Every range the recipe reads of `files`, with its SHA-256, in the
/// order of `split`'s wants. Each must be in the directory.
pub fn record(files: &[WebFile], web: &WebRecipe, store: &Store) -> Result<Vec<WebRange>> {
    let (wants, split) = (web.wants(), &web.split);
    let mut out = Vec::new();
    for (w, f) in wants.iter().zip(files_of(&wants, files)?) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        let known = planned.ranges.known();
        if known.len() < 3 {
            let missing = known.last().expect("the tail is always known");
            return Err(WebError::Missing { path: store.file(f, *missing) });
        }
        for span in known {
            let bytes = store.get(f, span)?.ok_or_else(|| WebError::Missing { path: store.file(f, span) })?;
            out.push(WebRange {
                repo: f.repo.clone(),
                path: f.path.clone(),
                span,
                sha256: Sha256::of(&bytes).to_string(),
            });
        }
    }
    Ok(out)
}

/// Checks that `listed` is exactly the ranges `files` give, as
/// [`record`] made them from the directory, in the same order.
pub fn check_ranges(listed: &[WebRange], made: &[WebRange]) -> Result<()> {
    for (i, (l, m)) in listed.iter().zip(made).enumerate() {
        if l != m {
            return Err(WebError::Ranges {
                message: format!(
                    "range {} is {} {} {} {}, the files give {} {} {} {}",
                    i + 1,
                    l.repo,
                    l.path,
                    l.span,
                    l.sha256,
                    m.repo,
                    m.path,
                    m.span,
                    m.sha256
                ),
            });
        }
    }
    if listed.len() != made.len() {
        return Err(WebError::Ranges {
            message: format!("{} ranges are listed, and the files give {}", listed.len(), made.len()),
        });
    }
    Ok(())
}

/// The SHA-256 of the files and ranges lists as [`write_files`] and
/// [`write_ranges`] write them: what the held-out set and the card name
/// the web data by.
pub fn lists_digest(files: &[WebFile], ranges: &[WebRange]) -> String {
    let mut text = write_files(files);
    text.push_str(&write_ranges(ranges));
    Sha256::of(text.as_bytes()).to_string()
}

/// The host of a URL, in lowercase and without a leading `www.` or a
/// trailing dot: what lies between `://` (or a leading `//`) and the next
/// `/`, `?` or `#`, less a user name and a port. A string without either
/// is read from its start.
pub fn host(url: &str) -> String {
    let rest = url.split_once("://").map_or_else(|| url.strip_prefix("//").unwrap_or(url), |(_, r)| r);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // A port follows the last colon, except inside an IPv6 address's
    // brackets.
    let h = match authority.rfind(':') {
        Some(i) if !authority[i..].contains(']') => &authority[..i],
        _ => authority,
    };
    let h = h.strip_suffix('.').unwrap_or(h).to_lowercase();
    match h.strip_prefix("www.") {
        Some(s) => s.to_owned(),
        None => h,
    }
}

/// The first `n` characters of `text`, or all of it when it has fewer.
pub fn window(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

/// A window trained on or held out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebSample {
    /// The language's index in [`WebRecipe::languages`].
    pub language: usize,
    /// The index of its file in the files list.
    pub file: usize,
    /// The document's row in its file.
    pub row: u64,
    /// The document's `id`.
    pub id: String,
    pub window: String,
}

/// How one file's documents were read and chosen.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileChoice {
    pub tatoeba: String,
    pub repo: String,
    pub path: String,
    pub side: Side,
    /// The documents wanted of the file, the most it gives.
    pub wanted: usize,
    pub file_rows: u64,
    pub file_row_groups: usize,
    pub row_groups_read: usize,
    pub rows_read: u64,
    /// Rows passed over for having fewer than [`WebSplit::min_chars`]
    /// characters.
    pub too_short: usize,
    /// Rows passed over for their host ([`EXCLUDED_HOSTS`]).
    pub excluded_host: usize,
    /// Windows passed over as one the language already has or an earlier
    /// row of the file gave (taken or not), or, held out, one any language
    /// trains on.
    pub duplicates: usize,
    /// Windows passed over for holding a Tatoeba sentence of their
    /// language from the other side of the split ([`Avoid`]).
    pub tatoeba_overlap: usize,
    /// Documents passed over at first for their host's share.
    pub host_deferred: usize,
    /// Of those, the ones taken afterwards to fill the file's share.
    pub host_taken_later: usize,
    pub chosen: usize,
}

/// The web documents trained on and held out.
#[derive(Clone, Debug, Default)]
pub struct WebPlan {
    /// Language by language in [`WebRecipe::languages`]' order, each source's in
    /// row order.
    pub training: Vec<WebSample>,
    /// The same way.
    pub held_out: Vec<WebSample>,
    /// Each file read, in the order of [`WebSplit::wants`].
    pub files: Vec<FileChoice>,
}

/// Reads the documents of every file of `files` from the directory and
/// chooses each language's training and held-out windows, none holding a
/// sentence of `avoid`.
///
/// # Errors
///
/// A range missing or not the listed bytes, a file whose leading row
/// groups cannot be read, or a null `text` or `id`.
pub fn plan_web(files: &[WebFile], web: &WebRecipe, store: &Store, avoid: &Avoid) -> Result<WebPlan> {
    let (wants, split) = (web.wants(), &web.split);
    let chosen_files = files_of(&wants, files)?;
    let index: HashMap<(&str, &str), usize> =
        files.iter().enumerate().map(|(i, f)| ((f.repo.as_str(), f.path.as_str()), i)).collect();
    let mut plan = WebPlan::default();
    // Each language's windows, and every training window.
    let mut by_language: Vec<HashSet<String>> = vec![HashSet::new(); web.languages.len()];
    let mut training: HashSet<String> = HashSet::new();
    for (w, f) in wants.iter().zip(chosen_files) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        let leading = planned.leading.ok_or_else(|| WebError::Missing {
            path: store.file(f, *planned.ranges.known().last().expect("the tail is always known")),
        })?;
        let (head_span, groups) = planned.ranges.head.expect("known once the footer's first bytes are read");
        let head = store.get(f, head_span)?.ok_or_else(|| WebError::Missing { path: store.file(f, head_span) })?;
        let name = reader_name(f);
        let table = Parquet::open_head(&name, &head, &leading, groups).map_err(read_err)?;
        let texts = table.strings("text").map_err(read_err)?;
        let urls = table.strings("url").map_err(read_err)?;
        let ids = table.strings("id").map_err(read_err)?;
        let mut choice = FileChoice {
            tatoeba: web.languages[w.language].tatoeba.to_owned(),
            repo: f.repo.clone(),
            path: f.path.clone(),
            side: w.side,
            wanted: w.documents,
            file_rows: leading.rows(),
            file_row_groups: leading.row_groups(),
            row_groups_read: groups,
            rows_read: table.rows() as u64,
            too_short: 0,
            excluded_host: 0,
            duplicates: 0,
            tatoeba_overlap: 0,
            host_deferred: 0,
            host_taken_later: 0,
            chosen: 0,
        };
        // The rows that give a window, in order, with their hosts.
        let seen = &mut by_language[w.language];
        let overlap = match w.side {
            Side::Training => &avoid.training,
            Side::HeldOut => &avoid.held_out,
        };
        let mut candidates: Vec<(usize, String, &str)> = Vec::new();
        let mut in_file: HashSet<&str> = HashSet::new();
        for row in 0..table.rows() {
            let text = texts.at(row).map_err(read_err)?;
            if text.chars().take(split.min_chars).count() < split.min_chars {
                choice.too_short += 1;
                continue;
            }
            let h = host(urls.get(row).unwrap_or(""));
            if excluded_host(&h) {
                choice.excluded_host += 1;
                continue;
            }
            let win = window(text, split.window);
            let other_side = w.side == Side::HeldOut && training.contains(win);
            if other_side || seen.contains(win) || in_file.contains(win) {
                choice.duplicates += 1;
                continue;
            }
            if overlap.found(w.language, win) {
                choice.tatoeba_overlap += 1;
                continue;
            }
            in_file.insert(win);
            candidates.push((row, h, win));
        }
        let limit = split.host_limit(w.documents);
        let mut per_host: HashMap<&str, usize> = HashMap::new();
        let mut taken: Vec<usize> = Vec::new();
        let mut deferred: Vec<usize> = Vec::new();
        for (k, (_, h, _)) in candidates.iter().enumerate() {
            if taken.len() == w.documents {
                break;
            }
            let n = per_host.entry(h.as_str()).or_insert(0);
            if *n < limit {
                *n += 1;
                taken.push(k);
            } else {
                deferred.push(k);
            }
        }
        choice.host_deferred = deferred.len();
        for k in deferred {
            if taken.len() == w.documents {
                break;
            }
            taken.push(k);
            choice.host_taken_later += 1;
        }
        taken.sort_unstable();
        choice.chosen = taken.len();
        let file = index[&(f.repo.as_str(), f.path.as_str())];
        for k in taken {
            let (row, _, win) = candidates[k];
            let id = ids.at(row).map_err(read_err)?;
            seen.insert(win.to_owned());
            if w.side == Side::Training {
                training.insert(win.to_owned());
            }
            let sample =
                WebSample { language: w.language, file, row: row as u64, id: id.to_owned(), window: win.to_owned() };
            match w.side {
                Side::Training => plan.training.push(sample),
                Side::HeldOut => plan.held_out.push(sample),
            }
        }
        plan.files.push(choice);
    }
    Ok(plan)
}

/// The contents of the list of web documents trained on: for each, its
/// dataset's repository, its file, its row and its `id`, tab separated, a
/// line each in that order, every line ending in a line feed.
pub fn documents_file(plan: &WebPlan, files: &[WebFile]) -> Vec<u8> {
    let rows: BTreeSet<(&str, &str, u64, &str)> = plan
        .training
        .iter()
        .map(|s| (files[s.file].repo.as_str(), files[s.file].path.as_str(), s.row, s.id.as_str()))
        .collect();
    let mut out = String::new();
    for (repo, path, row, id) in rows {
        out.push_str(&format!("{repo}\t{path}\t{row}\t{id}\n"));
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::langdetect::LANGUAGES;

    #[test]
    fn the_web_languages_are_the_recipe_s_in_its_order() {
        let codes: Vec<&str> = WEB_LANGUAGES.iter().map(|l| l.tatoeba).collect();
        let recipe: Vec<&str> = LANGUAGES.iter().map(|p| p.tatoeba).collect();
        assert_eq!(codes, recipe);
        let mut configs = HashSet::new();
        for l in &WEB_LANGUAGES {
            assert!(!l.sources.is_empty(), "{}", l.tatoeba);
            for s in l.sources {
                assert!(configs.insert(format!("{s:?}")), "{} repeats {s:?}", l.tatoeba);
                if let Source::FineWeb2(c) = s {
                    let (code, script) = c.split_once('_').unwrap();
                    assert!(code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()), "{c}");
                    assert!(script.len() == 4 && script.starts_with(|c: char| c.is_ascii_uppercase()), "{c}");
                    assert!(!script.ends_with("_removed"), "{c}");
                }
            }
        }
        assert_eq!(WEB_LANGUAGES.iter().filter(|l| l.sources == [Source::FineWeb]).count(), 1);
    }

    #[test]
    fn documents_are_shared_among_sources() {
        let wants = WEB.wants();
        assert_eq!(wants.len(), 2 * 102);
        let srp: Vec<usize> =
            wants.iter().filter(|w| WEB_LANGUAGES[w.language].tatoeba == "srp").map(|w| w.documents).collect();
        assert_eq!(srp, [500, 500, 250, 250]);
        let odd = WebRecipe { split: WebSplit { training: 7, held_out: 3, ..WEB_SPLIT }, ..WEB };
        let uzb: Vec<(Side, usize)> = odd
            .wants()
            .iter()
            .filter(|w| WEB_LANGUAGES[w.language].tatoeba == "uzb")
            .map(|w| (w.side, w.documents))
            .collect();
        assert_eq!(uzb, [(Side::Training, 4), (Side::Training, 3), (Side::HeldOut, 2), (Side::HeldOut, 1)]);
        // Every training want comes before every held-out one.
        let first_held_out = wants.iter().position(|w| w.side == Side::HeldOut).unwrap();
        assert!(wants[first_held_out..].iter().all(|w| w.side == Side::HeldOut));
        assert_eq!(WEB_SPLIT.host_limit(1_000), 50);
        assert_eq!(WEB_SPLIT.host_limit(250), 12);
        assert_eq!(WEB_SPLIT.host_limit(3), 1);
    }

    #[test]
    fn hosts_and_windows() {
        assert_eq!(host("https://www.Example.org/a/b?c#d"), "example.org");
        assert_eq!(host("http://user:pw@de.wikipedia.org:8080/wiki"), "de.wikipedia.org");
        assert_eq!(host("http://[2001:db8::1]:80/x"), "[2001:db8::1]");
        assert_eq!(host("http://[2001:db8::1]/x"), "[2001:db8::1]");
        assert_eq!(host("example.com/path"), "example.com");
        assert_eq!(host("//Example.com./x"), "example.com");
        assert_eq!(host("https://ÜBER.example/"), "über.example");
        assert_eq!(host(""), "");
        assert!(excluded_host("tatoeba.org") && excluded_host("en.tatoeba.org"));
        assert!(!excluded_host("nottatoeba.org") && !excluded_host("tatoeba.org.example"));
        assert_eq!(window("héllo wörld", 5), "héllo");
        assert_eq!(window("short", 300), "short");
        assert_eq!(window("日本語のテキスト", 3), "日本語");
    }

    #[test]
    fn overlaps_are_found_byte_for_byte() {
        let o =
            Overlap::new(2, 16, [(0, "Ein Satz über München."), (0, "zu kurz"), (1, "another language's sentence")]);
        assert!(o.found(0, "Er sagte: Ein Satz über München. Und mehr."));
        assert!(o.found(0, "Ein Satz über München."));
        assert!(!o.found(0, "Ein Satz über München"));
        assert!(!o.found(0, "ein satz über münchen."));
        // Too short to be looked for.
        assert!(!o.found(0, "zu kurz, und noch etwas mehr"));
        assert!(!o.found(0, "another language's sentence"));
        assert!(o.found(1, "and another language's sentence"));
        assert!(!Overlap::none(2).found(0, "Ein Satz über München."));
    }

    #[test]
    fn lists_are_read_strictly() {
        let sha = "a".repeat(64);
        let line =
            format!("{}\t{}\tdata/afr_Latn/train/000_00000.parquet\t100\t{sha}\n", FINEWEB_2.repo, FINEWEB_2.revision);
        let files = parse_files("f", &format!("# header\n\n{line}")).unwrap();
        assert_eq!(files[0].size, 100);
        assert_eq!(parse_files("f", &write_files(&files)).unwrap(), files);
        let e = parse_files("f", &format!("{line}{line}")).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("f:2: {} data/afr_Latn/train/000_00000.parquet is listed twice", FINEWEB_2.repo)
        );
        let e = parse_files("f", &line.replace(FINEWEB_2.revision, "main")).unwrap_err();
        assert!(e.to_string().starts_with("f:1: HuggingFaceFW/fineweb-2 at main, not the recipe's revision"), "{e}");
        let e = parse_files("f", &line.replace("\t100\t", "\t0100\t")).unwrap_err();
        assert_eq!(e.to_string(), "f:1: the size \"0100\" is not a number");
        let e = parse_files("f", &line.replace(&sha, &"A".repeat(64))).unwrap_err();
        assert!(e.to_string().contains("not a SHA-256 in lowercase hex"), "{e}");
        let e = parse_files("f", "a\tb\n").unwrap_err();
        assert_eq!(e.to_string(), "f:1: 2 fields, not 5");

        let r = format!("{}\tdata/x.parquet\t10\t20\t{sha}\n", FINEWEB.repo);
        let ranges = parse_ranges("r", &r).unwrap();
        assert_eq!(ranges[0].span, Span { start: 10, end: 20 });
        assert_eq!(parse_ranges("r", &write_ranges(&ranges)).unwrap(), ranges);
        let e = parse_ranges("r", &r.replace("\t20\t", "\t10\t")).unwrap_err();
        assert_eq!(e.to_string(), "r:1: the range 10-10 is empty");
        let e = parse_ranges("r", &format!("{r}{r}")).unwrap_err();
        assert!(e.to_string().ends_with("is listed twice"), "{e}");
    }

    #[test]
    fn listed_ranges_must_be_the_ones_made() {
        let r = |start, end| WebRange {
            repo: FINEWEB.repo.to_owned(),
            path: "p".to_owned(),
            span: Span { start, end },
            sha256: "0".repeat(64),
        };
        assert_eq!(check_ranges(&[r(0, 4), r(4, 8)], &[r(0, 4), r(4, 8)]), Ok(()));
        let e = check_ranges(&[r(0, 4)], &[r(0, 4), r(4, 8)]).unwrap_err();
        assert_eq!(e.to_string(), "fineweb-ranges.txt: 1 ranges are listed, and the files give 2");
        let zeros = "0".repeat(64);
        let e = check_ranges(&[r(0, 5)], &[r(0, 4)]).unwrap_err();
        let want = format!(
            "range 1 is HuggingFaceFW/fineweb p 0-5 {zeros}, the files give HuggingFaceFW/fineweb p 0-4 {zeros}"
        );
        assert!(e.to_string().contains(&want), "{e}");
        // A range that differs only in its digest names both.
        let other = WebRange { sha256: "1".repeat(64), ..r(0, 4) };
        let e = check_ranges(&[other], &[r(0, 4)]).unwrap_err();
        assert!(e.to_string().contains(&format!("p 0-4 {}, the files give", "1".repeat(64))), "{e}");
    }

    #[test]
    fn the_committed_lists_are_read() {
        let files = parse_files(FILES_NAME, FILES).unwrap();
        let ranges = parse_ranges(RANGES_NAME, RANGES).unwrap();
        // The files are exactly the ones the recipe reads, and each has
        // its three ranges.
        assert_eq!((files.len(), ranges.len()), (204, 612));
        files_of(&WEB.wants(), &files).unwrap();
    }
}
