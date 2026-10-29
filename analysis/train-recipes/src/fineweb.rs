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

mod lists;
mod plan;
mod store;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use turbo_zoo::Licence;

pub use lists::{
    Span, WebError, WebFile, WebRange, check_ranges, files_of, lists_digest, parse_files, parse_ranges, write_files,
    write_ranges,
};
pub use plan::{FileChoice, WebPlan, WebSample, documents_file, host, plan_web, window};
pub use store::{FileRanges, Store, needed, record};

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
            if let Some(bucket) = by_language.get_mut(language)
                && text.len() >= OVERLAP_KEY
                && text.chars().nth(min_chars.saturating_sub(1)).is_some()
            {
                bucket.entry(&text.as_bytes()[..OVERLAP_KEY]).or_default().push(text);
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
        let Some(texts) = self.by_language.get(language) else {
            return false;
        };
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
                if n == 0 {
                    continue;
                }
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
        (documents as u64).saturating_mul(self.rows_per_document)
    }

    /// The most documents of one file one host gives at first.
    pub fn host_limit(&self, documents: usize) -> usize {
        documents.checked_div(self.host_share).unwrap_or(documents).max(1)
    }
}

pub(super) type Result<T> = std::result::Result<T, WebError>;
