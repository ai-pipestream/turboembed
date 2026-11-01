//! Recipes for the Rust trainer (#1518): the corpora heads train on,
//! pinned by digest; the majority baseline and the scoring every run
//! reports; the receipt a run writes; and the `turbo-train-head` command
//! that ties them to `turbo-trainer` and the encoder.
//!
//! | Module | What it holds |
//! |---|---|
//! | `corpus` | Sentences, splits, file pins and the corpus record |
//! | `masc_pos` | MASC 3.0.0 Penn tags on the recipes' recorded release and split |
//! | `multiconer` | MultiCoNER v2 English, pinned file by file, and its coarse groups |
//! | `ud` | Universal Dependencies treebanks as syntactic words, one tag corpus a column |
//! | `ud_heads` | Lemma and dependency heads on a treebank, their examples and baselines |
//! | `ud_split` | A treebank as surface tokens, for the token split head, its labels and baseline |
//! | `split_labels` | The label each training token's split takes, chosen by the tokens it serves |
//! | `redocred` | Re-DocRED documents, their entities and relations, pinned file by file |
//! | `redocred_eval` | The relation head's spec, examples and test scoring on Re-DocRED |
//! | `peer` | A peer system's tags on a test split, exported and read back |
//! | `eval` | Accuracy, span F1, the majority baseline, the paired bootstrap |
//! | `ner_documents` | Strict span scores and paired bootstrap over whole documents |
//! | `run` | Head spec, encoder features, prediction, outputs |
//! | `receipt` | `receipt.json` |
//! | `safetensors` | Writing the head's F32 tensors |

pub mod cache;
pub mod cli;
pub mod corpus;
pub mod eval;
pub mod fit_cache;
pub mod germeval;
pub mod litbank;
pub mod litbank_eval;
pub mod masc_ne;
pub mod masc_pos;
pub mod multiconer;
pub mod ner_documents;
pub mod ner_peer;
pub mod peer;
pub mod receipt;
pub mod redocred;
pub mod redocred_eval;
pub mod run;
pub mod safetensors;
pub mod sentences;
pub mod split_labels;
pub mod ud;
pub mod ud_banks;
pub mod ud_banks_sa;
pub mod ud_banks_sa2;
pub mod ud_heads;
pub mod ud_split;
pub mod wnut17;

/// An error, as the message the command prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The corpora the trainer reads, by id: MASC, MultiCoNER, WNUT-17,
/// GermEval, LitBank, and a tag corpus for each column of each UD treebank.
pub fn corpora() -> Vec<String> {
    let mut all = vec![
        masc_ne::ID.to_owned(),
        masc_pos::ID.to_owned(),
        multiconer::ID.to_owned(),
        wnut17::ID.to_owned(),
        litbank::ID.to_owned(),
    ];
    all.extend([germeval::OUTER.to_owned(), germeval::INNER.to_owned()]);
    all.extend(ud::corpus_ids());
    all
}

/// Reads the corpus `id` from `dir`.
///
/// # Errors
///
/// An unknown id, or what the corpus's reader refuses.
pub fn read_corpus(id: &str, dir: &std::path::Path, unrecorded: bool) -> Result<corpus::Corpus> {
    if let Some(level) = germeval::Level::from_id(id) {
        return germeval::read(level, dir, unrecorded);
    }
    match id {
        masc_pos::ID => masc_pos::read(dir, unrecorded),
        masc_ne::ID => masc_ne::read(dir, unrecorded).map_err(|e| Error(e.to_string())),
        wnut17::ID => wnut17::read(dir, unrecorded),
        multiconer::ID => multiconer::read(dir, unrecorded),
        litbank::ID => litbank::read(dir, unrecorded),
        _ => match ud::find(id) {
            Some((bank, column)) => ud::read(&bank, column, dir, unrecorded),
            None => Err(Error(format!("unknown corpus {id:?}: one of {}", corpora().join(", ")))),
        },
    }
}

/// The name and layer of the head a corpus trains.
pub fn head(id: &str, task: corpus::Task) -> (&'static str, &'static str) {
    match id {
        germeval::OUTER => return ("ner_outer", "turboembed:entities"),
        germeval::INNER => return ("ner_inner", "turboembed:entities"),
        _ => (),
    }
    match (ud::find(id), task) {
        (Some((_, column)), _) => column.head(),
        (None, corpus::Task::Tags) => ("pos", "turboembed:pos"),
        (None, corpus::Task::Bio) => ("ner", "turboembed:entities"),
        (None, corpus::Task::Coref) => ("coref", "turboembed:coref"),
        (None, corpus::Task::Relations) => (redocred::HEAD, redocred::LAYER),
    }
}

/// The coarse groups of a corpus's types, where it has them.
pub fn coarse(id: &str) -> Option<fn(&str) -> &str> {
    (id == multiconer::ID).then_some(multiconer::coarse as fn(&str) -> &str)
}

/// The publisher-declared types of fixed-schema NER recipes, independent
/// of which labels happen to occur in an individual training split.
pub fn fixed_entity_types(id: &str) -> Option<&'static [&'static str]> {
    match id {
        wnut17::ID => Some(&wnut17::TYPES),
        masc_ne::ID => Some(&masc_ne::TYPES),
        germeval::OUTER | germeval::INNER => Some(&germeval::TYPES),
        _ => None,
    }
}
