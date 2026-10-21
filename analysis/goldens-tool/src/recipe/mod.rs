//! What the goldens are made from, as constants: the container's base
//! image, the OpenNLP and OpenNLP addons commits, the models, the checked
//! in fixtures and the input sets. The container recipe
//! (`container/Dockerfile`) repeats the base image and the commits; a test
//! checks that both say the same.

/// The base image, pinned by the digest of its multi-platform index.
pub const BASE_IMAGE: &str = "eclipse-temurin:21-jdk";
pub const BASE_IMAGE_DIGEST: &str = "sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b";

/// The OpenNLP fork and the commit on its `kristian-3.x-features` branch the
/// generator runs.
pub const OPENNLP_REPO: &str = "https://github.com/ai-pipestream/opennlp";
pub const OPENNLP_COMMIT: &str = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

/// The OpenNLP addons repository and the commit on its `ready/numeric`
/// branch whose `numeric` module the generator runs (`NumberScan`). The
/// module is built against the OpenNLP fork above.
pub const ADDONS_REPO: &str = "https://github.com/ai-pipestream/opennlp-addons";
pub const ADDONS_COMMIT: &str = "33e95a4956f1535cdc074056b29246fe88b3c81a";

/// The commit of the OpenNLP fork whose code point `StringPattern` the
/// generator runs beside the one of [`OPENNLP_COMMIT`], for the sequence
/// crate's string pattern goldens. The container compiles that class and
/// the two it needs from this commit into a jar of their own.
pub const STRING_PATTERN_COMMIT: &str = "b2473ca80d65ae2ef10a24b08b1d0ae6a9231109";

/// The commit of the OpenNLP fork's `OPENNLP-1927-hunspell-compatibility`
/// branch whose Hunspell classes the generator runs for the Hunspell
/// goldens. The container compiles them into a jar of their own, as it does
/// the string pattern classes.
pub const HUNSPELL_COMMIT: &str = "fb299cf32ccf6f5dbc81b39448d2d7bea1cf66dc";

/// The commit of the OpenNLP addons fork, on its `ready/geo-place-profiles`
/// branch, whose `PlaceProfiles` the generator runs. The class is not in
/// [`ADDONS_COMMIT`]; the container compiles its source from this commit
/// into the `geo` module built from that one.
pub const PLACE_PROFILES_COMMIT: &str = "07e1f670818917d74e9a94367080f51ae6365bfa";

/// The commit of the OpenNLP addons, on their `main` branch, whose
/// liblinear add-on (`liblinear-addon`) the generator runs. Its classes are
/// in the default package; the container compiles them from this commit into
/// a jar of their own.
pub const LIBLINEAR_ADDON_COMMIT: &str = "84631d0111629110ebf094eadb6e938bbf7d738f";

/// The commit of the OpenNLP addons, on their `ready/neural-postag-bilstm`
/// branch, whose `neural-postag` module (the BiLSTM-CRF and feedforward
/// part-of-speech taggers) the generator runs. The module is not in
/// [`ADDONS_COMMIT`]; the container builds its pom and main sources from
/// this commit against the OpenNLP fork above.
pub const NEURAL_POSTAG_COMMIT: &str = "88767f8d09ffd350bd045e532bdd6f9d76cd9dd8";

/// The commit of the OpenNLP fork's `ready/agents-md` branch whose
/// `ConlluDependencySampleStream`, `DependencySample`, `DependencyTree` and
/// `DependencyArc` the generator runs for the CoNLL-U dependency goldens.
/// The container compiles them, with the branch's `ParamChecks`, into a jar
/// of their own.
pub const CONLLU_DEPENDENCY_COMMIT: &str = "2da0ee764cd7340b968a207b925cce852485cccb";

/// The commit of the OpenNLP addons, on their `ready/coref` branch, whose
/// `ConlluCorefDocumentStream` (with `CorefAnnotator` and `CorefMention`)
/// the generator runs for the CorefUD goldens. The container compiles the
/// three classes into a jar of their own.
pub const COREF_ADDONS_COMMIT: &str = "be11df610e24a9a2a10f80b7199349fe5b3594a9";

/// The tag `turbo-goldens image` gives the image it builds. It names the
/// OpenNLP and addons commits, so an image built from other commits has
/// another tag.
pub const IMAGE_TAG: &str = "turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1";

mod excerpts;
mod fixtures;
mod models;
#[cfg(test)]
mod tests;

pub use self::excerpts::{EXCERPTS, Excerpt};
pub use self::fixtures::{FIXTURES, Fixture};
pub use self::models::{
    CORPORA, Corpus, DICTIONARIES, Dictionary, DictionaryEncoding, HUNSPELL_DICTIONARIES_COMMIT, HUNSPELL_FILES,
    HunspellFile, LOCAL_MODELS, MODELS, Model, UTF8_COPY_SUFFIX,
};

/// The input sets under `analysis/goldens/data/inputs`.
pub const INPUT_SETS: [&str; 49] = [
    "en",
    "multi",
    "norm",
    "num",
    "money",
    "quantity",
    "temporal",
    "pii",
    "edge",
    "surrogate",
    "abbreviations",
    "alphanumeric",
    "sentence-fixture",
    "detokenizer",
    "subwords",
    "zh",
    "bpe",
    "ja",
    "ko",
    "train",
    "tagged",
    "formats",
    "ner",
    "ad",
    "parse",
    "parse-treebank",
    "ontonotes",
    "frenchtreebank",
    "head-rules",
    "gazetteer",
    "place-profiles",
    "langdetect",
    "sequence",
    "descriptor-errors",
    "codec",
    "tag-format",
    "namefind",
    "stem",
    "stopword",
    "hunspell",
    "glossary",
    "svm-doccat",
    "svm-doccat-train",
    "sentiment",
    "sentiment-train",
    "doccat",
    "doccat-train",
    "conllu",
    "corpora",
];

/// The arguments the driver passes to the generator after the fixed ones
/// (the inputs, models and output directories and the commit, which the
/// container's entrypoint supplies). `--dumps` makes it write a decision
/// point dump beside every golden of the [`DUMPED_ANNOTATORS`], so every
/// `generate` makes both.
pub const GENERATOR_ARGS: [&str; 1] = ["--dumps"];

/// The annotator directories the generator writes, which `generate`
/// replaces and `verify` compares. The data directory holds other things
/// (the inputs, the fixtures, the rate snapshot, a README) that neither
/// touches.
pub const ANNOTATORS: [&str; 52] = [
    "sentdetect",
    "tokenize",
    "postag",
    "lemmatize",
    "lemmatize-dictionary",
    "chunk",
    "langdetect",
    "normalize",
    "numbers",
    "money",
    "money-convert",
    "quantity",
    "temporal",
    "pii",
    "pii-rewrite",
    "detokenize",
    "subwords",
    "unigram",
    "bpe",
    "lattice",
    "train",
    "train-events",
    "train-langdetect",
    "train-liblinear",
    "formats",
    "evaluate",
    "masc",
    "ner",
    "ad",
    "parse",
    "ontonotes",
    "frenchtreebank",
    "head-rules",
    "gazetteer",
    "place-profiles",
    "string-pattern",
    "features",
    "codec",
    "beam",
    "tag-dictionary",
    "tag-format",
    "namefind",
    "stem",
    "stopword",
    "hunspell",
    "glossary",
    "svm-doccat",
    "sentiment",
    "doccat",
    "conllu",
    "corpora",
    "postag-neural",
];

/// The annotators with runs that ask a maxent model for their decisions:
/// each golden of theirs whose header names a model has a decision point
/// dump beside it. Their runs without a model (the simple, whitespace and
/// UAX #29 tokenizers and the newline detector) have none, nor do the
/// normalizers, the number scan, the money, quantity, temporal and PII
/// annotators, the PII rewrites, the detokenizer, the WordPiece encoder,
/// the unigram segmenter, the BPE trainer and the lattice tokenizer, which
/// ask no model.
pub const DUMPED_ANNOTATORS: [&str; 6] = ["sentdetect", "tokenize", "postag", "lemmatize", "chunk", "langdetect"];

/// The input set the normalizers run on. They run on no other set, and the
/// other annotators do not run on this one.
pub const NORMALIZER_INPUT_SET: &str = "norm";

/// The input set the number scan runs on. It runs on no other set, and the
/// other annotators do not run on this one.
pub const NUMBER_INPUT_SET: &str = "num";

/// The input set the money annotators run on. They run on no other set, and
/// the other annotators do not run on this one.
pub const MONEY_INPUT_SET: &str = "money";

/// The input set the quantity extractor runs on. It runs on no other set,
/// and the other annotators do not run on this one.
pub const QUANTITY_INPUT_SET: &str = "quantity";

/// The input set the temporal extractor runs on. It runs on no other set,
/// and the other annotators do not run on this one.
pub const TEMPORAL_INPUT_SET: &str = "temporal";

/// The input sets of the SVM document categorizer: the documents its models
/// categorize, and the corpora they are trained on. No other annotator runs
/// on them.
pub const SVM_DOCCAT_INPUT_SET: &str = "svm-doccat";
pub const SVM_DOCCAT_TRAIN_SET: &str = "svm-doccat-train";

/// The input sets of the sentiment detector: the documents its models
/// score, and the corpus they are trained on. No other annotator runs on
/// them.
pub const SENTIMENT_INPUT_SET: &str = "sentiment";
pub const SENTIMENT_TRAIN_SET: &str = "sentiment-train";

/// The input sets of the maxent document categorizer: the documents its
/// models categorize, and the corpus written for this project that they
/// are trained on after the downloaded one ([`CORPORA`]). No other
/// annotator runs on them.
pub const DOCCAT_INPUT_SET: &str = "doccat";
pub const DOCCAT_TRAIN_SET: &str = "doccat-train";

/// The input set the PII packs and the PII rewrites run on. They run on no
/// other set, and the other annotators do not run on this one.
/// The input set of the corpus readers: files and directories in the
/// layouts of the Moses, LetsMT, NKJP, Irish Sentence Bank, Leipzig and
/// 20 Newsgroups corpora. No other annotator runs on it.
pub const CORPORA_INPUT_SET: &str = "corpora";

/// The input sets of the OntoNotes and French Treebank streams: bundles of
/// corpus directories in their layouts, written for the goldens. Each set is
/// read by its own streams only.
pub const ONTONOTES_INPUT_SET: &str = "ontonotes";
pub const FRENCH_TREEBANK_INPUT_SET: &str = "frenchtreebank";

/// The input set the PII packs run on. They run on no other set, and the
/// other annotators do not run on this one.
pub const PII_INPUT_SET: &str = "pii";

/// The input set the Arvores Deitadas streams read: files in the format of
/// the Floresta Sinta(c)tica treebank, written for the goldens. The streams
/// read no other set.
pub const AD_INPUT_SET: &str = "ad";

/// The rate snapshot the money conversion reads: a directory beside the
/// inputs directory under `analysis/goldens/data`, which the container gets
/// at `/rates`, and the file in it.
pub const RATES_DIR: &str = "rates";
pub const RATES_FILE: &str = "ecb-snapshot.csv";
