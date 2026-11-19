//! The function bodies that were in two files of one crate when the check
//! of `duplicates.rs` was written (#2459), each by the first place it is
//! written. Folding one into a single function removes its line here; no
//! line is ever added.

pub const KNOWN: &[&str] = &[
    "analysis/doccat/src/categorizer.rs::all_results",
    "analysis/encoding/src/detect/cjk/chinese.rs::maybe_set_as_pending",
    "analysis/formats/src/ad/names.rs::next",
    "analysis/formats/src/conll/names.rs::cover",
    "analysis/formats/src/conllu/mod.rs::block",
    "analysis/formats/src/conllu/words.rs::escape_conllu_spaces",
    "analysis/goldens-tool/src/docker.rs::current_user",
    "analysis/grammar/src/lang/en/mod.rs::output",
    "analysis/grammar/src/lang/en/time/date/named.rs::b_week_of",
    "analysis/grammar/src/lang/es/unit/lexicon.rs::entries",
    "analysis/grammar/src/lang/es/unit/rules.rs::coded",
    "analysis/grammar/src/lang/es/unit/rules.rs::compound",
    "analysis/grammar/src/lang/es/unit/rules.rs::raw",
    "analysis/grammar/src/lang/es/unit/rules.rs::speed",
    "analysis/grammar/src/lang/it/numbers.rs::and_fraction",
    "analysis/grammar/src/lang/it/numbers.rs::scale",
    "analysis/namefind/src/consensus.rs::compute",
    "analysis/train-components/src/chunk.rs::package_chunker_model",
    "analysis/train-components/src/sentdetect.rs::package_sentence_model",
    "analysis/train-recipes/src/litbank.rs::files",
    "analysis/train/src/perceptron.rs::argmax",
    "analysis/unicode/src/casemap.rs::lower",
    "analysis/unicode/src/casemap.rs::upper",
    "bench/src/postag/mod.rs::write",
    "bench/src/tag/mod.rs::slug",
    "core/src/cpu/deberta.rs::config",
    "core/src/cpu/model.rs::f16_to_f32",
    "core/src/ffi.rs::new_error",
    "core/src/levelzero.rs::device_count",
    "core/src/qa/session.rs::drop",
    "trainer/src/lemma.rs::language_row",
    "trainer/src/relation.rs::new",
];
