//! Building the dictionary from the two files read: OpenNLP's
//! `HunspellDictionary` constructor.

use crate::affix::Affix;
use crate::flag::{Flag, FlagSet};
use crate::hash::FoldMap;
use crate::read::affix_file::AffixFile;
use crate::read::bytes::UnsupportedDirective;
use crate::read::word_list::WordList;
use crate::text::{hash_map_bucket, hash_map_capacity, java_hash, utf16_len};

use super::case::{CaseMapping, CaseType, case_type};
use super::{AffixIndex, Dictionary, Entry, EntryId, FlagSetId};

/// The field of the word list that lists a misspelling of its entry.
const PHONETIC_FIELD: &str = "ph:";

impl Dictionary {
    pub(crate) fn build(
        affix: AffixFile,
        list: WordList,
        unsupported: Vec<UnsupportedDirective>,
        fingerprint: u64,
    ) -> Dictionary {
        let case = CaseMapping::for_language(&affix.settings.language);
        let language = &*affix.settings.language;
        let hungarian = language == "hu" || language.starts_with("hu_");

        let mut flag_ids: FoldMap<Box<[Flag]>, FlagSetId> = FoldMap::default();
        let mut flag_sets: Vec<FlagSet> = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();
        // The words in the order of their first line, with their entries.
        let mut words: Vec<(Box<str>, Box<[EntryId]>)> = Vec::with_capacity(list.words.len());
        for (word, raw) in list.words {
            let mut ids = Vec::with_capacity(raw.len());
            for e in raw {
                let flags = *flag_ids.entry(e.flags.into_boxed_slice()).or_insert_with_key(|f| {
                    flag_sets.push(FlagSet::new(f.to_vec()));
                    FlagSetId(flag_sets.len() as u32 - 1)
                });
                ids.push(EntryId(entries.len() as u32));
                entries.push(Entry { flags, morphology: e.morphology });
            }
            words.push((word, ids.into_boxed_slice()));
        }
        drop(flag_ids);

        let mut replacements = affix.replacements;
        if affix.settings.check_compound_rep {
            for (word, ids) in &words {
                for id in ids.iter() {
                    let Some(fields) = &entries[id.0 as usize].morphology else { continue };
                    for field in fields.iter() {
                        if let Some(value) = field.strip_prefix(PHONETIC_FIELD) {
                            push_phonetic(&mut replacements, value, word);
                        }
                    }
                }
            }
        }
        replacements.index();

        let longest_spaced_form = longest_spaced_form(&words, &affix.prefixes, &affix.suffixes);
        let hidden = hidden_capitalized(&words, &entries, &flag_sets, affix.settings.forbidden_word, case);
        let longest_reading = longest_reading(&words, &hidden, &affix.prefixes, &affix.suffixes);
        let listed: FoldMap<Box<str>, Box<[EntryId]>> = words.into_iter().collect();

        Dictionary {
            listed,
            hidden,
            entries: entries.into(),
            flag_sets: flag_sets.into(),
            suffixes: AffixIndex::new(affix.suffixes, true),
            prefixes: AffixIndex::new(affix.prefixes, false),
            settings: affix.settings,
            case,
            hungarian,
            ignored: affix.ignored,
            input_conversion: affix.input_conversion,
            output_conversion: affix.output_conversion,
            replacements,
            compound_rules: affix.compound_rules.into(),
            compound_patterns: affix.compound_patterns.into(),
            word_breaks: affix.word_breaks.into(),
            unsupported: unsupported.into(),
            longest_spaced_form,
            longest_reading,
            fingerprint,
        }
    }
}

/// Adds the replacement a `ph:` field of `word` gives: `ph:wrong` replaces
/// `wrong` with the word, `ph:wrong->right` replaces `wrong` with `right`,
/// and `ph:wrong*` replaces `wrong` and the word each without their last
/// character.
fn push_phonetic(table: &mut crate::conversion::Conversion, value: &str, word: &str) {
    let (mut from, mut to) = (value, word);
    if let Some(arrow) = from.find("->") {
        to = &from[arrow + 2..];
        from = &from[..arrow];
    } else if let Some(stripped) = from.strip_suffix('*') {
        from = stripped;
        if !from.is_empty() && !to.is_empty() {
            from = drop_last_char(from);
            to = drop_last_char(to);
        }
    }
    if !from.is_empty() {
        table.push_phonetic(from, to);
    }
}

fn drop_last_char(s: &str) -> &str {
    let mut chars = s.chars();
    chars.next_back();
    chars.as_str()
}

/// The longest UTF-16 length of a word holding a space that can have a
/// reading: such a reading needs a word or affix text with a space, undoes
/// at most two prefixes and two suffixes, and is found from a case variant
/// at least half as long as the input (the sharp s variant writes two
/// letters as one). `None` when no word with a space can have a reading.
fn longest_spaced_form(words: &[(Box<str>, Box<[EntryId]>)], prefixes: &[Affix], suffixes: &[Affix]) -> Option<usize> {
    let spaced_affix = prefixes.iter().chain(suffixes).any(|a| a.add.contains(' '));
    let longest_word =
        words.iter().filter(|(w, _)| spaced_affix || w.contains(' ')).map(|(w, _)| utf16_len(w)).max()?;
    Some(readable_length(longest_word, prefixes, suffixes))
}

/// The longest UTF-16 length of any word that can have a direct or affixed
/// reading, by the bound of [`longest_spaced_form`]: a listed or hidden
/// word, at most two prefixes and two suffixes, from a case variant at
/// least half as long as the input. A longer word has no such reading, so
/// the compound checks that ask for one need not read it.
fn longest_reading(
    words: &[(Box<str>, Box<[EntryId]>)],
    hidden: &FoldMap<Box<str>, Box<[EntryId]>>,
    prefixes: &[Affix],
    suffixes: &[Affix],
) -> usize {
    let longest_word = words.iter().map(|(w, _)| w).chain(hidden.keys()).map(|w| utf16_len(w)).max().unwrap_or(0);
    readable_length(longest_word, prefixes, suffixes)
}

/// The longest input a word of `longest_word` UTF-16 units can be read
/// from: two prefixes and two suffixes of the longest affix texts added,
/// doubled for the sharp s variant.
fn readable_length(longest_word: usize, prefixes: &[Affix], suffixes: &[Affix]) -> usize {
    let longest = |rules: &[Affix]| rules.iter().map(|a| utf16_len(&a.add)).max().unwrap_or(0);
    2 * (longest_word + 2 * longest(prefixes) + 2 * longest(suffixes))
}

/// The capitalized forms of the mixed-case words and of the all-uppercase
/// words with flags, so that `EBOOKS` finds `eBook` and `ACME'S` finds
/// `ACME`. A form that is itself listed is left out, and so is a forbidden
/// entry.
///
/// OpenNLP builds this by iterating a `java.util.HashMap` of the words, so
/// where two words have the same capitalized form, the order of their
/// entries is the map's order: by bucket of the word's `String.hashCode`,
/// then by the order the words were first listed. The words are visited in
/// that order here. (A bucket the JDK turns into a tree, which takes eight
/// words colliding in one bucket of a large table, keeps another order;
/// the port does not reproduce that.)
fn hidden_capitalized(
    words: &[(Box<str>, Box<[EntryId]>)],
    entries: &[Entry],
    flag_sets: &[FlagSet],
    forbidden: Flag,
    case: CaseMapping,
) -> FoldMap<Box<str>, Box<[EntryId]>> {
    let listed: FoldMap<&str, ()> = words.iter().map(|(w, _)| (&**w, ())).collect();
    let capacity = hash_map_capacity(words.len());
    let mut order: Vec<(usize, usize)> =
        words.iter().enumerate().map(|(i, (w, _))| (hash_map_bucket(java_hash(w), capacity), i)).collect();
    order.sort_unstable();
    let mut hidden: FoldMap<Box<str>, Vec<EntryId>> = FoldMap::default();
    for (_, i) in order {
        let (word, ids) = &words[i];
        let kind = case_type(word);
        if !matches!(kind, CaseType::Mixed | CaseType::MixedInitial | CaseType::Upper) {
            continue;
        }
        let capitalized = case.upper_initial(&case.lower(word));
        if listed.contains_key(capitalized.as_str()) {
            continue;
        }
        for &id in ids.iter() {
            let flags = &flag_sets[entries[id.0 as usize].flags.0 as usize];
            if (kind == CaseType::Upper && flags.is_empty()) || flags.contains(forbidden) {
                continue;
            }
            hidden.entry(capitalized.as_str().into()).or_default().push(id);
        }
    }
    hidden.into_iter().map(|(w, ids)| (w, ids.into_boxed_slice())).collect()
}
