//! Clean-room English morphological tables and inflection synthesis for
//! subject-verb agreement (`SVA-001`), determiner-noun agreement (`DET-001`),
//! modal/auxiliary government (`AUX-001`), and pronoun case (`CASE-001`).

/// Subject person and number governing finite verb agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubjectPersonNumber {
    FirstSing,
    SecondOrPlural,
    ThirdSing,
}

/// Grammatical number of a noun or nominal head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NounNumber {
    Singular,
    Plural,
    Invariant,
}

const PLURAL_SUBJECT_PRONOUNS: &[&str] =
    &["you", "we", "they", "these", "those", "both", "several", "many", "few", "fewer", "others"];

const INVARIANT_NOUNS: &[&str] =
    &["series", "species", "means", "headquarters", "sheep", "deer", "fish", "aircraft", "spacecraft", "data", "media"];

const IRREGULAR_PLURAL_MAP: &[&[(&str, &str)]] = &[
    &[("alumnus", "alumni"), ("analysis", "analyses"), ("appendix", "appendices"), ("axis", "axes")],
    &[("bacterium", "bacteria"), ("basis", "bases"), ("calf", "calves"), ("child", "children")],
    &[("crisis", "crises"), ("criterion", "criteria"), ("curriculum", "curricula"), ("datum", "data")],
    &[("diagnosis", "diagnoses"), ("foot", "feet"), ("fungus", "fungi"), ("goose", "geese")],
    &[("half", "halves"), ("hypothesis", "hypotheses"), ("index", "indices"), ("knife", "knives")],
    &[("leaf", "leaves"), ("life", "lives"), ("loaf", "loaves"), ("man", "men"), ("matrix", "matrices")],
    &[("medium", "media"), ("mouse", "mice"), ("nucleus", "nuclei"), ("oasis", "oases"), ("ox", "oxen")],
    &[("parenthesis", "parentheses"), ("person", "people"), ("phenomenon", "phenomena"), ("self", "selves")],
    &[("shelf", "shelves"), ("stimulus", "stimuli"), ("syllabus", "syllabi"), ("synthesis", "syntheses")],
    &[("thesis", "theses"), ("thief", "thieves"), ("tooth", "teeth"), ("vertex", "vertices")],
    &[("wife", "wives"), ("wolf", "wolves"), ("woman", "women")],
];

fn is_third_sing_pronoun(lower: &str) -> bool {
    matches!(lower, "he" | "she" | "it" | "this" | "that" | "each" | "either" | "neither" | "one" | "no-one")
        || ["every", "some", "any", "no"]
            .iter()
            .any(|p| lower.strip_prefix(p).is_some_and(|s| matches!(s, "one" | "body" | "thing")))
}

fn is_singular_s_noun(lower: &str) -> bool {
    lower.ends_with("ics")
        || lower.ends_with("sis")
        || lower.ends_with("ss")
        || lower.ends_with("us")
        || matches!(lower, "news" | "measles" | "diabetes" | "rabies" | "lens" | "bus" | "gas")
        || matches!(lower, "bias" | "alias" | "canvas" | "atlas" | "axis" | "chaos" | "cosmos" | "ethos" | "pathos")
}

/// Classifies a subject pronoun or noun token into [`SubjectPersonNumber`].
pub(crate) fn classify_subject_token(surface: &str, pos_tag: &str) -> Option<SubjectPersonNumber> {
    let lower = surface.to_ascii_lowercase();
    if lower == "i" {
        return Some(SubjectPersonNumber::FirstSing);
    }
    if PLURAL_SUBJECT_PRONOUNS.contains(&lower.as_str()) {
        return Some(SubjectPersonNumber::SecondOrPlural);
    }
    if is_third_sing_pronoun(&lower) {
        return Some(SubjectPersonNumber::ThirdSing);
    }
    match classify_noun_number(&lower, pos_tag)? {
        NounNumber::Singular => Some(SubjectPersonNumber::ThirdSing),
        NounNumber::Plural => Some(SubjectPersonNumber::SecondOrPlural),
        NounNumber::Invariant => None,
    }
}

/// Classifies a noun token's grammatical number using POS tag and morphological rules.
pub(crate) fn classify_noun_number(lower: &str, pos_tag: &str) -> Option<NounNumber> {
    if lower.is_empty() || !lower.chars().all(|c| c.is_alphabetic() || c == '-') {
        return None;
    }
    if INVARIANT_NOUNS.contains(&lower) {
        return Some(NounNumber::Invariant);
    }
    if is_irregular_plural_noun(lower) {
        return Some(NounNumber::Plural);
    }
    if is_singular_s_noun(lower) {
        return Some(NounNumber::Singular);
    }
    match pos_tag {
        "NNS" | "NNPS" => Some(NounNumber::Plural),
        "NN" | "NNP" => Some(NounNumber::Singular),
        "NOUN" | "PROPN" => {
            Some(if looks_plural_orthography(lower) { NounNumber::Plural } else { NounNumber::Singular })
        }
        _ => None,
    }
}

fn looks_plural_orthography(lower: &str) -> bool {
    if is_irregular_plural_noun(lower) {
        return true;
    }
    if is_singular_s_noun(lower) || lower.len() <= 2 {
        return false;
    }
    lower.ends_with('s') && !lower.ends_with("is")
}

fn is_irregular_plural_noun(lower: &str) -> bool {
    IRREGULAR_PLURAL_MAP.iter().flat_map(|g| g.iter()).any(|&(_, pl)| pl == lower)
}

/// Checks if a finite verb disagrees with `target` and returns the corrected verb surface form.
pub(crate) fn check_and_repair_sva_verb(
    verb_surface: &str,
    verb_pos: &str,
    lemma: Option<&str>,
    target: SubjectPersonNumber,
) -> Option<String> {
    let lower = verb_surface.to_ascii_lowercase();
    let fixed = match (lower.as_str(), target) {
        ("is", SubjectPersonNumber::FirstSing) | ("are", SubjectPersonNumber::FirstSing) => Some("am"),
        ("is", SubjectPersonNumber::SecondOrPlural) | ("am", SubjectPersonNumber::SecondOrPlural) => Some("are"),
        ("are", SubjectPersonNumber::ThirdSing) | ("am", SubjectPersonNumber::ThirdSing) => Some("is"),
        ("was", SubjectPersonNumber::SecondOrPlural) => Some("were"),
        ("were", SubjectPersonNumber::ThirdSing | SubjectPersonNumber::FirstSing) => Some("was"),
        ("has", SubjectPersonNumber::SecondOrPlural | SubjectPersonNumber::FirstSing) => Some("have"),
        ("have", SubjectPersonNumber::ThirdSing) => Some("has"),
        ("does", SubjectPersonNumber::SecondOrPlural | SubjectPersonNumber::FirstSing) => Some("do"),
        ("do", SubjectPersonNumber::ThirdSing) => Some("does"),
        ("doesn't", SubjectPersonNumber::SecondOrPlural | SubjectPersonNumber::FirstSing) => Some("don't"),
        ("don't", SubjectPersonNumber::ThirdSing) => Some("doesn't"),
        ("isn't", SubjectPersonNumber::SecondOrPlural) => Some("aren't"),
        ("aren't", SubjectPersonNumber::ThirdSing) => Some("isn't"),
        ("wasn't", SubjectPersonNumber::SecondOrPlural) => Some("weren't"),
        ("weren't", SubjectPersonNumber::ThirdSing | SubjectPersonNumber::FirstSing) => Some("wasn't"),
        ("hasn't", SubjectPersonNumber::SecondOrPlural | SubjectPersonNumber::FirstSing) => Some("haven't"),
        ("haven't", SubjectPersonNumber::ThirdSing) => Some("hasn't"),
        ("is" | "are" | "am" | "was" | "were" | "has" | "have" | "does" | "do", _) => return None,
        _ => None,
    };
    if let Some(rep) = fixed {
        return Some(match_case(verb_surface, rep));
    }
    if matches!(verb_pos, "MD" | "VBD" | "VBN" | "VBG") || is_modal_or_past_verb(&lower) {
        return None;
    }

    let is_3sg_form = verb_pos == "VBZ" || (matches!(verb_pos, "VERB" | "AUX") && looks_3sg_verb(&lower));
    let is_base_present_form = verb_pos == "VBP"
        || (verb_pos == "VERB" && !looks_3sg_verb(&lower) && !lower.ends_with("ed") && !lower.ends_with("ing"));

    match target {
        SubjectPersonNumber::SecondOrPlural | SubjectPersonNumber::FirstSing if is_3sg_form => {
            let base = lemma
                .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case(&lower))
                .map(str::to_owned)
                .unwrap_or_else(|| deinflect_3sg_verb(&lower));
            (!base.eq_ignore_ascii_case(&lower)).then(|| match_case(verb_surface, &base))
        }
        SubjectPersonNumber::ThirdSing if is_base_present_form => {
            let inflected = inflect_3sg_verb(&lower);
            (!inflected.eq_ignore_ascii_case(&lower)).then(|| match_case(verb_surface, &inflected))
        }
        _ => None,
    }
}

fn is_modal_or_past_verb(lower: &str) -> bool {
    matches!(lower, "can" | "could" | "may" | "might" | "must" | "shall" | "should" | "will" | "would")
        || (!lower.ends_with('s') && irregular_verb_base(lower).is_some())
}

fn looks_3sg_verb(lower: &str) -> bool {
    lower.len() > 2 && lower.ends_with('s') && !lower.ends_with("ss") && !lower.ends_with("us")
}

pub(crate) fn deinflect_3sg_verb(lower: &str) -> String {
    match lower {
        "has" => return "have".into(),
        "does" => return "do".into(),
        "goes" => return "go".into(),
        "quizzes" => return "quiz".into(),
        "is" | "are" | "am" => return "be".into(),
        _ => {}
    }
    if let Some(stem) = lower.strip_suffix("ies").filter(|s| !s.is_empty()) {
        return format!("{stem}y");
    }
    if !matches!(lower, "caches" | "aches" | "niches" | "avalanches") {
        for es_ending in ["shes", "ches", "sses", "xes", "zzes", "oes"] {
            if lower.ends_with(es_ending) {
                return lower[..lower.len() - 2].to_owned();
            }
        }
    }
    if let Some(stem) = lower.strip_suffix("es").filter(|s| s.ends_with("us") || s.ends_with("as"))
        && NON_SILENT_E_STEMS.iter().any(|group| group.contains(&stem))
    {
        return stem.to_owned();
    }
    if let Some(stem) = lower.strip_suffix('s').filter(|s| !s.ends_with('s')) {
        return stem.to_owned();
    }
    lower.to_owned()
}

pub(crate) fn inflect_3sg_verb(lower: &str) -> String {
    match lower {
        "have" => return "has".into(),
        "do" => return "does".into(),
        "go" => return "goes".into(),
        "quiz" => return "quizzes".into(),
        "be" => return "is".into(),
        _ => {}
    }
    if let Some(stem) = lower.strip_suffix('y')
        && stem.chars().last().is_some_and(|c| !matches!(c, 'a' | 'e' | 'i' | 'o' | 'u'))
    {
        return format!("{stem}ies");
    }
    if ["sh", "ch", "s", "x", "z", "o"].iter().any(|suf| lower.ends_with(suf)) {
        return format!("{lower}es");
    }
    format!("{lower}s")
}

/// Converts a singular English count noun to its plural form, preserving case.
pub(crate) fn pluralize_noun(surface: &str) -> String {
    let lower = surface.to_ascii_lowercase();
    if lower.ends_with('o')
        && !matches!(
            lower.as_str(),
            "potato" | "tomato" | "hero" | "echo" | "torpedo" | "veto" | "embargo" | "volcano" | "mosquito"
        )
    {
        return match_case(surface, &format!("{lower}s"));
    }
    let plural = IRREGULAR_PLURAL_MAP
        .iter()
        .flat_map(|g| g.iter())
        .find_map(|&(sg, pl)| (sg == lower).then(|| pl.to_owned()))
        .unwrap_or_else(|| inflect_3sg_verb(&lower));
    match_case(surface, &plural)
}

/// Converts a plural English count noun to its singular form, preserving case.
pub(crate) fn singularize_noun(surface: &str) -> String {
    let lower = surface.to_ascii_lowercase();
    let singular = IRREGULAR_PLURAL_MAP
        .iter()
        .flat_map(|g| g.iter())
        .find_map(|&(sg, pl)| (pl == lower).then(|| sg.to_owned()))
        .unwrap_or_else(|| deinflect_3sg_verb(&lower));
    match_case(surface, &singular)
}

/// Maps an inflected verb governed by a bare-infinitive modal to its base infinitive lemma.
pub(crate) fn inflected_verb_to_base(surface: &str, pos_tag: &str, lemma: Option<&str>) -> Option<String> {
    let lower = surface.to_ascii_lowercase();
    if let Some(lem) = lemma.filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case(&lower)) {
        return Some(match_case(surface, &lem.to_ascii_lowercase()));
    }
    if let Some(base) = irregular_verb_base(&lower) {
        return Some(match_case(surface, base));
    }
    if matches!(pos_tag, "VBZ" | "VBD" | "VBN" | "VBG") {
        if let Some(stem) = lower.strip_suffix("ied") {
            let valid = count_vowel_groups(stem) >= 1
                && !matches!(stem.as_bytes().last(), Some(b'a' | b'e' | b'i' | b'o' | b'u' | b'y'));
            return valid.then(|| match_case(surface, &format!("{stem}y")));
        }
        if matches!(lower.as_str(), "dying" | "lying" | "tying") {
            return Some(match_case(surface, &format!("{}ie", &lower[..lower.len() - 4])));
        }
        if let Some(stem) = lower.strip_suffix("ed").filter(|s| s.len() >= 2) {
            return restore_regular_verb_stem(stem).map(|base| match_case(surface, &base));
        }
        if let Some(stem) = lower.strip_suffix("ing").filter(|s| s.len() >= 2) {
            return restore_regular_verb_stem(stem).map(|base| match_case(surface, &base));
        }
        if pos_tag == "VBZ" || looks_3sg_verb(&lower) {
            let base = deinflect_3sg_verb(&lower);
            if base != lower {
                return Some(match_case(surface, &base));
            }
        }
    }
    None
}

fn count_vowel_groups(stem: &str) -> usize {
    let bytes = stem.as_bytes();
    let n = bytes.len();
    let is_vowel = |b: u8| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u');
    let is_stem_vowel =
        |k: usize| is_vowel(bytes[k]) || (bytes[k] == b'y' && k > 0 && (k + 1 == n || !is_vowel(bytes[k + 1])));
    (0..n).filter(|&k| is_stem_vowel(k) && (k == 0 || !is_stem_vowel(k - 1))).count()
}

fn restore_regular_verb_stem(stem: &str) -> Option<String> {
    if stem == "quizz" {
        return Some("quiz".to_owned());
    }
    if NON_SILENT_E_STEMS.iter().any(|group| group.contains(&stem)) {
        return Some(stem.to_owned());
    }
    if SILENT_E_STEMS.iter().any(|group| group.contains(&stem)) || has_unambiguous_silent_e_ending(stem) {
        return Some(format!("{stem}e"));
    }
    let vg = count_vowel_groups(stem);
    if stem.len() >= 4 && ["bb", "gg", "mm", "nn", "pp", "rr", "tt"].iter().any(|d| stem.ends_with(d)) {
        let keep = vg == 1 && !matches!(stem, "purr" | "whirr" | "butt" | "putt" | "mutt" | "watt" | "matt" | "shirr");
        return keep.then(|| stem[..stem.len() - 1].to_owned());
    }
    let bytes = stem.as_bytes();
    let n = bytes.len();
    let is_vowel = |b: u8| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u');
    if n < 3 || ["dd", "ng", "th", "ast"].iter().any(|s| stem.ends_with(s)) || is_vowel(bytes[n - 1]) {
        return None;
    }
    if bytes[n - 1] == b'y' {
        return Some(stem.to_owned());
    }
    if vg == 1 {
        let last = bytes[n - 1];
        let prev = bytes[n - 2];
        if last == b's' && prev != b's' {
            return Some(format!("{stem}e"));
        }
        let is_vce = matches!(last, b'b' | b'd' | b'f' | b'g' | b'k' | b'l' | b'm' | b'n' | b'p' | b'r' | b't' | b'z');
        let gu_or_qu = n >= 4 && matches!(bytes[n - 4], b'q' | b'g') && bytes[n - 3] == b'u';
        if is_vce && (is_vowel(prev) || prev == b'y') && (!is_vowel(bytes[n - 3]) || gu_or_qu) {
            return Some(format!("{stem}e"));
        }
        return Some(stem.to_owned());
    }
    None
}

const NON_SILENT_E_STEMS: &[&[&str]] = &[
    &["open", "iron", "edit", "exit", "omit", "emit", "echo", "arc", "sync", "zinc", "disc", "spec", "bus", "gas"],
    &["yes", "plus", "bias", "unbias", "canvas", "atlas", "alias", "focus", "monitor", "honor", "abandon", "button"],
    &["pardon", "poison", "reason", "season", "summon", "question", "mention", "function", "position", "condition"],
];

const SILENT_E_STEMS: &[&[&str]] = &[
    &["creat", "pleas", "displeas", "appeas", "teas", "releas", "increas", "decreas", "ceas", "deceas", "leas", "eas"],
    &["diseas", "greas", "creas", "purchas", "paraphras", "prais", "apprais", "rais", "cruis", "bruis", "guis"],
    &["disguis", "nois", "pois", "choos", "loos", "hous", "arous", "espous", "paus", "caus", "leav", "achiev"],
    &["believ", "receiv", "perceiv", "conceiv", "reliev", "retriev", "pars", "laps", "collaps", "elaps", "relaps"],
    &["eclips", "glimps", "rins", "sens", "licens", "condens", "dispens", "compens", "tens", "intens", "revers"],
    &["convers", "travers", "immers", "dispers", "curs", "nurs", "purs", "reimburs", "endors", "impuls", "repuls"],
    &["puls", "cleans", "invit", "ignit", "unit", "reunit", "excit", "recit", "incit", "complet", "compet", "delet"],
    &["deplet", "obsolet", "devot", "promot", "demot", "quot", "denot", "annot", "ignor", "explor", "restor"],
    &["underscor", "ador", "implor", "deplor", "abus", "accus", "amus", "confus", "diffus", "excus", "infus", "misus"],
    &["perus", "refus", "reus", "suffus", "transfus", "disus", "overus", "compar", "prepar", "declar", "welcom"],
    &["refin", "defin", "combin", "requir", "acquir", "inquir"],
];

fn has_unambiguous_silent_e_ending(stem: &str) -> bool {
    let bytes = stem.as_bytes();
    let n = bytes.len();
    if n < 2 || count_vowel_groups(stem) == 0 {
        return false;
    }
    let last = bytes[n - 1];
    let prev = bytes[n - 2];
    if matches!(last, b'c' | b'v' | b'u')
        || (last == b'z' && prev != b'z' && !matches!(stem, "quiz" | "whiz" | "fez"))
        || (last == b'g' && !matches!(prev, b'g' | b'n') && stem != "catalog")
    {
        return true;
    }
    let is_vowel = |b: u8| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u');
    last == b'l' && !is_vowel(prev) && !matches!(prev, b'l' | b'r' | b'm' | b'n' | b'w' | b'y')
}

pub(crate) fn irregular_verb_base(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "went" | "gone" | "goes" | "going" => "go",
        "ran" | "runs" | "running" => "run",
        "bought" | "buys" | "buying" => "buy",
        "brought" | "brings" | "bringing" => "bring",
        "caught" | "catches" | "catching" => "catch",
        "taught" | "teaches" | "teaching" => "teach",
        "thought" | "thinks" | "thinking" => "think",
        "fought" | "fights" | "fighting" => "fight",
        "sought" | "seeks" | "seeking" => "seek",
        "wrote" | "written" | "writes" | "writing" => "write",
        "took" | "taken" | "takes" | "taking" => "take",
        "gave" | "given" | "gives" | "giving" => "give",
        "saw" | "seen" | "sees" | "seeing" => "see",
        "came" | "comes" | "coming" => "come",
        "did" | "done" | "does" | "doing" => "do",
        "had" | "has" | "having" => "have",
        "was" | "were" | "been" | "is" | "are" | "am" | "being" => "be",
        "ate" | "eaten" | "eats" | "eating" => "eat",
        "spoke" | "spoken" | "speaks" | "speaking" => "speak",
        "broke" | "broken" | "breaks" | "breaking" => "break",
        "chose" | "chosen" | "chooses" | "choosing" => "choose",
        "drove" | "driven" | "drives" | "driving" => "drive",
        "fell" | "fallen" | "falls" | "falling" => "fall",
        "flew" | "flown" | "flies" | "flying" => "fly",
        "forgot" | "forgotten" | "forgets" | "forgetting" => "forget",
        "froze" | "frozen" | "freezes" | "freezing" => "freeze",
        "grew" | "grown" | "grows" | "growing" => "grow",
        "hid" | "hidden" | "hides" | "hiding" => "hide",
        "knew" | "known" | "knows" | "knowing" => "know",
        "began" | "begun" | "begins" | "beginning" => "begin",
        "drank" | "drunk" | "drinks" | "drinking" => "drink",
        "sang" | "sung" | "sings" | "singing" => "sing",
        "sank" | "sunk" | "sinks" | "sinking" => "sink",
        "swam" | "swum" | "swims" | "swimming" => "swim",
        "stole" | "stolen" | "steals" | "stealing" => "steal",
        "threw" | "thrown" | "throws" | "throwing" => "throw",
        "wore" | "worn" | "wears" | "wearing" => "wear",
        "rode" | "ridden" | "rides" | "riding" => "ride",
        "rose" | "risen" | "rises" | "rising" => "rise",
        "shook" | "shaken" | "shakes" | "shaking" => "shake",
        "built" | "builds" | "building" => "build",
        "felt" | "feels" | "feeling" => "feel",
        "found" | "finds" | "finding" => "find",
        "held" | "holds" | "holding" => "hold",
        "kept" | "keeps" | "keeping" => "keep",
        "left" | "leaves" | "leaving" => "leave",
        "lost" | "loses" | "losing" => "lose",
        "made" | "makes" | "making" => "make",
        "meant" | "means" | "meaning" => "mean",
        "met" | "meets" | "meeting" => "meet",
        "paid" | "pays" | "paying" => "pay",
        "said" | "says" | "saying" => "say",
        "sold" | "sells" | "selling" => "sell",
        "sent" | "sends" | "sending" => "send",
        "sat" | "sits" | "sitting" => "sit",
        "slept" | "sleeps" | "sleeping" => "sleep",
        "spent" | "spends" | "spending" => "spend",
        "stood" | "stands" | "standing" => "stand",
        "told" | "tells" | "telling" => "tell",
        "understood" | "understands" | "understanding" => "understand",
        "won" | "wins" | "winning" => "win",
        _ => return None,
    })
}

/// Maps a nominative personal pronoun to its objective form when governed by a preposition.
pub(crate) fn nominative_to_objective_pronoun(surface: &str) -> Option<String> {
    let lower = surface.to_ascii_lowercase();
    let obj = match lower.as_str() {
        "i" => return Some("me".into()),
        "he" => "him",
        "she" => "her",
        "we" => "us",
        "they" => "them",
        _ => return None,
    };
    Some(match_case(surface, obj))
}

pub(crate) fn match_case(reference: &str, replacement_lower: &str) -> String {
    let mut chars = reference.chars();
    let Some(first) = chars.next() else {
        return replacement_lower.to_owned();
    };
    if first.is_uppercase() && reference.chars().count() > 1 && reference.chars().all(|c| !c.is_lowercase()) {
        replacement_lower.to_uppercase()
    } else if first.is_uppercase() {
        let mut rep_chars = replacement_lower.chars();
        match rep_chars.next() {
            Some(r0) => r0.to_uppercase().collect::<String>() + rep_chars.as_str(),
            None => String::new(),
        }
    } else {
        replacement_lower.to_owned()
    }
}
