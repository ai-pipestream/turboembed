//! Adversarial and regression tests for `GrammarChecker`, `GrammarDiagnostics`,
//! view projections across deleted spans, and replacement limits.

use turbo_analysis::export::{export, import};
use turbo_analysis::{Document, Ix, Key, Layer, Provenance, Span, Spans, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    Dependencies, GrammarCategory, GrammarChecker, GrammarDiagnostic, GrammarDiagnostics, Lemmas,
    MAX_DIAGNOSTIC_REPLACEMENTS, MorphFeatures, PosTags, Sentences, Strs, Tokens,
};

fn build_doc(text: &str, tagged_tokens: &str) -> Document {
    let mut doc = Document::new(text);
    let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
    sentences.spans.push(Span::new(0, text.len() as u32));

    let parsed: Vec<(&str, &str, Option<&str>)> = tagged_tokens
        .split_whitespace()
        .map(|tok| {
            let parts: Vec<&str> = tok.split('/').collect();
            assert!(parts.len() >= 2, "invalid tagged token: {tok}");
            (parts[0], parts[1], parts.get(2).copied())
        })
        .collect();

    let mut unique_pos: Vec<&str> = parsed.iter().map(|&(_, pos, _)| pos).collect();
    unique_pos.sort_unstable();
    unique_pos.dedup();
    let tagset = TagSet::new("penn", &unique_pos);

    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sentence = Vec::new();
    let mut tags = Vec::new();
    let mut lemmas = Strs::default();
    let mut has_lemmas = false;

    let mut cursor = 0usize;
    for &(surf, pos, lemma) in &parsed {
        let rel =
            text[cursor..].find(surf).unwrap_or_else(|| panic!("token {surf:?} not found after {cursor} in {text:?}"));
        let start = cursor + rel;
        let end = start + surf.len();
        cursor = end;
        tok_spans.push(Span::new(start as u32, end as u32));
        tok_sentence.push(Ix::new(0));
        tags.push(tagset.tag(pos).unwrap());
        if let Some(l) = lemma {
            has_lemmas = true;
            lemmas.push(l);
        } else {
            lemmas.push(&surf.to_ascii_lowercase());
        }
    }

    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test:sent")).unwrap();
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: tok_spans, sentence: Some(tok_sentence) },
        Provenance::by("test:tok"),
    )
    .unwrap();
    doc.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob: None }, Provenance::by("test:pos")).unwrap();
    if has_lemmas {
        doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma: lemmas }, Provenance::by("test:lemma")).unwrap();
    }
    doc
}

#[test]
fn common_correct_sentences_are_not_rewritten() {
    let cases = [
        ("I think he is ready.", "I/PRP think/VBP he/PRP is/VBZ ready/JJ ./."),
        ("The cats sleep and the dog barks.", "The/DT cats/NNS sleep/VBP and/CC the/DT dog/NN barks/VBZ ./."),
        ("If I were rich, I would travel.", "If/IN I/PRP were/VBD rich/JJ ,/, I/PRP would/MD travel/VB ./."),
        ("The boys kiss her.", "The/DT boys/NNS kiss/VBP her/PRP ./."),
        ("An urn is here.", "An/DT urn/NN is/VBZ here/RB ./."),
        ("She does dishes.", "She/PRP does/VBZ dishes/NNS ./."),
        ("She wants a can opener.", "She/PRP wants/VBZ a/DT can/NN opener/NN ./."),
        ("The dog that I saw is sleeping.", "The/DT dog/NN that/WDT I/PRP saw/VBD is/VBZ sleeping/VBG ./."),
        (
            "According to published reports, sales rose.",
            "According/VBG to/TO published/VBN reports/NNS ,/, sales/NNS rose/VBD ./.",
        ),
        (
            "According to published reports, sales rose.",
            "According/VBG to/IN published/VBN reports/NNS ,/, sales/NNS rose/VBD ./.",
        ),
        ("He objected to proposed changes.", "He/PRP objected/VBD to/TO proposed/VBN changes/NNS ./."),
        ("We need much data.", "We/PRP need/VBP much/JJ data/NNS ./."),
        ("This data is clean.", "This/DT data/NN is/VBZ clean/JJ ./."),
        ("These data are clean.", "These/DT data/NNS are/VBP clean/JJ ./."),
        ("He is taller than I.", "He/PRP is/VBZ taller/JJR than/IN I/PRP ./."),
        (
            "We queried a SQL database and an SQL database.",
            "We/PRP queried/VBD a/DT SQL/NNP database/NN and/CC an/DT SQL/NNP database/NN ./.",
        ),
        ("Either Alice or Bob is ready.", "Either/CC Alice/NNP or/CC Bob/NNP is/VBZ ready/JJ ./."),
        ("Neither Alice nor Bob is ready.", "Neither/CC Alice/NNP nor/CC Bob/NNP is/VBZ ready/JJ ./."),
    ];
    let mut failures = Vec::new();
    for (text, tags) in cases {
        let d = GrammarChecker::new().check_document(&build_doc(text, tags)).unwrap();
        if !d.is_empty() {
            failures.push(format!("{text}: {:?}", d.diagnostics));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn morph_features_mood_sub_suppresses_irrealis_were_rewrite() {
    let mut doc = build_doc("I were rich.", "I/PRP were/VBD rich/JJ ./,");
    let mut feats = Strs::default();
    feats.push("Case=Nom|Number=Sing|Person=1");
    feats.push("Mood=Sub|Number=Sing|Person=1|Tense=Past|VerbForm=Fin");
    feats.push("Degree=Pos");
    feats.push("_");
    doc.add(&Key::<MorphFeatures<Tokens>>::DEFAULT, MorphFeatures::new(feats), Provenance::by("test:morph")).unwrap();
    let d = GrammarChecker::new().check_document(&doc).unwrap();
    assert!(d.is_empty(), "Mood=Sub should suppress SVA-001 on irrealis were: {d:?}");
}

#[test]
fn missing_infinitive_and_regular_verb_repairs() {
    let cases = [
        ("He wants to goes.", "He/PRP wants/VBZ to/TO goes/VBZ/go ./.", "AUX-001", "go"),
        ("He can danced.", "He/PRP can/MD danced/VBD ./.", "AUX-001", "dance"),
        ("He can swimming.", "He/PRP can/MD swimming/VBG ./.", "AUX-001", "swim"),
        ("They can smiling.", "They/PRP can/MD smiling/VBG ./.", "AUX-001", "smile"),
        ("They realizes the risk.", "They/PRP realizes/VBZ the/DT risk/NN ./.", "SVA-001", "realize"),
        ("They organizes the files.", "They/PRP organizes/VBZ the/DT files/NNS ./.", "SVA-001", "organize"),
        ("They caches the response.", "They/PRP caches/VBZ the/DT response/NN ./.", "SVA-001", "cache"),
        ("They focuses on it.", "They/PRP focuses/VBZ on/IN it/PRP ./.", "SVA-001", "focus"),
        ("They biases it.", "They/PRP biases/VBZ it/PRP ./.", "SVA-001", "bias"),
        ("She will focused on the task.", "She/PRP will/MD focused/VBD on/IN the/DT task/NN ./.", "AUX-001", "focus"),
        ("She will monitored the logs.", "She/PRP will/MD monitored/VBD the/DT logs/NNS ./.", "AUX-001", "monitor"),
        ("She will honored the request.", "She/PRP will/MD honored/VBD the/DT request/NN ./.", "AUX-001", "honor"),
        ("He can compared it.", "He/PRP can/MD compared/VBD it/PRP ./.", "AUX-001", "compare"),
        ("He can prepared it.", "He/PRP can/MD prepared/VBD it/PRP ./.", "AUX-001", "prepare"),
        ("He can declared it.", "He/PRP can/MD declared/VBD it/PRP ./.", "AUX-001", "declare"),
        ("He can welcomed it.", "He/PRP can/MD welcomed/VBD it/PRP ./.", "AUX-001", "welcome"),
        ("He can refined it.", "He/PRP can/MD refined/VBD it/PRP ./.", "AUX-001", "refine"),
        ("He can required it.", "He/PRP can/MD required/VBD it/PRP ./.", "AUX-001", "require"),
        ("He can buzzed.", "He/PRP can/MD buzzed/VBD ./.", "AUX-001", "buzz"),
        ("She took many photo.", "She/PRP took/VBD many/JJ photo/NN ./.", "DET-001", "photos"),
        ("Many datum exist.", "Many/JJ datum/NN exist/VBP ./.", "DET-001", "data"),
        ("Either Alice or Bob are ready.", "Either/CC Alice/NNP or/CC Bob/NNP are/VBP ready/JJ ./.", "SVA-001", "is"),
    ];
    let mut failures = Vec::new();
    for (text, tags, rule, repair) in cases {
        let d = GrammarChecker::new().check_document(&build_doc(text, tags)).unwrap();
        if !d.iter().any(|x| x.rule_id == rule && x.replacements.iter().any(|r| r.as_ref() == repair)) {
            failures.push(format!("{text}: expected {rule} -> {repair}, got {:?}", d.diagnostics));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn disjunctive_or_nor_dependencies_respect_proximity_agreement() {
    let mut doc = build_doc("Either Alice or Bob is ready.", "Either/CC Alice/NNP or/CC Bob/NNP is/VBZ ready/JJ ./.");
    let relations = TagSet::new("ud", &["cc:preconj", "nsubj", "cc", "conj", "cop", "root", "punct"]);
    let d = Dependencies {
        head: vec![
            Some(Ix::new(1)),
            Some(Ix::new(5)),
            Some(Ix::new(3)),
            Some(Ix::new(1)),
            Some(Ix::new(5)),
            None,
            Some(Ix::new(5)),
        ],
        relation: ["cc:preconj", "nsubj", "cc", "conj", "cop", "root", "punct"]
            .iter()
            .map(|x| relations.tag(x).unwrap())
            .collect(),
        relations: relations.clone(),
    };
    doc.add(&Key::<Dependencies>::DEFAULT, d, Provenance::by("review:or_dep")).unwrap();
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    assert!(diags.is_empty(), "Either Alice or Bob is ready should not fire SVA-001: {diags:?}");

    let mut bad_doc =
        build_doc("Either Alice or Bob are ready.", "Either/CC Alice/NNP or/CC Bob/NNP are/VBP ready/JJ ./.");
    let bad_deps = Dependencies {
        head: vec![
            Some(Ix::new(1)),
            Some(Ix::new(5)),
            Some(Ix::new(3)),
            Some(Ix::new(1)),
            Some(Ix::new(5)),
            None,
            Some(Ix::new(5)),
        ],
        relation: ["cc:preconj", "nsubj", "cc", "conj", "cop", "root", "punct"]
            .iter()
            .map(|x| relations.tag(x).unwrap())
            .collect(),
        relations,
    };
    bad_doc.add(&Key::<Dependencies>::DEFAULT, bad_deps, Provenance::by("review:or_dep")).unwrap();
    let bad_diags = GrammarChecker::new().check_document(&bad_doc).unwrap();
    assert!(
        bad_diags.iter().any(|x| x.rule_id == "SVA-001" && x.replacements[0].as_ref() == "is"),
        "expected SVA-001 -> is on 'Either Alice or Bob are ready.': {bad_diags:?}"
    );
}

#[test]
fn overlapping_valid_tokens_return_error_without_panicking() {
    let doc = Document::new("a");
    let view = doc.get::<View>().unwrap();
    let sentences = Sentences { spans: spans_from(ViewId::SOURCE, [Span::new(0, 1)]) };
    let tokens = Tokens { spans: spans_from(ViewId::SOURCE, [Span::new(0, 1), Span::new(0, 1)]), sentence: None };
    tokens.check_contiguous_bounds_in(view).unwrap();
    let tagset = TagSet::new("penn", &["DT"]);
    let pos = PosTags { tags: vec![tagset.tag("DT").unwrap(); 2], tagset, prob: None };
    let result = GrammarChecker::new().check_view(view, &sentences, &tokens, &pos, None, None);
    assert!(result.is_err(), "ordered disjoint tokens required by gap slicing");
}

#[test]
fn supplied_subject_dependency_after_inverted_verb_is_respected() {
    let mut doc = build_doc("Is they ready?", "Is/VBZ they/PRP ready/JJ ?/.");
    let relations = TagSet::new("ud", &["cop", "nsubj", "root", "punct"]);
    let d = Dependencies {
        head: vec![Some(Ix::new(2)), Some(Ix::new(2)), None, Some(Ix::new(2))],
        relation: ["cop", "nsubj", "root", "punct"].iter().map(|x| relations.tag(x).unwrap()).collect(),
        relations,
    };
    doc.add(&Key::<Dependencies>::DEFAULT, d, Provenance::by("review:dependency")).unwrap();
    let diagnostics = GrammarChecker::new().check_document(&doc).unwrap();
    assert!(
        diagnostics.iter().any(|d| d.rule_id == "SVA-001" && d.replacements[0].as_ref() == "Are"),
        "{diagnostics:?}"
    );
}

#[test]
fn diagnostic_can_project_across_removed_text() {
    let mut doc = Document::new("the REMOVED the");
    let src = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(src.text());
    vb.keep(4);
    vb.replace(12, "");
    let (text, align) = vb.finish();
    doc.add(&Key::<View>::named("edited"), View::derived(&src, text, align), Provenance::by("review:edit")).unwrap();
    let id = doc.get_key(&Key::<View>::named("edited")).unwrap().id();
    let mut diagnostics = GrammarDiagnostics::new(id);
    diagnostics.push(GrammarDiagnostic::new(
        Span::new(0, 7),
        "REP-001",
        GrammarCategory::RepeatedToken,
        "Repeated token",
        ["the"],
        0.94,
    ));
    diagnostics.check().unwrap();
    let projected = doc.project_layer(&diagnostics, ViewId::SOURCE);
    assert!(projected.is_ok(), "{projected:?}");
}

fn spans_from<L>(view: ViewId, items: impl IntoIterator<Item = Span>) -> Spans<L> {
    let mut spans = Spans::new(view);
    for item in items {
        spans.push(item);
    }
    spans
}

#[test]
fn diagnostic_can_project_to_a_view_that_deleted_its_text() {
    let mut doc = Document::new("bad good");
    let src = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(src.text());
    vb.replace(4, "");
    let (text, align) = vb.finish();
    doc.add(&Key::<View>::named("edited"), View::derived(&src, text, align), Provenance::by("review:edit")).unwrap();
    let id = doc.get_key(&Key::<View>::named("edited")).unwrap().id();
    let mut diagnostics = GrammarDiagnostics::new(ViewId::SOURCE);
    diagnostics.push(GrammarDiagnostic::new(
        Span::new(0, 3),
        "REP-001",
        GrammarCategory::RepeatedToken,
        "Repeated token",
        ["the"],
        0.94,
    ));
    let projected = doc.project_layer(&diagnostics, id);
    assert!(projected.is_ok(), "{projected:?}");
}

#[test]
fn every_accepted_diagnostic_can_roundtrip_its_replacements() {
    let mut doc = Document::new("bad");
    let mut diagnostics = GrammarDiagnostics::new(ViewId::SOURCE);
    diagnostics.push(GrammarDiagnostic::new(
        Span::new(0, 3),
        "REP-001",
        GrammarCategory::RepeatedToken,
        "Repeated token",
        vec!["good"; 65],
        0.94,
    ));
    doc.add(&Key::<GrammarDiagnostics>::DEFAULT, diagnostics, Provenance::by("review:wire")).unwrap();
    let registry = codecs();
    let graph = export(&doc, &registry).unwrap();
    let restored = import(&graph, &registry);
    assert!(restored.is_ok(), "{}", restored.err().unwrap());

    let over_limit = GrammarDiagnostic::new(
        Span::new(0, 3),
        "REP-001",
        GrammarCategory::RepeatedToken,
        "Repeated token",
        vec!["good"; MAX_DIAGNOSTIC_REPLACEMENTS + 1],
        0.94,
    );
    assert!(over_limit.check().is_err());
}

#[test]
fn invalid_pos_tags_are_rejected_without_panic() {
    let doc = build_doc("The dog runs", "The/DT dog/NN runs/VBZ");
    let view = doc.get::<View>().unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let pos = PosTags { tagset: TagSet::new("penn", &["NN"]), tags: vec![turbo_analysis::Tag(10); 3], prob: None };
    assert!(GrammarChecker::new().check_view(view, sentences, tokens, &pos, None, None).is_err());
}
