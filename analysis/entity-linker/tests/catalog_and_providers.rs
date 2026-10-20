//! Tests for alias normalization, string similarity, `KnowledgeBase` CSV/TSV
//! loading, binary snapshot serialization, vector index integration, and
//! pluggable `KbProvider` implementations (`CatalogProvider`, `GazetteerProvider`,
//! `WordNetProvider`, and `GlossaryProvider`).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use turbo_analysis::ModelRef;
use turbo_entity_linker::{
    CatalogError, CatalogProvider, GazetteerProvider, GlossaryProvider, KbEntity, KbProvider, KnowledgeBase,
    WordNetProvider, alias_similarity, extract_context_terms, fold_alias, jaro_winkler, token_dice_similarity,
};
use turbo_geo::BundledGazetteer;
use turbo_glossary_layers::Glossary;
use turbo_vector_index::{IndexBuilder, Metric, Normalization, Space};
use turbo_wordnet::{HypernymTyper, InMemoryWordNetLexicon, LemmaKey, Synset, WordNetPos, WordNetRelation};

#[test]
fn fold_alias_and_similarity_handle_diacritics_initialisms_and_tokens() {
    assert_eq!(fold_alias("U.S.A."), "usa");
    assert_eq!(fold_alias("U.S."), "us");
    assert_eq!(fold_alias("São Paulo"), "sao paulo");
    assert_eq!(fold_alias("Apple, Inc."), "apple inc");
    assert_eq!(fold_alias("Saint-Louis"), "saint louis");
    assert_eq!(fold_alias("  München (Straße) "), "munchen strasse");
    assert_eq!(fold_alias(""), "");

    assert!((jaro_winkler("apple inc", "appel inc") - 0.95).abs() < 0.05);
    assert!((token_dice_similarity("new york city", "new york") - 0.80).abs() < 1e-6);
    assert_eq!(alias_similarity("apple", "apple"), 1.0);
    assert!(alias_similarity("abraham lincoln", "abraham lincon") > 0.92);

    let terms = extract_context_terms("The quick brown fox jumps over the lazy dog in 2026.");
    assert!(terms.contains(&"quick".to_owned()));
    assert!(terms.contains(&"brown".to_owned()));
    assert!(terms.contains(&"2026".to_owned()));
    assert!(!terms.contains(&"the".to_owned()));
    assert!(!terms.contains(&"in".to_owned()));
}

#[test]
fn knowledge_base_tsv_csv_and_snapshot_roundtrip() {
    let tsv = "\
# Sample Wikidata catalog
id\tname\taliases\ttypes\tdescription\tprior\trelated_ids
Q312\tApple Inc.\tApple:4.0|AAPL:1.5|Apple Computer:1.0\tORG|COMPANY\tAmerican multinational technology company in Cupertino making iPhone and Mac\t0.92\tQ19837|Q95
Q89\tApple\tApple:1.5|Malus domestica:1.0\tFOOD|FRUIT|PRODUCT\tEdible fruit produced by an apple tree in orchards\t0.65\t
Q91\tAbraham Lincoln\tLincoln:3.0|Honest Abe:1.0|President Lincoln:2.0\tPERSON|PRESIDENT\t16th President of the United States during the American Civil War\t0.95\tQ30
";
    let kb = KnowledgeBase::from_tsv("wikidata", tsv).expect("valid TSV catalog");
    assert_eq!(kb.provider_id(), "wikidata");
    assert_eq!(kb.len(), 3);
    assert_eq!(kb.entity_by_id("Q312").unwrap().1.name, "Apple Inc.");

    // Binary snapshot roundtrip
    let snap = kb.encode_snapshot();
    let restored = KnowledgeBase::decode_snapshot(&snap).expect("valid binary snapshot");
    assert_eq!(restored.provider_id(), kb.provider_id());
    assert_eq!(restored.entities(), kb.entities());

    // CSV parsing with quoted fields
    let csv = "\
id,name,aliases,types,description,prior,related_ids
Q312,\"Apple, Inc.\",Apple:3.0|AAPL:1.0,ORG|COMPANY,\"Technology company in Cupertino, California\",0.90,Q19837
Q90,\"Paris, France\",Paris:4.0|City of Light:1.0,GPE|CITY,\"Capital and largest city of France, home to the Louvre\",0.95,Q142
";
    let csv_kb = KnowledgeBase::from_csv("wikidata-csv", csv).expect("valid CSV catalog");
    assert_eq!(csv_kb.len(), 2);
    assert_eq!(csv_kb.entity_by_id("Q312").unwrap().1.name, "Apple, Inc.");
    assert_eq!(csv_kb.entity_by_id("Q90").unwrap().1.name, "Paris, France");
}

#[test]
fn knowledge_base_validation_and_corrupted_snapshots_are_rejected() {
    assert_eq!(KnowledgeBase::from_entities("  ", vec![]).unwrap_err(), CatalogError::BlankProviderId);
    assert_eq!(
        KnowledgeBase::from_entities("kb", vec![KbEntity::new("", "Name")]).unwrap_err(),
        CatalogError::BlankEntityId { row: 0 }
    );
    assert_eq!(
        KnowledgeBase::from_entities("kb", vec![KbEntity::new("Q1", "  ")]).unwrap_err(),
        CatalogError::BlankEntityName { id: "Q1".into() }
    );
    assert_eq!(
        KnowledgeBase::from_entities("kb", vec![KbEntity::new("Q1", "A"), KbEntity::new("Q1", "B")]).unwrap_err(),
        CatalogError::DuplicateEntityId { id: "Q1".into() }
    );
    assert!(matches!(
        KnowledgeBase::from_entities("kb", vec![KbEntity::new("Q1", "A").with_prior(1.5)]).unwrap_err(),
        CatalogError::InvalidPrior { .. }
    ));
    assert!(matches!(
        KnowledgeBase::from_entities("kb", vec![KbEntity::new("Q1", "A").with_alias("A", -1.0)]).unwrap_err(),
        CatalogError::InvalidAliasWeight { .. }
    ));

    // Corrupted snapshots
    assert!(KnowledgeBase::decode_snapshot(b"BADMAGIC").is_err());
    let kb = KnowledgeBase::from_entities("kb", vec![KbEntity::new("Q1", "Alpha")]).unwrap();
    let mut bytes = kb.encode_snapshot();
    bytes.push(0xFF);
    assert!(KnowledgeBase::decode_snapshot(&bytes).is_err());
    bytes.pop();
    bytes.truncate(bytes.len() - 3);
    assert!(KnowledgeBase::decode_snapshot(&bytes).is_err());
}

#[test]
fn catalog_provider_supports_exact_fuzzy_and_vector_candidates() {
    let entities = vec![
        KbEntity::new("Q312", "Apple Inc.")
            .with_alias("Apple", 4.0)
            .with_types(["ORG", "COMPANY"])
            .with_description("Technology company in Cupertino")
            .with_prior(0.9),
        KbEntity::new("Q89", "Apple (fruit)")
            .with_alias("Apple", 1.0)
            .with_types(["PRODUCT", "FOOD"])
            .with_description("Edible orchard fruit")
            .with_prior(0.6),
        KbEntity::new("Q95", "Alphabet Inc.")
            .with_alias("Google", 3.0)
            .with_types(["ORG", "COMPANY"])
            .with_description("Search engine and cloud company in Mountain View")
            .with_prior(0.9),
    ];

    let model = ModelRef { id: "test-embed".into(), revision: "1".into(), sha256: "00".into() };
    let space = Space::new(model, 2, Normalization::Unit);
    let mut builder = IndexBuilder::new(space.clone(), Metric::Cosine).unwrap();
    builder.push(&[1.0, 0.0], 0u32).unwrap();
    builder.push(&[0.0, 1.0], 1u32).unwrap();
    builder.push(&[0.70710677, 0.70710677], 2u32).unwrap();
    let vindex = Arc::new(builder.build());

    let kb = KnowledgeBase::from_entities("wikidata", entities).unwrap().with_vector_index(vindex).unwrap();
    let provider = CatalogProvider::new(kb);

    // Exact match on "Apple" returns both Q312 and Q89 with normalized alias priors 0.8 and 0.2
    let exact = provider.generate_candidates("Apple", None, 10, 1.0);
    assert_eq!(exact.len(), 2);
    assert_eq!(exact[0].kb_id, "Q312");
    assert!((exact[0].alias_prior - 0.8).abs() < 1e-6);
    assert_eq!(exact[1].kb_id, "Q89");
    assert!((exact[1].alias_prior - 0.2).abs() < 1e-6);

    // Fuzzy match on typo "Appel Inc" finds Q312 when fuzzy_threshold <= 0.85
    let fuzzy = provider.generate_candidates("Appel Inc", None, 5, 0.85);
    assert!(!fuzzy.is_empty());
    assert_eq!(fuzzy[0].kb_id, "Q312");

    // Vector nearest neighbor retrieval surfaces candidates even when surface form is novel
    let vec_cands =
        provider.generate_candidates("Mountain View Giant", Some((&space, &[0.70710677, 0.70710677])), 5, 1.0);
    assert!(vec_cands.iter().any(|c| c.kb_id == "Q95" && c.vector_score.is_some()));
}

#[test]
fn gazetteer_wordnet_and_glossary_providers_generate_candidates() {
    // 1. GazetteerProvider over BundledGazetteer
    let geo_provider = GazetteerProvider::new("natural-earth", BundledGazetteer::parse().unwrap());
    let paris_cands = geo_provider.generate_candidates("Paris", None, 10, 0.9);
    assert!(!paris_cands.is_empty());
    assert!(paris_cands[0].entity_types.contains(&"GPE".to_owned()));
    assert!(paris_cands.iter().any(|c| c.related_ids.contains(&"country:FR".to_owned())));
    let first_id = &paris_cands[0].kb_id;
    assert!(geo_provider.lookup_by_id(first_id).is_some());

    // 2. WordNetProvider over InMemoryWordNetLexicon + HypernymTyper
    let mut fin_rels = BTreeMap::new();
    fin_rels.insert(WordNetRelation::Hypernym, vec!["syn-org".to_owned()]);
    let syn_org =
        Synset::new("syn-org", WordNetPos::Noun, vec!["organization".into()], "an institution", BTreeMap::new())
            .unwrap();
    let syn_bank_fin = Synset::new(
        "syn-bank-fin",
        WordNetPos::Noun,
        vec!["bank".into(), "depository_financial_institution".into()],
        "a financial institution that accepts deposits and channels money into lending activities",
        fin_rels,
    )
    .unwrap();

    let mut loc_rels = BTreeMap::new();
    loc_rels.insert(WordNetRelation::Hypernym, vec!["syn-loc".to_owned()]);
    let syn_loc = Synset::new(
        "syn-loc",
        WordNetPos::Noun,
        vec!["location".into()],
        "a point or extent in space",
        BTreeMap::new(),
    )
    .unwrap();
    let syn_bank_riv = Synset::new(
        "syn-bank-riv",
        WordNetPos::Noun,
        vec!["bank".into()],
        "sloping land beside a body of water or river",
        loc_rels,
    )
    .unwrap();

    let mut synsets = HashMap::new();
    synsets.insert("syn-org".into(), syn_org);
    synsets.insert("syn-loc".into(), syn_loc);
    synsets.insert("syn-bank-fin".into(), syn_bank_fin);
    synsets.insert("syn-bank-riv".into(), syn_bank_riv);

    let mut sense_order = HashMap::new();
    sense_order.insert(LemmaKey::of("organization", WordNetPos::Noun), vec!["syn-org".into()]);
    sense_order.insert(LemmaKey::of("location", WordNetPos::Noun), vec!["syn-loc".into()]);
    sense_order.insert(LemmaKey::of("bank", WordNetPos::Noun), vec!["syn-bank-fin".into(), "syn-bank-riv".into()]);

    let lexicon = Arc::new(InMemoryWordNetLexicon::new(synsets, sense_order).unwrap());
    let typer = HypernymTyper::new(Arc::clone(&lexicon), [("organization", "ORG"), ("location", "LOC")]).unwrap();
    let wn_provider = WordNetProvider::new("wordnet", lexicon).with_hypernym_typer(typer);

    let bank_cands = wn_provider.generate_candidates("bank", None, 10, 1.0);
    assert_eq!(bank_cands.len(), 2);
    assert!(bank_cands[0].entity_types.contains(&"ORG".to_owned()));
    assert!(bank_cands[1].entity_types.contains(&"LOC".to_owned()));

    // 3. GlossaryProvider over turbo_glossary_layers::Glossary
    let mut glossary = Glossary::new();
    glossary.push("GLOSS-NLP", "Natural Language Processing").unwrap();
    glossary.push("GLOSS-NLP", "NLP").unwrap();
    let gloss_provider = GlossaryProvider::from_glossary("domain-glossary", &glossary, "CONCEPT");
    let nlp_cands = gloss_provider.generate_candidates("NLP", None, 5, 1.0);
    assert_eq!(nlp_cands.len(), 1);
    assert_eq!(nlp_cands[0].kb_id, "GLOSS-NLP");
}
