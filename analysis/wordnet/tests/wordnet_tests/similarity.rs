//! Synset similarity, hypernym typing, and semantic role relations.

use super::*;

// ============================================================================
// SynsetSimilarityTest & HypernymTyperTest
// ============================================================================

struct FixtureKb {
    by_id: HashMap<String, Synset>,
    by_lemma: HashMap<String, Vec<String>>,
}

impl FixtureKb {
    fn new() -> Self {
        Self { by_id: HashMap::new(), by_lemma: HashMap::new() }
    }

    fn add(&mut self, id: &str, lemma: &str, rel: WordNetRelation, parents: &[&str]) {
        let mut relations = BTreeMap::new();
        if !parents.is_empty() {
            relations.insert(rel, parents.iter().map(|s| (*s).to_string()).collect());
        }
        let synset = Synset::new(id, WordNetPos::Noun, vec![lemma.to_string()], "fixture", relations).unwrap();
        self.by_id.insert(id.to_string(), synset);
        self.by_lemma.entry(lemma.to_string()).or_default().push(id.to_string());
    }
}

impl LexicalKnowledgeBase for FixtureKb {
    fn lookup(&self, lemma: &str, _pos: WordNetPos) -> Vec<&Synset> {
        let Some(ids) = self.by_lemma.get(lemma) else {
            return Vec::new();
        };
        ids.iter().filter_map(|id| self.by_id.get(id)).collect()
    }

    fn synset(&self, synset_id: &str) -> Option<&Synset> {
        self.by_id.get(synset_id)
    }
}

fn similarity_taxonomy() -> FixtureKb {
    let mut kb = FixtureKb::new();
    kb.add("n1", "entity", WordNetRelation::Hypernym, &[]);
    kb.add("n2", "physical", WordNetRelation::Hypernym, &["n1"]);
    kb.add("n3", "organism", WordNetRelation::Hypernym, &["n2"]);
    kb.add("n4", "person", WordNetRelation::Hypernym, &["n3"]);
    kb.add("n5", "scientist", WordNetRelation::Hypernym, &["n4"]);
    kb.add("n6", "chemist", WordNetRelation::Hypernym, &["n5"]);
    kb.add("n7", "location", WordNetRelation::Hypernym, &["n2"]);
    kb.add("n8", "city", WordNetRelation::Hypernym, &["n7"]);
    kb.add("n9", "organization", WordNetRelation::Hypernym, &["n1"]);
    kb.add("n10", "company", WordNetRelation::Hypernym, &["n9"]);
    kb.add("n11", "paris", WordNetRelation::InstanceHypernym, &["n8"]);
    kb.add("n12", "abstract", WordNetRelation::Hypernym, &[]);
    kb
}

#[test]
fn test_synset_similarity_measures() {
    let sim = SynsetSimilarity::new(similarity_taxonomy());
    assert!((sim.path("n5", "n5") - 1.0).abs() < 1e-9);
    assert!((sim.path("n6", "n5") - 0.5).abs() < 1e-9);
    assert!((sim.path("n6", "n8") - 1.0 / 7.0).abs() < 1e-9);
    assert!((sim.path("n6", "n12") - 0.0).abs() < 1e-9);
    assert!((sim.path("n11", "n8") - 0.5).abs() < 1e-9);

    assert!((sim.wu_palmer("n5", "n6") - 10.0 / 11.0).abs() < 1e-9);
    let sibling = sim.wu_palmer("n6", "n8");
    assert!(sibling < sim.wu_palmer("n5", "n6") && sibling > 0.0);
    assert!((sim.wu_palmer("n6", "n12") - 0.0).abs() < 1e-9);
    assert!((sim.wu_palmer("n1", "n1") - 1.0).abs() < 1e-9);
    assert!(sim.wu_palmer("n8", "n10") > 0.0);

    assert!((sim.leacock_chodorow("n5", "n6", 10).unwrap() - 10.0_f64.ln()).abs() < 1e-9);
    assert!((sim.leacock_chodorow("n6", "n12", 10).unwrap() - 0.0).abs() < 1e-9);
    assert!((sim.leacock_chodorow("n6", "n8", 3).unwrap() - (-(7.0_f64 / 6.0).ln())).abs() < 1e-9);
    assert!(sim.leacock_chodorow("n5", "n6", 0).is_err());

    assert_eq!(None, sim.shortest_distance("n5", "missing"));
    assert_eq!(None, sim.shortest_distance("missing", "missing"));
    assert_eq!(0.0, sim.path("missing", "missing"));
    assert_eq!(0.0, sim.wu_palmer("missing", "missing"));

    let mut dag1 = FixtureKb::new();
    dag1.add("root", "root", WordNetRelation::Hypernym, &[]);
    dag1.add("middle-1", "middle-1", WordNetRelation::Hypernym, &["root"]);
    dag1.add("middle-2", "middle-2", WordNetRelation::Hypernym, &["middle-1"]);
    dag1.add("common", "common", WordNetRelation::Hypernym, &["root", "middle-2"]);
    dag1.add("left", "left", WordNetRelation::Hypernym, &["common"]);
    dag1.add("right", "right", WordNetRelation::Hypernym, &["common"]);
    assert!((SynsetSimilarity::new(dag1).wu_palmer("left", "right") - 0.8).abs() < 1e-9);

    let mut dag2 = FixtureKb::new();
    dag2.add("root", "root", WordNetRelation::Hypernym, &[]);
    dag2.add("common", "common", WordNetRelation::Hypernym, &["root"]);
    dag2.add("right", "right", WordNetRelation::Hypernym, &["common"]);
    dag2.add("branch-1", "branch-1", WordNetRelation::Hypernym, &["root"]);
    dag2.add("branch-2", "branch-2", WordNetRelation::Hypernym, &["branch-1"]);
    dag2.add("branch-3", "branch-3", WordNetRelation::Hypernym, &["branch-2"]);
    dag2.add("left", "left", WordNetRelation::Hypernym, &["common", "branch-3"]);
    assert!((SynsetSimilarity::new(dag2).wu_palmer("left", "right") - 0.5).abs() < 1e-9);

    let mut dangling = FixtureKb::new();
    dangling.add("left", "left", WordNetRelation::Hypernym, &["missing-parent"]);
    dangling.add("right", "right", WordNetRelation::Hypernym, &["missing-parent"]);
    let dsim = SynsetSimilarity::new(dangling);
    assert_eq!(None, dsim.shortest_distance("left", "right"));
    assert_eq!(0.0, dsim.path("left", "right"));
    assert_eq!(0.0, dsim.wu_palmer("left", "right"));
}

#[test]
fn test_hypernym_typer() {
    let typer = HypernymTyper::new(similarity_taxonomy(), [("person", "person"), ("location", "location")]).unwrap();
    assert_eq!(Some("person"), typer.type_lemma("chemist").unwrap());
    assert_eq!(Some("location"), typer.type_lemma("paris").unwrap());
    assert_eq!(Some("person"), typer.type_lemma("person").unwrap());
    assert_eq!(Some("location"), typer.type_synset("n8"));
    assert_eq!(None, typer.type_lemma("abstract").unwrap());
    assert_eq!(None, typer.type_lemma("unknownword").unwrap());
    assert_eq!(None, typer.type_synset("n12"));
    assert_eq!(None, typer.type_synset("missing"));

    let nested = HypernymTyper::new(similarity_taxonomy(), [("person", "person"), ("scientist", "scientist")]).unwrap();
    assert_eq!(Some("scientist"), nested.type_lemma("chemist").unwrap());
    assert_eq!(Some("scientist"), nested.type_lemma("scientist").unwrap());
    assert_eq!(None, nested.type_lemma("organism").unwrap());

    // Conflicting labels for same synset rejected
    let lmf = lmf_fixture();
    assert!(HypernymTyper::new(&lmf, [("dog", "animal"), ("domestic dog", "pet")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), Vec::<(&str, &str)>::new()).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [(" ", "person")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [("person", "\u{00A0}")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [("notaword", "label")]).is_err());
    assert!(typer.type_lemma(" ").is_err());
}

#[test]
fn test_navigate_semantic_role_relations_and_report_footprint() {
    let path = fixtures_dir().join("relation-usage-wn-lmf.xml");
    let start = Instant::now();
    let lexicon = WnLmfReader::read(&path).unwrap();
    let load_elapsed = start.elapsed();

    let purchase = lexicon.lookup("purchase", WordNetPos::Verb)[0];
    let agent_id = &purchase.related(WordNetRelation::InvolvedAgent)[0];
    let instrument_id = &purchase.related(WordNetRelation::InvolvedInstrument)[0];
    assert_eq!(&["buyer"], lexicon.synset(agent_id).unwrap().lemmas());
    assert_eq!(&["payment card"], lexicon.synset(instrument_id).unwrap().lemmas());

    // Verify load time and memory footprint for miniature WNDB + WN-LMF fixtures (#201)
    let wndb_start = Instant::now();
    let wndb = wndb_fixture();
    let wndb_elapsed = wndb_start.elapsed();
    let wndb_bytes = wndb.memory_footprint_bytes();
    let lmf_bytes = lexicon.memory_footprint_bytes();
    assert!(load_elapsed.as_secs() < 5);
    assert!(wndb_elapsed.as_secs() < 5);
    assert!(wndb_bytes > 0);
    assert!(lmf_bytes > 0);
}
