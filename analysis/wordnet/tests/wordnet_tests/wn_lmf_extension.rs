//! WN-LMF lexicon extensions, composition, and how dependencies are
//! resolved.

use super::*;

// ============================================================================
// WnLmfExtensionTest, WnLmfCompositionValidationTest, WnLmfResolutionContractTest
// ============================================================================

struct CatalogResolver {
    catalog: Mutex<HashMap<String, String>>,
    calls: AtomicUsize,
}

impl CatalogResolver {
    fn new() -> Self {
        Self { catalog: Mutex::new(HashMap::new()), calls: AtomicUsize::new(0) }
    }

    fn with(self, r: &str, version: &str, doc: impl Into<String>) -> Self {
        self.catalog.lock().unwrap().insert(format!("{r}@{version}"), doc.into());
        self
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WnLmfResolver for CatalogResolver {
    fn resolve(&self, reference: &WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let key = format!("{}@{}", reference.id(), reference.version());
        let Some(doc) = self.catalog.lock().unwrap().get(&key).cloned() else {
            return Ok(None);
        };
        Ok(Some(WnLmfSource::named_bytes(format!("{}-{}.xml", reference.id(), reference.version()), doc.into_bytes())?))
    }
}

fn base_resolver() -> CatalogResolver {
    CatalogResolver::new().with("ewn", "2020", fixture_string("extension-base.xml"))
}

#[test]
fn test_computer_science_extension_example() {
    let ext_xml = fixture_string("extension-computer-science.xml");
    let resolver = base_resolver();
    let resource = WnLmfReader::read_resource_named_with_resolver(
        Cursor::new(ext_xml.as_bytes()),
        "extension-computer-science.xml",
        &resolver,
    )
    .unwrap();
    assert_eq!(1, resource.lexicons().len());
    let composed = &resource.lexicons()[0];
    assert_eq!("ewn-cs-example", composed.id());
    assert_eq!("English WordNet Computer Science Terms (example)", composed.label());
    assert_eq!("en", composed.language());
    assert_eq!("1.0", composed.version());
    assert_eq!(Some(&WnLmfDependency::new("ewn", "2020", None).unwrap()), composed.extension_of());
    assert_eq!(
        &[WnLmfDependency::new("omw-fr", "1.4", Some("https://example.org/omw-fr-1.4.xml".to_string())).unwrap()],
        composed.dependencies()
    );

    let kb = composed.knowledge_base();
    let process_ids: Vec<&str> = kb.lookup("process", WordNetPos::Noun).iter().map(|s| s.id()).collect();
    assert_eq!(vec!["ewn-08056231-n", "ewn-05984756-n", "ewn-20000123-n"], process_ids);

    let computer_process = kb.synset("ewn-20000123-n").unwrap();
    assert_eq!(&["process"], computer_process.lemmas());
    assert_eq!("a running instance of a computer program", computer_process.gloss());
    assert_eq!(&["ewn-06581154-n"], computer_process.related(WordNetRelation::Hypernym));
    assert_eq!(6, kb.size());
}

#[test]
fn test_extension_definition_fills_empty_base_gloss_but_never_replaces_non_empty() {
    let doc = "<LexicalResource><LexiconExtension id=\"ext\" label=\"ext\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <ExternalSynset id=\"ewn-90000001-n\">\
        <Definition>a placeholder definition</Definition></ExternalSynset>\
        <ExternalSynset id=\"ewn-08056231-n\">\
        <Definition>must not replace the base definition</Definition></ExternalSynset>\
        </LexiconExtension></LexicalResource>";
    let kb = WnLmfReader::read_named_with_resolver(Cursor::new(doc.as_bytes()), "ext.xml", &base_resolver()).unwrap();
    assert_eq!("a placeholder definition", kb.synset("ewn-90000001-n").unwrap().gloss());
    assert_eq!("a phenomenon that unfolds over time", kb.synset("ewn-08056231-n").unwrap().gloss());
}

#[test]
fn test_sibling_extensions_compose_independently_and_share_one_resolver_call() {
    let doc = "<LexicalResource>\
        <LexiconExtension id=\"ext-a\" label=\"ext-a\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <LexicalEntry id=\"ext-a-alpha-n\"><Lemma writtenForm=\"alpha\" partOfSpeech=\"n\"/>\
        <Sense id=\"ext-a-alpha-n-1\" synset=\"ext-a-1-n\"/></LexicalEntry>\
        <Synset id=\"ext-a-1-n\" partOfSpeech=\"n\"/></LexiconExtension>\
        <LexiconExtension id=\"ext-b\" label=\"ext-b\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <LexicalEntry id=\"ext-b-beta-n\"><Lemma writtenForm=\"beta\" partOfSpeech=\"n\"/>\
        <Sense id=\"ext-b-beta-n-1\" synset=\"ext-b-1-n\"/></LexicalEntry>\
        <Synset id=\"ext-b-1-n\" partOfSpeech=\"n\"/></LexiconExtension>\
        </LexicalResource>";
    let resolver = base_resolver();
    let resource =
        WnLmfReader::read_resource_named_with_resolver(Cursor::new(doc.as_bytes()), "siblings.xml", &resolver).unwrap();
    assert_eq!(1, resolver.calls());
    let first = resource.lexicon("ext-a").unwrap().knowledge_base();
    let second = resource.lexicon("ext-b").unwrap().knowledge_base();
    assert_eq!(1, first.lookup("alpha", WordNetPos::Noun).len());
    assert!(first.lookup("beta", WordNetPos::Noun).is_empty());
    assert_eq!(1, second.lookup("beta", WordNetPos::Noun).len());
    assert!(second.lookup("alpha", WordNetPos::Noun).is_empty());
}

#[test]
fn test_extension_cycles_and_depth_limit_are_enforced() {
    let self_cycle = "<LexicalResource><LexiconExtension id=\"loop\" label=\"loop\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"loop\" version=\"1.0\"/></LexiconExtension></LexicalResource>";
    let resolver = CatalogResolver::new().with("loop", "1.0", self_cycle);
    let err =
        WnLmfReader::read_named_with_resolver(Cursor::new(self_cycle.as_bytes()), "loop.xml", &resolver).unwrap_err();
    assert!(err.to_string().contains("cycle"), "{err}");

    let make_chain = |depth: usize| {
        let mut r = CatalogResolver::new().with(
            "chain-0",
            "1.0",
            "<LexicalResource><Lexicon id=\"chain-0\" label=\"chain-0\" language=\"en\" version=\"1.0\">\
             <LexicalEntry id=\"c0-e\"><Lemma writtenForm=\"process\" partOfSpeech=\"n\"/>\
             <Sense id=\"c0-s\" synset=\"c0-syn\"/></LexicalEntry>\
             <Synset id=\"c0-syn\" partOfSpeech=\"n\"/></Lexicon></LexicalResource>",
        );
        for i in 1..depth {
            let prev = i - 1;
            r = r.with(
                &format!("chain-{i}"),
                "1.0",
                format!(
                    "<LexicalResource><LexiconExtension id=\"chain-{i}\" label=\"chain-{i}\" \
                     language=\"en\" version=\"1.0\"><Extends ref=\"chain-{prev}\" version=\"1.0\"/>\
                     </LexiconExtension></LexicalResource>"
                ),
            );
        }
        let prev = depth - 1;
        let top = format!(
            "<LexicalResource><LexiconExtension id=\"chain-{depth}\" label=\"chain-{depth}\" \
             language=\"en\" version=\"1.0\"><Extends ref=\"chain-{prev}\" version=\"1.0\"/>\
             </LexiconExtension></LexicalResource>"
        );
        (top, r)
    };

    let (top_ok, res_ok) = make_chain(MAX_EXTENSION_DEPTH);
    let kb = WnLmfReader::read_named_with_resolver(Cursor::new(top_ok.as_bytes()), "ok.xml", &res_ok).unwrap();
    assert_eq!(1, kb.lookup("process", WordNetPos::Noun).len());

    let (top_deep, res_deep) = make_chain(MAX_EXTENSION_DEPTH + 1);
    let err =
        WnLmfReader::read_named_with_resolver(Cursor::new(top_deep.as_bytes()), "deep.xml", &res_deep).unwrap_err();
    assert!(err.to_string().contains(&MAX_EXTENSION_DEPTH.to_string()), "{err}");
}

#[test]
fn test_normalized_and_rejected_member_separators() {
    let base_entry = "<LexicalEntry id=\"base-entry\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
        <Sense id=\"base-sense\" synset=\"base-synset\"/></LexicalEntry>\
        <LexicalEntry id=\"dog-entry\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
        <Sense id=\"dog-sense\" synset=\"base-synset\"/></LexicalEntry>";

    for sep in [" ", "\t", "\n", "\r", "\r\n", "\t \r\n", "&#32;", "&#x20;"] {
        let doc = wrap_lmf(&format!(
            "{base_entry}<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"{sep}dog-sense{sep}{sep}base-sense{sep}\"/>"
        ));
        let kb = parse_lmf(&doc).unwrap();
        assert_eq!(&["dog", "cat"], kb.synset("base-synset").unwrap().lemmas());
    }

    for sep in ["&#9;", "&#x9;", "&#10;", "&#xA;", "&#13;", "&#xD;", "&#xA0;", "&#x2003;"] {
        let doc = wrap_lmf(&format!(
            "{base_entry}<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"dog-sense{sep}base-sense\"/>"
        ));
        let err = parse_lmf(&doc).unwrap_err();
        assert!(err.to_string().contains("dog-sense"), "{err}");
    }
}
