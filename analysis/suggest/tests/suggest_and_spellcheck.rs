use std::time::Instant;

use turbo_analysis::Span;
use turbo_stem::{SnowballAlgorithm, SnowballStemmer};
use turbo_suggest::{ImageBuf, MappedSurfaceLexicon, QuerySpellchecker, SpellcheckConfig, SurfaceLexiconBuilder};

#[test]
fn mapped_surface_lexicon_completes_stemmed_and_diacritic_terms_in_place() {
    let stemmer = SnowballStemmer::new(SnowballAlgorithm::English);
    let mut builder = SurfaceLexiconBuilder::new().with_top_k_prefix(8).with_top_k_stem(8);

    // Stemmed terms ("courtesy" -> "courtesi", "courtesies" -> "courtesi", "courteous" -> "courteous")
    for _ in 0..50 {
        builder.observe_document("Courtesy of the diplomatic office.", &stemmer, true);
    }
    for _ in 0..12 {
        builder.observe_document("Diplomatic courtesies were extended.", &stemmer, true);
    }
    for _ in 0..30 {
        builder.observe_document("Courteous staff welcomed guests.", &stemmer, true);
    }
    // Unicode diacritics
    builder.observe("café", "cafe", 40);
    builder.observe("cafeteria", "cafeteria", 20);
    builder.observe("naïve", "naiv", 15);

    let img: ImageBuf = builder.build_image().expect("build lexicon image");
    let lex = MappedSurfaceLexicon::from_bytes(img.bytes()).expect("zero-copy load");

    // 1. Prefix "courtes" suggests human surfaces ("courtesy", "courtesies"), never raw stem "courtesi"!
    let comp = lex.suggest_prefix("courtes", 5);
    assert_eq!(comp.len(), 2);
    assert_eq!(comp[0].surface, "courtesy");
    assert_eq!(comp[0].stem, "courtesi");
    assert_eq!(comp[0].count, 50);
    assert_eq!(comp[1].surface, "courtesies");
    assert_eq!(comp[1].stem, "courtesi");
    assert_eq!(comp[1].count, 12);

    // 2. Full English word "courtesy" matches surface "courtesy" backed by stem "courtesi"!
    let full = lex.suggest_prefix("courtesy", 5);
    assert_eq!(full.len(), 1);
    assert_eq!(full[0].surface, "courtesy");
    assert_eq!(full[0].stem, "courtesi");

    // 3. Stem-to-surface reverse lookup maps "courtesi" -> ["courtesy", "courtesies"].
    let surfaces = lex.stem_surfaces("courtesi", 5);
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].surface, "courtesy");
    assert_eq!(surfaces[1].surface, "courtesies");
    assert_eq!(lex.best_surface_for_stem("courtesi"), Some("courtesy"));

    // 4. Diacritic folding: ASCII "cafe" completes to "café" and "cafeteria"; accented "café" ranks "café" first!
    let ascii_cafe = lex.suggest_prefix("cafe", 5);
    assert_eq!(ascii_cafe[0].surface, "café");
    assert_eq!(ascii_cafe[1].surface, "cafeteria");

    let accented_cafe = lex.suggest_prefix("café", 5);
    assert_eq!(accented_cafe[0].surface, "café");

    let naive = lex.suggest_prefix("naiv", 5);
    assert_eq!(naive[0].surface, "naïve");

    // 5. Microsecond lookup verification (< 50 ms for 1,000 lookups in debug).
    let t0 = Instant::now();
    let mut hits = 0usize;
    for _ in 0..1000 {
        hits += lex.suggest_prefix("courtes", 5).len();
        hits += lex.stem_surfaces("courtesi", 5).len();
    }
    let elapsed = t0.elapsed();
    println!("suggest_and_stem_1000_pairs_elapsed={elapsed:?} per_call_ns={}", elapsed.as_nanos() / 2000);
    assert_eq!(hits, 4000);
    assert!(elapsed.as_micros() < 50_000, "2000 zero-copy lookups took {elapsed:?}");
}

#[test]
fn query_spellchecker_disambiguates_bigrams_preserves_syntax_and_projects_utf16_spans() {
    let mut corpus = Vec::new();
    corpus.extend(std::iter::repeat_n("The United States Supreme Court issued a landmark opinion today.", 40));
    // Add "cart" and "cort" with unigram occurrences in unrelated contexts so
    // context-blind single-term spellcheck would fail on real-word "cort"/"cart"!
    corpus.extend(std::iter::repeat_n("Push the shopping cart down the grocery aisle.", 60));
    corpus.extend(std::iter::repeat_n("A legacy cort measurement appeared in old notes.", 5));

    let checker = QuerySpellchecker::from_documents(&corpus, SpellcheckConfig::default());

    // Multi-word contextual repair + astral emoji to prove UTF-8 vs UTF-16 vs code-point spans!
    let query = "⚖️🚀 united states supreem cort AND \"supreem cort\" field:cort";
    let res = checker.check_query(query, &[Span::new(0, 0)]);
    assert!(res.did_change);
    assert_eq!(res.corrected_query, "⚖️🚀 united states supreme court AND \"supreem cort\" field:cort");
    assert!(res.confidence > 0.5 && res.confidence <= 1.0);
    assert_eq!(res.edits.len(), 2);

    // Edit 0: "supreem" -> "supreme"
    let e0 = &res.edits[0];
    assert_eq!(e0.original, "supreem");
    assert_eq!(e0.replacement, "supreme");
    assert_eq!(&query[e0.utf8_span.range()], "supreem");
    // "⚖️" is U+2696 (3 bytes, 1 UTF-16, 1 CP) + U+FE0F (3 bytes, 1 UTF-16, 1 CP)
    // "🚀" is U+1F680 (4 bytes, 2 UTF-16, 1 CP)
    // " united states " is 15 ASCII chars (15 bytes, 15 UTF-16, 15 CP)
    // Total before "supreem": 25 UTF-8 bytes, 19 UTF-16 units, 18 code points!
    assert_eq!((e0.utf8_span.start, e0.utf8_span.end), (25, 32));
    assert_eq!((e0.utf16_span.start.0, e0.utf16_span.end.0), (19, 26));
    assert_eq!((e0.code_point_span.start.0, e0.code_point_span.end.0), (18, 25));

    // Edit 1: real-word "cort" -> "court" (NOT higher-unigram "cart"!)
    let e1 = &res.edits[1];
    assert_eq!(e1.original, "cort");
    assert_eq!(e1.replacement, "court");
    assert_eq!(&query[e1.utf8_span.range()], "cort");
}
