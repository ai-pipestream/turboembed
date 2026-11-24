//! Which requests share an analyzer, and that a shared one answers as a
//! new one does (#2418).

use super::*;
use crate::proto::result::QueryDenseTargetProto;

fn body_field() -> QueryFieldProfileProto {
    QueryFieldProfileProto {
        field: "body".into(),
        tokenizer: QueryFieldTokenizerProto::QueryFieldTokenizerWord as i32,
        term_form: QueryTermFormProto::QueryTermFormStem as i32,
        language: String::new(),
        keep_invisible: false,
        include_stopwords_in_residual: false,
        synonyms: Vec::new(),
    }
}

fn nyc() -> QuerySynonymRuleProto {
    QuerySynonymRuleProto {
        trigger: vec!["new".into(), "york".into(), "city".into()],
        expansion: vec!["nyc".into()],
        bidirectional: true,
    }
}

fn request(query: &str) -> AnalyzeQueryRequest {
    AnalyzeQueryRequest {
        query: query.into(),
        default_language: "en".into(),
        reference_date_iso: "2026-07-14".into(),
        field_profiles: vec![body_field()],
        global_synonyms: vec![nyc()],
        ..Default::default()
    }
}

#[test]
fn requests_of_one_profile_share_one_analyzer() {
    let kept = AnalyzerCache::default();
    let first = analyzer_for(&kept, &request("startups in Berlin")).unwrap();

    // The query, the offset mode and the targets are not part of a profile.
    let mut other = request("new york city since 2024");
    other.offsets = crate::proto::result::OffsetMode::Utf16 as i32;
    other.dense_targets.push(QueryDenseTargetProto { target_id: "dense".into(), ..Default::default() });
    // The language and the date are trimmed before they are read.
    other.default_language = " en ".into();
    other.reference_date_iso = "2026-07-14 ".into();

    assert!(Arc::ptr_eq(&first, &analyzer_for(&kept, &other).unwrap()));
    assert_eq!(kept.len(), 1);
}

#[test]
fn each_field_of_the_profile_gives_an_analyzer_of_its_own() {
    let field = |change: fn(&mut QueryFieldProfileProto)| {
        let mut req = request("q");
        change(&mut req.field_profiles[0]);
        req
    };
    let top = |change: fn(&mut AnalyzeQueryRequest)| {
        let mut req = request("q");
        change(&mut req);
        req
    };
    let profiles = [
        request("q"),
        top(|r| r.default_language = "es".into()),
        top(|r| r.default_language.clear()),
        top(|r| r.reference_date_iso = "2026-07-15".into()),
        top(|r| r.reference_date_iso.clear()),
        top(|r| r.global_synonyms.clear()),
        top(|r| r.global_synonyms[0].bidirectional = false),
        top(|r| r.global_synonyms[0].expansion[0] = "ny".into()),
        top(|r| r.field_profiles.clear()),
        top(|r| r.field_profiles.push(QueryFieldProfileProto { field: "title".into(), ..body_field() })),
        // The same rule, for the one field and not for every field.
        top(|r| r.field_profiles[0].synonyms = std::mem::take(&mut r.global_synonyms)),
        field(|f| f.field = "title".into()),
        field(|f| f.tokenizer = QueryFieldTokenizerProto::QueryFieldTokenizerWhitespace as i32),
        field(|f| f.term_form = QueryTermFormProto::QueryTermFormFolded as i32),
        field(|f| f.language = "es".into()),
        field(|f| f.keep_invisible = true),
        field(|f| f.include_stopwords_in_residual = true),
        field(|f| f.synonyms.push(nyc())),
    ];
    let kept = AnalyzerCache::default();
    let analyzers: Vec<_> = profiles.iter().map(|req| analyzer_for(&kept, req).unwrap()).collect();
    for (i, a) in analyzers.iter().enumerate() {
        for (j, b) in analyzers.iter().enumerate().skip(i + 1) {
            assert!(!Arc::ptr_eq(a, b), "profiles {i} and {j} share an analyzer");
        }
    }
    assert_eq!(kept.len(), profiles.len());
    // Asked again, each profile gets the analyzer it got before.
    for (req, before) in profiles.iter().zip(&analyzers) {
        assert!(Arc::ptr_eq(before, &analyzer_for(&kept, req).unwrap()));
    }
}

#[test]
fn a_refused_profile_is_refused_each_time_and_not_kept() {
    let kept = AnalyzerCache::default();
    let mut bad_date = request("q");
    bad_date.reference_date_iso = "2026-02-30".into();
    let mut bad_rule = request("q");
    bad_rule.global_synonyms[0].expansion.clear();
    for req in [&bad_date, &bad_rule, &bad_date, &bad_rule] {
        let Err(refused) = analyzer_for(&kept, req) else { panic!("the profile is refused") };
        assert_eq!(refused.code, TURBO_E_INVALID_ARGUMENT);
    }
    assert_eq!(kept.len(), 0);
}

#[test]
fn a_shared_analyzer_answers_each_query_as_a_new_one_does() {
    let queries = [
        "Caf\u{e9} \u{200B} new york city \u{1F680} startups over $5M in Berlin since 2024",
        "",
        "invoices from last week under 200 EUR",
        "new york city",
        "the running of the bulls in Pamplona, 12 km from here",
        "Caf\u{e9} \u{200B} new york city \u{1F680} startups over $5M in Berlin since 2024",
    ];
    let kept = AnalyzerCache::default();
    for query in queries {
        let req = request(query);
        let shared = analyzer_for(&kept, &req).unwrap().analyze(query).unwrap();
        let new = build_analyzer(&req).unwrap().analyze(query).unwrap();
        assert_eq!(shared, new, "query {query:?}");
    }
    assert_eq!(kept.len(), 1);
}

#[tokio::test]
async fn the_call_keeps_one_analyzer_for_the_requests_of_one_profile() {
    let state = Arc::new(State::new(Vec::new(), Vec::new()));
    let call = |req: AnalyzeQueryRequest| {
        analyze_query(
            Arc::clone(&state),
            req,
            crate::priority::Priority::Interactive,
            crate::deadline::Deadline::none(),
        )
    };
    let first = call(request("new york city since 2024")).await.unwrap();
    assert_eq!(state.query_analyzers.len(), 1);
    let again = call(request("new york city since 2024")).await.unwrap();
    assert_eq!(first, again);
    call(request("startups in Berlin")).await.unwrap();
    assert_eq!(state.query_analyzers.len(), 1);

    let mut spanish = request("startups en Berlin");
    spanish.default_language = "es".into();
    call(spanish).await.unwrap();
    assert_eq!(state.query_analyzers.len(), 2);
}
