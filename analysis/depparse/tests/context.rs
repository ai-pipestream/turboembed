use turbo_analysis_layers::Tokens;
use turbo_depparse::{ArcStandardState, DependencyContextGenerator, Transition};

fn context(tokens: &[&str], tags: &[&str]) -> Vec<String> {
    let mut state = ArcStandardState::<Tokens>::new(tokens.len()).unwrap();
    state.apply(&Transition::Shift).unwrap();
    state.apply(&Transition::Shift).unwrap();
    DependencyContextGenerator::context(&state, tokens, tags).unwrap()
}

fn feature<'a>(features: &'a [String], name: &str) -> &'a str {
    features.iter().find(|f| f.starts_with(name)).map(String::as_str).expect("the feature is emitted")
}

#[test]
fn combined_features_escape_separators_in_values() {
    let tokens = ["w0", "w1", "w2", "w3"];
    // s1 is token 0, s0 is token 1: without escaping, X|Y with Z and X
    // with Y|Z would give the same combined feature.
    let a = context(&tokens, &["X|Y", "Z", "T", "U"]);
    let b = context(&tokens, &["X", "Y|Z", "T", "U"]);
    assert_eq!(feature(&a, "s1t,s0t="), r"s1t,s0t=X\|Y|Z");
    assert_eq!(feature(&b, "s1t,s0t="), r"s1t,s0t=X|Y\|Z");
    // Single-value features keep the value as it is.
    assert_eq!(feature(&a, "s1t="), "s1t=X|Y");
}

#[test]
fn combined_features_escape_slashes_and_backslashes() {
    let tags = ["T", "T", "T", "U"];
    let a = context(&["w0", "a/b", "w2", "w3"], &tags);
    let b = context(&["w0", "a", "w2", "w3"], &["T", "b/T", "T", "U"]);
    assert_eq!(feature(&a, "s0wt="), r"s0wt=a\/b/T");
    assert_eq!(feature(&b, "s0wt="), r"s0wt=a/b\/T");
    assert_ne!(feature(&a, "s0wt="), feature(&b, "s0wt="));
    let c = context(&["w0", "a\\b", "w2", "w3"], &tags);
    assert_eq!(feature(&c, "s0wt="), r"s0wt=a\\b/T");
    assert_eq!(feature(&c, "s0w="), "s0w=a\\b");
}
