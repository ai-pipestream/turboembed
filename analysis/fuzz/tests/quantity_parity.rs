//! JVM output from the nightly failure and real pinned-oracle controls.
use turbo_fuzz::differential::{RunCache, check_golden};
use turbo_fuzz::{Outcome, Resources};

#[test]
fn quantity_oracle_includes_regional_currency_controls() {
    let data = turbo_goldens::data_dir();
    let resources = Resources::from_env();
    let mut cache = RunCache::new(&resources);
    let mut failures = Vec::new();
    for notation in ["latin-us", "latin-eu"] {
        for id in [
            "nightly-aud-prefix",
            "nightly-usd-prefix",
            "nightly-hkd-prefix",
            "nightly-dollar-control",
            "nightly-plain-control",
            "nightly-decimal-comma",
            "nightly-utf8-prefix",
            "nightly-inner-prefix",
        ] {
            let golden = data.join("quantity").join(notation).join("quantity").join(format!("{id}.golden"));
            let found = check_golden(&golden, &data.join("inputs"), &mut cache).unwrap();
            if found.outcome != Outcome::Match || !found.violations.is_empty() {
                failures.push(found);
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
