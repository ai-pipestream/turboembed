//! The PII output policy flag and its variable (docs/grpc-analysis.md, PII
//! output): what a request's PiiOutputOptions may turn on (#1904).

use crate::results::disclosure::PiiOutputPolicy;

pub const ENV_PII_OUTPUT_ALLOW: &str = "TURBO_KSERVE_PII_OUTPUT_ALLOW";

/// `--pii-output-allow` as given, or its variable; every option but values
/// when neither is set.
pub(super) fn resolve(flag: Option<String>, env: impl Fn(&str) -> Option<String>) -> Result<PiiOutputPolicy, String> {
    match flag.or_else(|| env(ENV_PII_OUTPUT_ALLOW).filter(|v| !v.is_empty())) {
        None => Ok(PiiOutputPolicy::ALL),
        Some(v) => PiiOutputPolicy::parse(&v).map_err(|e| format!("--pii-output-allow {v}: {e}")),
    }
}

/// One `--pii-policy` path: a binary `RedactionPolicy` file (#1882).
pub(super) fn policy_path(v: String) -> Result<std::path::PathBuf, String> {
    if v.is_empty() {
        return Err("--pii-policy needs a path".into());
    }
    Ok(v.into())
}
