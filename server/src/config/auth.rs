//! Command-line and environment parsing for bearer token authentication
//! (`#2346`). Token values are never echoed in error messages.

use crate::auth::AuthConfig;

pub const ENV_BEARER_TOKENS: &str = "TURBO_KSERVE_BEARER_TOKENS";
pub const ENV_ADMIN_BEARER_TOKENS: &str = "TURBO_KSERVE_ADMIN_BEARER_TOKENS";

#[derive(Default)]
pub(super) struct AuthFlags {
    bearer: Vec<String>,
    admin_bearer: Vec<String>,
}

impl AuthFlags {
    pub(super) fn take(
        &mut self,
        flag: &str,
        mut value: impl FnMut() -> Result<String, String>,
    ) -> Result<bool, String> {
        match flag {
            "--bearer-token" => {
                let v = value()?;
                if v.trim().is_empty() {
                    return Err("--bearer-token must not be empty".into());
                }
                self.bearer.push(v);
                Ok(true)
            }
            "--admin-bearer-token" => {
                let v = value()?;
                if v.trim().is_empty() {
                    return Err("--admin-bearer-token must not be empty".into());
                }
                self.admin_bearer.push(v);
                Ok(true)
            }
            "--bearer-token-file" => {
                self.bearer.push(read_token_file("--bearer-token-file", &value()?)?);
                Ok(true)
            }
            "--admin-bearer-token-file" => {
                self.admin_bearer.push(read_token_file("--admin-bearer-token-file", &value()?)?);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(super) fn resolve(mut self, env: &impl Fn(&str) -> Option<String>) -> Result<Option<AuthConfig>, String> {
        if self.bearer.is_empty()
            && let Some(raw) = env(ENV_BEARER_TOKENS).filter(|s| !s.trim().is_empty())
        {
            for tok in split_tokens(&raw, ENV_BEARER_TOKENS)? {
                self.bearer.push(tok);
            }
        }
        if self.admin_bearer.is_empty()
            && let Some(raw) = env(ENV_ADMIN_BEARER_TOKENS).filter(|s| !s.trim().is_empty())
        {
            for tok in split_tokens(&raw, ENV_ADMIN_BEARER_TOKENS)? {
                self.admin_bearer.push(tok);
            }
        }
        if self.bearer.is_empty() && self.admin_bearer.is_empty() {
            return Ok(None);
        }
        AuthConfig::from_tokens(&self.bearer, &self.admin_bearer).map(Some)
    }
}

fn read_token_file(flag: &str, path: &str) -> Result<String, String> {
    let p = path.trim();
    if p.is_empty() {
        return Err(format!("{flag} path must not be empty"));
    }
    let raw = std::fs::read_to_string(p).map_err(|e| format!("{flag} {p}: {e}"))?;
    let tok = raw.trim();
    if tok.is_empty() {
        return Err(format!("{flag} {p} must not be empty"));
    }
    Ok(tok.to_owned())
}

fn split_tokens(raw: &str, source: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for piece in raw.split([',', ';']) {
        let t = piece.trim();
        if t.is_empty() {
            return Err(format!("{source} contains an empty token entry"));
        }
        out.push(t.to_owned());
    }
    Ok(out)
}
