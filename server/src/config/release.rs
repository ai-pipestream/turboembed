//! The PII release flags and their variables (docs/grpc-analysis.md, PII
//! value release, #1909): the grants per verified client certificate, the
//! token keys, the encryption keys of the reversible redaction strategies
//! (#1885), the audit service, and whether a values request must name
//! the policy the graph recorded (#1977). Every one but the encryption
//! keys needs `--pii-grant`, and a grant needs mTLS and the audit
//! endpoint. An encryption key alone releases nothing: a redaction
//! policy's AES-256-GCM and FF1 arms encrypt with it (#2111).

use std::time::Duration;

use crate::release::ReleaseConfig;
use crate::release::audit::DEFAULT_AUDIT_TIMEOUT;
use crate::release::caps::Grants;
use crate::release::keys::{self, KeyRef, TokenKeyFile};
use crate::tls::TlsConfig;

pub const ENV_PII_GRANTS: &str = "TURBO_KSERVE_PII_GRANTS";
pub const ENV_PII_TOKEN_KEYS: &str = "TURBO_KSERVE_PII_TOKEN_KEYS";
pub const ENV_PII_ENCRYPTION_KEYS: &str = "TURBO_KSERVE_PII_ENCRYPTION_KEYS";
pub const ENV_PII_TOKEN_KEY_ACTIVE: &str = "TURBO_KSERVE_PII_TOKEN_KEY_ACTIVE";
pub const ENV_PII_ENCRYPTION_KEY_ACTIVE: &str = "TURBO_KSERVE_PII_ENCRYPTION_KEY_ACTIVE";
pub const ENV_PII_AUDIT_ENDPOINT: &str = "TURBO_KSERVE_PII_AUDIT_ENDPOINT";
pub const ENV_PII_AUDIT_TIMEOUT_MS: &str = "TURBO_KSERVE_PII_AUDIT_TIMEOUT_MS";
pub const ENV_PII_REQUIRE_RECORDED_POLICY: &str = "TURBO_KSERVE_PII_REQUIRE_RECORDED_POLICY";

/// The release flags as given on the command line, before their variables.
#[derive(Default)]
pub(super) struct ReleaseFlags {
    grants: Vec<String>,
    token_keys: Vec<String>,
    encryption_keys: Vec<String>,
    active: Option<String>,
    encryption_active: Option<String>,
    endpoint: Option<String>,
    timeout_ms: Option<String>,
    require_recorded: Option<bool>,
}

impl ReleaseFlags {
    /// Takes `flag` if it is a release flag: `inline` is its `=value`, and
    /// `value` reads the value the flag needs. Answers false for any other
    /// flag. `--pii-grant`, `--pii-token-key` and `--pii-encryption-key`
    /// repeat.
    pub(super) fn take(
        &mut self,
        flag: &str,
        inline: bool,
        value: impl FnOnce() -> Result<String, String>,
    ) -> Result<bool, String> {
        match flag {
            "--pii-require-recorded-policy" => {
                if inline || self.require_recorded.replace(true).is_some() {
                    return Err("--pii-require-recorded-policy takes no value and may be given once".into());
                }
            }
            "--pii-grant" => self.grants.push(value()?),
            "--pii-token-key" => self.token_keys.push(value()?),
            "--pii-encryption-key" => self.encryption_keys.push(value()?),
            "--pii-token-key-active" => super::once(flag, value()?, &mut self.active)?,
            "--pii-encryption-key-active" => super::once(flag, value()?, &mut self.encryption_active)?,
            "--pii-audit-endpoint" => super::once(flag, value()?, &mut self.endpoint)?,
            "--pii-audit-timeout-ms" => super::once(flag, value()?, &mut self.timeout_ms)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The release settings, with `env` filling in what the flags leave
    /// out; a repeated flag's variable is a `;`-separated list, read only
    /// when the flag is not given. Without a grant there is no release,
    /// and the encryption keys come back on their own, for the redaction
    /// policies' AES-256-GCM and FF1 arms.
    pub(super) fn resolve(
        self,
        env: impl Fn(&str) -> Option<String>,
        tls: Option<&TlsConfig>,
    ) -> Result<(Option<ReleaseConfig>, Vec<TokenKeyFile>), String> {
        let from_env = |slot: Option<String>, k: &str| slot.or_else(|| env(k).filter(|v| !v.is_empty()));
        let list = |given: Vec<String>, k: &str| -> Vec<String> {
            if !given.is_empty() {
                return given;
            }
            env(k)
                .map(|v| v.split(';').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect())
                .unwrap_or_default()
        };
        let grants = list(self.grants, ENV_PII_GRANTS);
        let token_keys = list(self.token_keys, ENV_PII_TOKEN_KEYS);
        let encryption_keys = list(self.encryption_keys, ENV_PII_ENCRYPTION_KEYS);
        if !encryption_keys.is_empty() && !cfg!(feature = "pii-reversible") {
            return Err("--pii-encryption-key needs a server built with the pii-reversible feature".into());
        }
        let active = from_env(self.active, ENV_PII_TOKEN_KEY_ACTIVE);
        let encryption_active = from_env(self.encryption_active, ENV_PII_ENCRYPTION_KEY_ACTIVE);
        let endpoint = from_env(self.endpoint, ENV_PII_AUDIT_ENDPOINT);
        let timeout_ms = from_env(self.timeout_ms, ENV_PII_AUDIT_TIMEOUT_MS);
        let require_recorded_policy = match (self.require_recorded, env(ENV_PII_REQUIRE_RECORDED_POLICY)) {
            (Some(on), _) => on,
            (None, None) => false,
            (None, Some(v)) => match v.as_str() {
                "" | "0" | "false" | "no" => false,
                "1" | "true" | "yes" => true,
                _ => {
                    return Err(format!(
                        "{ENV_PII_REQUIRE_RECORDED_POLICY}={v}: not one of 1, true, yes, 0, false, no"
                    ));
                }
            },
        };
        let encryption_keys = encryption_keys
            .iter()
            .map(|k| TokenKeyFile::parse(k).map_err(|e| format!("--pii-encryption-key: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        if grants.is_empty() {
            if !token_keys.is_empty()
                || active.is_some()
                || encryption_active.is_some()
                || endpoint.is_some()
                || timeout_ms.is_some()
                || require_recorded_policy
            {
                return Err("--pii-token-key, --pii-token-key-active, --pii-encryption-key-active, \
                            --pii-audit-endpoint, --pii-audit-timeout-ms \
                            and --pii-require-recorded-policy need --pii-grant"
                    .into());
            }
            keys::distinct("--pii-encryption-key", &encryption_keys)?;
            return Ok((None, encryption_keys));
        }
        let mut parsed = Grants::default();
        for g in &grants {
            parsed.add(g).map_err(|e| format!("--pii-grant: {e}"))?;
        }
        let token_keys = token_keys
            .iter()
            .map(|k| TokenKeyFile::parse(k).map_err(|e| format!("--pii-token-key: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let active_token_key =
            active.map(|a| KeyRef::parse(&a).map_err(|e| format!("--pii-token-key-active: {e}"))).transpose()?;
        let active_encryption_key = encryption_active
            .map(|a| KeyRef::parse(&a).map_err(|e| format!("--pii-encryption-key-active: {e}")))
            .transpose()?;
        let audit_endpoint = endpoint.ok_or("--pii-grant needs --pii-audit-endpoint: every release is audited")?;
        let audit_timeout = match timeout_ms {
            None => DEFAULT_AUDIT_TIMEOUT,
            Some(v) => match v.parse::<u64>() {
                Ok(n) if n > 0 => Duration::from_millis(n),
                _ => return Err(format!("--pii-audit-timeout-ms {v}: not a count of milliseconds over 0")),
            },
        };
        let config = ReleaseConfig {
            grants: parsed,
            token_keys,
            active_token_key,
            encryption_keys,
            active_encryption_key,
            audit_endpoint,
            audit_timeout,
            require_recorded_policy,
        };
        config.check(tls)?;
        Ok((Some(config), Vec::new()))
    }
}

#[cfg(test)]
#[path = "release_tests.rs"]
mod tests;
