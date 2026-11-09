//! PII value release to verified calling services (#1909,
//! docs/grpc-analysis.md, PII value release).
//!
//! A calling service is the verified client certificate of its connection
//! ([`identity`]), and only that: TurboEmbed keeps no user model, and which
//! end user may see what stays with the calling platform. A grant gives a
//! service a set of forms per kind ([`caps`]). For each mention the forms
//! in effect are requested ∩ policy ∩ the request's redaction policy's
//! release ceiling for the kind ([`policy`], #1943) ∩ granted
//! ([`caps::effective`]);
//! mentions with none stay withheld, each saying why (#1931). Before any value leaves, one access
//! event is recorded with the audit service and acknowledged
//! ([`audit`]); a failed record fails the call, with no value. Tokens are
//! HMAC-SHA-256 under the active registered key ([`keys`]). Ciphertext
//! waits for #1885. Only InspectDocumentGraph releases values; every other
//! surface stays as #1904 decides it.

pub mod audit;
pub mod caps;
pub mod identity;
pub mod keys;
mod plan;
pub mod policy;

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use turbo_pii::checksum::sha256;

use crate::api::{Failure, TURBO_E_BUSY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE};
use crate::pii_policy::{NamedPolicies, NoRegistry, PolicyDigest, PolicyResolver};
use crate::proto::audit::{PiiAccessEvent, PiiAccessKindGrant};
use crate::proto::result::{InspectDocumentGraphRequest, PiiOutputOptions, PiiValueForm, TypedDocumentResult};
use crate::results::disclosure::PiiOutputPolicy;
use crate::tls::TlsConfig;
use audit::AuditClient;
use caps::{Fingerprint, Form, Forms, Grants};
use keys::{KeyRef, TokenKey, TokenKeyFile};

/// The policy id every event names until #1882's policies replace the
/// server flag.
pub const POLICY_ID: &str = "turbo-kserve.pii-output-allow.v1";

/// `InspectDocumentGraphRequest.operation_key`.
const OPERATION_KEY_FIELD: u32 = 6;

/// The release settings: the grants, the token keys, the audit service,
/// and whether a values request must name the redaction policy the
/// graph's `turbo:pii-edits` recorded (#1977).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseConfig {
    pub grants: Grants,
    pub token_keys: Vec<TokenKeyFile>,
    pub active_token_key: Option<KeyRef>,
    pub audit_endpoint: String,
    pub audit_timeout: Duration,
    pub require_recorded_policy: bool,
}

impl ReleaseConfig {
    /// What the settings need beyond themselves: verified identities
    /// (`--tls-client-ca`), a token key when any grant includes token, and
    /// an active key among several. Answers the active key's file.
    pub fn check(&self, tls: Option<&TlsConfig>) -> Result<Option<&TokenKeyFile>, String> {
        if !tls.is_some_and(|t| t.client_ca.is_some()) {
            return Err("--pii-grant needs mTLS: --tls-cert, --tls-key and --tls-client-ca".into());
        }
        if self.audit_timeout.is_zero() {
            return Err("--pii-audit-timeout-ms must be over 0".into());
        }
        let active = keys::active(&self.token_keys, self.active_token_key.as_ref())?;
        if self.grants.any(Form::Token) && active.is_none() {
            return Err("a --pii-grant with token needs --pii-token-key".into());
        }
        Ok(active)
    }
}

/// What a release resolves a request's redaction policy with.
pub type Resolver = Arc<dyn PolicyResolver + Send + Sync>;

/// The running release: grants, the loaded token key, the audit client,
/// the redaction policies a request may name (#1943), and whether a
/// recorded policy is required (#1977).
pub struct Release {
    grants: Grants,
    token: Option<TokenKey>,
    audit: AuditClient,
    policies: NamedPolicies,
    resolver: Resolver,
    require_recorded_policy: bool,
}

impl Release {
    /// Checks `config` against `tls`, reads the active token key and makes
    /// the audit client; `None` without a config. A request may name only
    /// an inline policy with no registered reference.
    pub fn start(config: Option<&ReleaseConfig>, tls: Option<&TlsConfig>) -> Result<Option<Arc<Release>>, String> {
        Release::start_with(config, tls, NamedPolicies::default(), Arc::new(NoRegistry))
    }

    /// [`Release::start`], with the server's named `policies` and the
    /// `resolver` of an inline policy's references.
    pub fn start_with(
        config: Option<&ReleaseConfig>,
        tls: Option<&TlsConfig>,
        policies: NamedPolicies,
        resolver: Resolver,
    ) -> Result<Option<Arc<Release>>, String> {
        let Some(config) = config else { return Ok(None) };
        let active = config.check(tls)?;
        let token = active.map(TokenKey::load).transpose()?;
        let audit = AuditClient::new(&config.audit_endpoint, config.audit_timeout)?;
        Ok(Some(Arc::new(Release {
            grants: config.grants.clone(),
            token,
            audit,
            policies,
            resolver,
            require_recorded_policy: config.require_recorded_policy,
        })))
    }

    /// InspectDocumentGraph for a request that asks for values in `form`,
    /// from `service`. A request without an operation key, from a service
    /// granted `form` for no kind, or naming a redaction policy that does
    /// not resolve, is refused before the graph is read (#1931, #1943).
    /// Mentions the service is not granted, or the policy's ceiling
    /// excludes, stay withheld; the rest receive their value only after
    /// the audit acknowledgement.
    pub async fn inspect(
        self: Arc<Self>,
        service: Option<Fingerprint>,
        mut req: InspectDocumentGraphRequest,
        form: Form,
        policy: PiiOutputPolicy,
    ) -> Result<TypedDocumentResult, Failure> {
        let options = req.pii_output.take().unwrap_or_default();
        let operation_key = req.operation_key.clone();
        if form == Form::Ciphertext {
            return Err(Failure::new(
                crate::api::TURBO_E_UNSUPPORTED_OPTION,
                0,
                "pii_output.values PII_VALUE_FORM_CIPHERTEXT waits for #1885".to_owned(),
            ));
        }
        if !policy.values.contains(form) {
            return Err(Failure::new(
                TURBO_E_INVALID_STATE,
                0,
                format!("pii_output.values {} is not allowed by this server's PII output policy", form.name()),
            ));
        }
        // Causes that do not depend on the document are refused before the
        // graph is read, so they tell the caller nothing about it (#1931).
        if operation_key.is_empty() {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                OPERATION_KEY_FIELD,
                "pii_output.values needs operation_key: a release is recorded under it".to_owned(),
            ));
        }
        let grant = self.grants.of(service.as_ref());
        if !grant.allows_any(form) {
            return Err(Failure::new(
                TURBO_E_INVALID_STATE,
                0,
                format!("pii_output.values {}: this service is granted that form for no kind", form.name()),
            ));
        }
        let redaction = policy::select(req.pii_policy.as_ref(), &self.policies, &*self.resolver)?;
        req.pii_output = Some(PiiOutputOptions { values: PiiValueForm::Unspecified as i32, ..options });
        let doc_id = req.doc_id.clone();
        let g = grant.clone();
        let r = redaction.clone();
        let require_recorded_policy = self.require_recorded_policy;
        let (mut typed, plan) = tokio::task::spawn_blocking(move || {
            plan::project(&req, policy, Forms::of(form), &g, r.as_deref(), require_recorded_policy)
        })
        .await
        .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("the inspect task failed: {e}")))??;
        // Nothing granted in this document: the mentions carry their reasons.
        if plan.is_empty() {
            return Ok(typed);
        }
        let Some(service) = service else {
            return Err(Failure::new(TURBO_E_INTERNAL, 0, "a release planned without a service".to_owned()));
        };
        // The attempt is the one the answer's receipt names (#1908).
        let attempt_id = typed.receipt.as_ref().map(|r| r.attempt_id.clone()).unwrap_or_default();
        let mentions: Vec<_> = plan.iter().map(|p| p.audit(&typed)).collect();
        let mut effective: Vec<PiiAccessKindGrant> = Vec::new();
        for p in &plan {
            let builtin = p.kind.builtin_id().map_or(0, u32::from);
            if !effective.iter().any(|e| e.builtin_kind == builtin && e.custom_kind == p.custom) {
                let forms = p.forms.iter().map(|f| f.audit() as i32).collect();
                effective.push(PiiAccessKindGrant { builtin_kind: builtin, custom_kind: p.custom.clone(), forms });
            }
        }
        let policy_digest = policy_digest(policy, &grant, redaction.as_ref().map(|r| &r.digest)).to_vec();
        let input_digest = typed.input.as_ref().map(|i| i.source_text_digest.clone()).unwrap_or_default();
        let event_id = audit::event_id(&audit::EventKey {
            service_fingerprint: &service.0,
            operation_key: &operation_key,
            input_digest: &input_digest,
            mentions: &mentions,
            effective: &effective,
            policy_digest: &policy_digest,
        });
        let event = PiiAccessEvent {
            event_id,
            operation_key,
            attempt_id,
            input_digest,
            service_fingerprint: service.0.to_vec(),
            effective,
            mentions,
            policy_id: POLICY_ID.into(),
            policy_digest,
            time_unix_nanos: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64),
            doc_id,
        };
        audit::check_event(&event).map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e))?;
        self.audit.record(event).await.map_err(|why| {
            Failure::new(TURBO_E_BUSY, 0, format!("PII value release refused, no value released: {why}"))
        })?;
        plan::fill(&mut typed, plan, self.token.as_ref())?;
        if cfg!(debug_assertions) {
            crate::results::typed::validate_typed_document(&typed)?;
        }
        Ok(typed)
    }
}

/// The form a request asks values in, when the release path answers it:
/// a form other than withheld, on a request whose options parse.
pub fn asked(req: &InspectDocumentGraphRequest) -> Option<Form> {
    let o = req.pii_output.as_ref()?;
    Form::of_request(PiiValueForm::try_from(o.values).ok()?)
}

/// The SHA-256 of the rules a release was decided under: the server's
/// output policy, the calling service's grant and, when the request named
/// one, the digest of the redaction policy whose release ceilings applied
/// (#1943). Without one the rules, and so the digest, are as before.
pub fn policy_digest(policy: PiiOutputPolicy, grant: &caps::Grant, redaction: Option<&PolicyDigest>) -> [u8; 32] {
    let mut rules = format!(
        "rewritten-text={};full-text={};detail={};values={:?};{}",
        policy.rewritten_text,
        policy.full_text,
        policy.detail,
        policy.values.iter().map(Form::name).collect::<Vec<_>>(),
        grant.canonical()
    );
    if let Some(d) = redaction {
        rules.push_str(";redaction-policy=");
        rules.push_str(&d.to_hex());
    }
    sha256::digest(rules.as_bytes())
}

#[cfg(test)]
mod tests;
