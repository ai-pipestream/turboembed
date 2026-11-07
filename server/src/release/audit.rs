//! The acknowledged audit call (proto/turbo_audit.proto, #1909). Unlike
//! the sink, which drops items when its queue is full, a release waits
//! for the audit service to answer its event's id, and fails when the
//! call fails, times out or answers another id. The event names mentions
//! by layer and index and carries no value.

use std::fmt;
use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use turbo_pii::checksum::sha256::Sha256;

use crate::proto::audit::turbo_audit_service_client::TurboAuditServiceClient;
use crate::proto::audit::{PiiAccessEvent, PiiAccessKindGrant, PiiAccessMention, RecordPiiAccessRequest};

/// `--pii-audit-timeout-ms` when not given.
pub const DEFAULT_AUDIT_TIMEOUT: Duration = Duration::from_secs(5);

/// What the event id is a SHA-256 over, first.
const EVENT_DOMAIN: &[u8] = b"turbo-pii-access-v1\0";

/// Why an audit record failed. None of them carries event contents.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AuditFailure {
    /// The call ended with this gRPC status code.
    Call(tonic::Code),
    /// No answer within the audit timeout.
    Timeout,
    /// The service acknowledged another event id.
    Mismatch,
}

impl fmt::Display for AuditFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditFailure::Call(code) => write!(f, "the audit call failed with {code:?}"),
            AuditFailure::Timeout => f.write_str("the audit call timed out"),
            AuditFailure::Mismatch => f.write_str("the audit service acknowledged another event"),
        }
    }
}

/// The audit service's client: one lazily connected channel.
pub struct AuditClient {
    channel: Channel,
    timeout: Duration,
}

impl AuditClient {
    /// A client for `endpoint`. An endpoint that is not a URI is refused;
    /// one that does not answer fails each release that needs it.
    pub fn new(endpoint: &str, timeout: Duration) -> Result<AuditClient, String> {
        let ep = Endpoint::from_shared(endpoint.to_owned())
            .map_err(|e| format!("--pii-audit-endpoint {endpoint}: {e}"))?
            .connect_timeout(timeout);
        Ok(AuditClient { channel: ep.connect_lazy(), timeout })
    }

    /// Records `event` and waits for its acknowledgement.
    pub async fn record(&self, event: PiiAccessEvent) -> Result<(), AuditFailure> {
        let id = event.event_id.clone();
        let mut client = TurboAuditServiceClient::new(self.channel.clone());
        let call = client.record_pii_access(RecordPiiAccessRequest { event: Some(event) });
        match tokio::time::timeout(self.timeout, call).await {
            Err(_) => Err(AuditFailure::Timeout),
            Ok(Err(status)) => Err(AuditFailure::Call(status.code())),
            Ok(Ok(ack)) if ack.get_ref().event_id == id => Ok(()),
            Ok(Ok(_)) => Err(AuditFailure::Mismatch),
        }
    }
}

/// What a retry of one release keeps, which the event id is a digest
/// over. The attempt, the time, the service and the document id are not
/// part of it, so every attempt at one release has one id.
pub struct EventKey<'a> {
    pub operation_key: &'a str,
    /// The SHA-256 of the source text the mentions index into.
    pub input_digest: &'a [u8],
    pub mentions: &'a [PiiAccessMention],
    pub effective: &'a [PiiAccessKindGrant],
    pub policy_digest: &'a [u8],
}

/// The event id of a release: 64 lowercase hexadecimal digits of the
/// SHA-256 over `key`, with mentions in (layer, index) order and kinds in
/// (built-in id, custom name) order.
pub fn event_id(key: &EventKey<'_>) -> String {
    let mut mentions: Vec<&PiiAccessMention> = key.mentions.iter().collect();
    mentions.sort_by(|a, b| (&a.layer_instance, a.index).cmp(&(&b.layer_instance, b.index)));
    let mut effective: Vec<&PiiAccessKindGrant> = key.effective.iter().collect();
    effective.sort_by(|a, b| (a.builtin_kind, &a.custom_kind).cmp(&(b.builtin_kind, &b.custom_kind)));
    let mut h = Sha256::default();
    h.update(EVENT_DOMAIN);
    let bytes = |h: &mut Sha256, b: &[u8]| {
        h.update(&(b.len() as u64).to_be_bytes());
        h.update(b);
    };
    bytes(&mut h, key.operation_key.as_bytes());
    bytes(&mut h, key.input_digest);
    h.update(&(mentions.len() as u64).to_be_bytes());
    for m in mentions {
        bytes(&mut h, m.layer_instance.as_bytes());
        h.update(&m.index.to_be_bytes());
        h.update(&m.builtin_kind.to_be_bytes());
        bytes(&mut h, m.custom_kind.as_bytes());
        h.update(&m.form.to_be_bytes());
    }
    h.update(&(effective.len() as u64).to_be_bytes());
    for e in effective {
        h.update(&e.builtin_kind.to_be_bytes());
        bytes(&mut h, e.custom_kind.as_bytes());
        h.update(&(e.forms.len() as u64).to_be_bytes());
        for f in &e.forms {
            h.update(&f.to_be_bytes());
        }
    }
    bytes(&mut h, key.policy_digest);
    h.finish().iter().map(|b| format!("{b:02x}")).collect()
}
