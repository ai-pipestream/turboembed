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
use crate::proto::audit::{PiiAccessEvent, PiiAccessMention, RecordPiiAccessRequest};

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

/// The event id of a release: 64 lowercase hexadecimal digits of the
/// SHA-256 over the operation key, the attempt and the mentions in
/// (layer, index) order. The time, the service and the document id are
/// not part of it, so a retry of one release has one id.
pub fn event_id(operation_key: &str, attempt: u32, mentions: &[PiiAccessMention]) -> String {
    let mut sorted: Vec<&PiiAccessMention> = mentions.iter().collect();
    sorted.sort_by(|a, b| (&a.layer_instance, a.index).cmp(&(&b.layer_instance, b.index)));
    let mut h = Sha256::default();
    h.update(EVENT_DOMAIN);
    let bytes = |h: &mut Sha256, b: &[u8]| {
        h.update(&(b.len() as u64).to_be_bytes());
        h.update(b);
    };
    bytes(&mut h, operation_key.as_bytes());
    h.update(&attempt.to_be_bytes());
    h.update(&(sorted.len() as u64).to_be_bytes());
    for m in sorted {
        bytes(&mut h, m.layer_instance.as_bytes());
        h.update(&m.index.to_be_bytes());
        h.update(&m.builtin_kind.to_be_bytes());
        bytes(&mut h, m.custom_kind.as_bytes());
        h.update(&m.form.to_be_bytes());
    }
    h.finish().iter().map(|b| format!("{b:02x}")).collect()
}
