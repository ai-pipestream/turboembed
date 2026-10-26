//! `turbo-kserve health`: one `grpc.health.v1.Health/Check` against a
//! running server, for a container's health check where the orchestrator
//! has no gRPC probe of its own (docs/release.md, Images). It exits 0 when
//! the service answers `SERVING`, 1 when it answers anything else or does
//! not answer in time, and 2 on a usage error.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use crate::config::ENV_LISTEN;
use crate::proto::health::HealthCheckRequest;
use crate::proto::health::health_check_response::ServingStatus;
use crate::proto::health::health_client::HealthClient;

pub const HEALTH_USAGE: &str = "usage: turbo-kserve health [--addr ADDR:PORT] [--service NAME] [--timeout-ms N]\n\
the address is TURBO_KSERVE_LISTEN when absent, an unspecified one (0.0.0.0, ::) checked on loopback; \
the service is the server's own (empty) when absent; exit 0 on SERVING, 1 otherwise, 2 on a usage error";

/// How long a check waits for its answer when `--timeout-ms` is absent.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub addr: SocketAddr,
    /// The `service` of the request: empty for the server, or a model name.
    pub service: String,
    pub timeout: Duration,
}

/// Where to connect to reach a server listening on `listen`: the loopback
/// address of its family when it listens on every interface.
pub fn target(listen: SocketAddr) -> SocketAddr {
    match listen.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => SocketAddr::new(Ipv4Addr::LOCALHOST.into(), listen.port()),
        IpAddr::V6(ip) if ip.is_unspecified() => SocketAddr::new(Ipv6Addr::LOCALHOST.into(), listen.port()),
        _ => listen,
    }
}

impl Probe {
    /// The arguments after `health`, with the process environment.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Probe, String> {
        Probe::parse_with(args, |k| std::env::var(k).ok())
    }

    /// `parse` with `env` as the environment.
    pub fn parse_with(
        args: impl IntoIterator<Item = String>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Probe, String> {
        let (mut addr, mut service, mut timeout) = (None, None, None);
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            let (flag, inline) = match a.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
                _ => (a.clone(), None),
            };
            let v = inline.or_else(|| it.next()).ok_or_else(|| format!("{flag} needs a value"))?;
            let twice = || Err(format!("{flag} given twice"));
            match flag.as_str() {
                "--addr" => {
                    let a = v.parse::<SocketAddr>().map_err(|_| format!("--addr {v}: not ADDR:PORT"))?;
                    if addr.replace(a).is_some() {
                        return twice();
                    }
                }
                "--service" => {
                    if service.replace(v).is_some() {
                        return twice();
                    }
                }
                "--timeout-ms" => {
                    let ms = v.parse::<u64>().ok().filter(|&n| n > 0);
                    let ms = ms.ok_or_else(|| format!("--timeout-ms {v}: not a count over 0"))?;
                    if timeout.replace(Duration::from_millis(ms)).is_some() {
                        return twice();
                    }
                }
                _ => return Err(format!("unknown argument {a}")),
            }
        }
        let addr = match addr {
            Some(a) => a,
            None => {
                let v = env(ENV_LISTEN)
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| format!("--addr or {ENV_LISTEN} is required"))?;
                target(v.parse().map_err(|_| format!("{ENV_LISTEN}={v}: not ADDR:PORT"))?)
            }
        };
        Ok(Probe { addr, service: service.unwrap_or_default(), timeout: timeout.unwrap_or(DEFAULT_TIMEOUT) })
    }
}

/// The status the server answers for `probe.service`, or why there is none.
pub async fn check(probe: &Probe) -> Result<ServingStatus, String> {
    let ask = async {
        let mut c = HealthClient::connect(format!("http://{}", probe.addr))
            .await
            .map_err(|e| format!("connect {}: {e}", probe.addr))?;
        let r = c
            .check(HealthCheckRequest { service: probe.service.clone() })
            .await
            .map_err(|s| format!("Check {:?}: {:?} {}", probe.service, s.code(), s.message()))?;
        Ok(r.into_inner().status())
    };
    tokio::time::timeout(probe.timeout, ask)
        .await
        .map_err(|_| format!("{} gave no answer within {} ms", probe.addr, probe.timeout.as_millis()))?
}

#[cfg(test)]
mod tests;
