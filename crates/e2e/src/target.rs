//! Architecture the harness is talking to.

use std::fmt;
use std::str::FromStr;

/// One of the three arch gRPC servers, plus the local mock used in CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    Nvidia,
    Intel,
    Apple,
    /// Dev / CI only: mock backend registered under the same logical names.
    Mock,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Intel => "intel",
            Self::Apple => "apple",
            Self::Mock => "mock",
        }
    }

    /// Default `host:port` for a live worker. The harness never starts these.
    pub fn default_addr(self) -> &'static str {
        match self {
            Self::Nvidia => "krick:8461",
            Self::Intel => "krick-1:8461",
            Self::Apple => "krickert-mac:8461",
            Self::Mock => "127.0.0.1:8461",
        }
    }

    pub fn is_mock(self) -> bool {
        matches!(self, Self::Mock)
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Target {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "nvidia" => Ok(Self::Nvidia),
            "intel" => Ok(Self::Intel),
            "apple" => Ok(Self::Apple),
            "mock" => Ok(Self::Mock),
            other => Err(format!(
                "unknown target {other:?}; expected nvidia|intel|apple|mock"
            )),
        }
    }
}

/// Guess a target from a host:port when the caller did not set one.
pub fn infer_target_from_addr(addr: &str) -> Target {
    let host = addr
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(addr)
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    if host.contains("krick-1") {
        Target::Intel
    } else if host.contains("krickert-mac") || host.ends_with("-mac") {
        Target::Apple
    } else if host.contains("krick") {
        Target::Nvidia
    } else if host == "127.0.0.1" || host == "localhost" || host == "::1" {
        Target::Mock
    } else {
        Target::Nvidia
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_live_hosts() {
        assert_eq!(infer_target_from_addr("krick:8461"), Target::Nvidia);
        assert_eq!(infer_target_from_addr("krick-1:8461"), Target::Intel);
        assert_eq!(infer_target_from_addr("krickert-mac:8461"), Target::Apple);
        assert_eq!(infer_target_from_addr("127.0.0.1:8461"), Target::Mock);
    }
}
