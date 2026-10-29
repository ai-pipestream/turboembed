//! An S3 bucket as a source and a store: each file read by its SHA-256
//! from `<prefix>/blobs/sha256/<hex>`, the layout a zoo server's store and
//! a library's local copy share, with one signed GET, and kept there with
//! one signed PUT.

use std::time::{SystemTime, UNIX_EPOCH};

use super::sigv4::{Credentials, EMPTY_SHA256, amz_date, payload_sha256, sign};
use super::sources::Keep;
use super::transport::Transport;
use crate::digest::Sha256;
use crate::index::{Artifact, File, Url};

/// An S3 bucket, or a prefix in one, on AWS or any S3-compatible store
/// (RustFS). The bucket holds files the way [`turbo_zoo::Blobs`]
/// does, so a store copied there (`aws s3 sync <root> s3://bucket/zoo`)
/// serves as is.
#[derive(Clone)]
pub struct S3Bucket {
    address: String,
    /// `scheme://authority`, with no path.
    base: String,
    /// The Host header the request carries and the signature covers.
    host: String,
    /// The path up to `blobs`: `/bucket/prefix` path-style, `/prefix` on a
    /// bucket's own host.
    path: String,
    region: String,
    creds: Option<Credentials>,
    agent: ureq::Agent,
}

impl S3Bucket {
    /// The bucket `s3://bucket[/prefix]` as the AWS tools name it from the
    /// environment: `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL` for a
    /// store other than AWS, `AWS_REGION` or `AWS_DEFAULT_REGION` (else
    /// `us-east-1`), and `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
    /// `AWS_SESSION_TOKEN`. With no key it reads a public bucket unsigned.
    pub fn new(address: &str) -> Result<S3Bucket, String> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let endpoint = var("AWS_ENDPOINT_URL_S3").or_else(|| var("AWS_ENDPOINT_URL"));
        let region = var("AWS_REGION").or_else(|| var("AWS_DEFAULT_REGION")).unwrap_or_else(|| "us-east-1".into());
        let creds = match (var("AWS_ACCESS_KEY_ID"), var("AWS_SECRET_ACCESS_KEY")) {
            (Some(key), Some(secret)) => Some(Credentials { key, secret, token: var("AWS_SESSION_TOKEN") }),
            (None, None) => None,
            _ => return Err(format!("{address}: set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or neither")),
        };
        S3Bucket::with(address, endpoint.as_deref(), &region, creds)
    }

    /// The bucket at `endpoint` (`http://host:port` or `https://host`,
    /// addressed path-style, as RustFS serves), or on AWS when
    /// none is given, read with `creds`, or unsigned without.
    pub fn with(
        address: &str,
        endpoint: Option<&str>,
        region: &str,
        creds: Option<Credentials>,
    ) -> Result<S3Bucket, String> {
        let bad = |why: &str| format!("{address}: {why}");
        let rest = address.strip_prefix("s3://").ok_or_else(|| bad("an S3 source starts s3://"))?;
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let prefix = prefix.trim_end_matches('/');
        if !valid_bucket(bucket) {
            return Err(bad("a bucket name is 3 to 63 lower-case letters, digits, dots and hyphens"));
        }
        if !valid_prefix(prefix) {
            return Err(bad("a prefix holds letters, digits, '.', '_', '-' and single '/' between names"));
        }
        if region.is_empty() || !region.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
            return Err(bad(&format!("{region:?} is not a region name")));
        }
        let under = if prefix.is_empty() { String::new() } else { format!("/{prefix}") };
        let (base, host, path) = match endpoint {
            Some(e) => {
                let (scheme, authority) = split_endpoint(e).ok_or_else(|| {
                    bad(&format!("the endpoint {e} is http:// or https:// with a host and port alone"))
                })?;
                (format!("{scheme}://{authority}"), authority, format!("/{bucket}{under}"))
            }
            // A dotted name breaks the bucket's own certificate, so it is
            // addressed path-style.
            None if bucket.contains('.') => {
                let host = format!("s3.{region}.amazonaws.com");
                (format!("https://{host}"), host, format!("/{bucket}{under}"))
            }
            None => {
                let host = format!("{bucket}.s3.{region}.amazonaws.com");
                (format!("https://{host}"), host, under)
            }
        };
        // A signed request is not replayed elsewhere, so no redirect is
        // followed; the bytes are asked for unencoded, as the zoo's own
        // HTTPS transport asks.
        let agent = ureq::Agent::config_builder().max_redirects(0).accept_encoding("identity").build().into();
        Ok(S3Bucket { address: address.to_owned(), base, host, path, region: region.to_owned(), creds, agent })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// The URL the file with SHA-256 `hex` is read from.
    pub fn url_of(&self, hex: &str) -> String {
        format!("{}{}/blobs/sha256/{hex}", self.base, self.path)
    }

    /// `method` on the object for `hex`, signed when there are credentials.
    fn request(&self, method: &str, hex: &str, payload: &str) -> Result<(String, Vec<(String, String)>), String> {
        let path = format!("{}/blobs/sha256/{hex}", self.path);
        let url = format!("{}{path}", self.base);
        let Some(creds) = &self.creds else { return Ok((url, Vec::new())) };
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs();
        let headers = sign(method, creds, &self.region, (&self.host, &path), payload, &[], &amz_date(now))?;
        Ok((url, headers))
    }

    /// Keeps `bytes` in the bucket under their own SHA-256, with one PUT,
    /// and returns that SHA-256.
    pub fn put(&self, bytes: &[u8]) -> Result<Sha256, String> {
        let sha256 = Sha256::of(bytes);
        let (url, headers) = self.request("PUT", &sha256.to_string(), &payload_sha256(bytes))?;
        let mut req = self.agent.put(&url);
        for (name, value) in headers {
            req = req.header(name, value);
        }
        req.send(bytes).map_err(|e| format!("{url}: {e}"))?;
        Ok(sha256)
    }
}

impl Keep for S3Bucket {
    fn keep(&self, bytes: &[u8]) -> Result<(), String> {
        self.put(bytes).map(|_| ())
    }
}

impl Transport for S3Bucket {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: an S3 source serves a file by its SHA-256, not its address"))
    }

    /// The file, unpacked, as the store keeps it, at most its size and one
    /// byte more so the zoo's size check fails on a longer one.
    fn get_file(&self, _artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        let (url, headers) = self.request("GET", &file.sha256.to_string(), EMPTY_SHA256)?;
        let mut req = self.agent.get(&url);
        for (name, value) in headers {
            req = req.header(name, value);
        }
        let mut resp = req.call().map_err(|e| format!("{url}: {e}"))?;
        resp.body_mut()
            .with_config()
            .limit(file.size.saturating_add(1))
            .read_to_vec()
            .map_err(|e| format!("{url}: {e}"))
    }
}

/// S3's bucket naming rules, less the ones a name breaks only on AWS.
fn valid_bucket(name: &str) -> bool {
    let b = name.as_bytes();
    (3..=63).contains(&b.len())
        && b.iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'.' || c == b'-')
        && b[0].is_ascii_alphanumeric()
        && b[b.len() - 1].is_ascii_alphanumeric()
        && !name.contains("..")
}

/// Names that need no URI encoding, so the signed path is the sent path.
fn valid_prefix(prefix: &str) -> bool {
    prefix.is_empty()
        || prefix.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
        })
}

/// `http://host:port` or `https://host` as (scheme, authority), the
/// authority as the Host header carries it: a default port is dropped.
fn split_endpoint(endpoint: &str) -> Option<(&'static str, String)> {
    let (scheme, rest) = if let Some(r) = endpoint.strip_prefix("https://") {
        ("https", r)
    } else {
        ("http", endpoint.strip_prefix("http://")?)
    };
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.is_empty() || authority.contains(['/', '?', '#', '@']) {
        return None;
    }
    let default = if scheme == "https" { ":443" } else { ":80" };
    Some((scheme, authority.strip_suffix(default).unwrap_or(authority).to_ascii_lowercase()))
}
