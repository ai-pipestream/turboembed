//! AWS Signature Version 4 for one S3 GET with no query and no body, the
//! only request an S3 source makes. HMAC-SHA-256 is the RFC 2104
//! construction over `sha2`, so no AWS SDK or HMAC crate is needed.

use sha2::{Digest, Sha256};

/// The SHA-256 of an empty body, the payload hash of every GET.
pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// An access key, its secret, and for temporary credentials the session
/// token.
#[derive(Clone)]
pub struct Credentials {
    pub key: String,
    pub secret: String,
    pub token: Option<String>,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials").field("key", &self.key).finish_non_exhaustive()
    }
}

/// HMAC-SHA-256 of `data` under `key` (RFC 2104).
pub fn hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let (mut inner, mut outer) = (Sha256::new(), Sha256::new());
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(data);
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    outer.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `secs` after the Unix epoch as SigV4 writes it: `20130524T000000Z`.
pub fn amz_date(secs: u64) -> String {
    let (days, rest) = (secs / 86_400, secs % 86_400);
    // Days to a civil date (H. Hinnant's algorithm), for dates after 1970.
    let z = days + 719_468;
    let (era, doe) = (z / 146_097, z % 146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z", rest / 3600, rest % 3600 / 60, rest % 60)
}

/// The headers that sign a GET of `path` (already URI-encoded) on `host`
/// (as the Host header carries it) in `region`, at `date` ([`amz_date`]):
/// `x-amz-date`, `x-amz-content-sha256`, the session token when there is
/// one, and `authorization`. `extra` are further headers to sign and send,
/// named in lower case.
pub fn sign_get(
    creds: &Credentials,
    region: &str,
    host: &str,
    path: &str,
    extra: &[(&str, &str)],
    date: &str,
) -> Vec<(String, String)> {
    let mut signed: Vec<(String, String)> =
        vec![("x-amz-content-sha256".into(), EMPTY_SHA256.into()), ("x-amz-date".into(), date.into())];
    if let Some(token) = &creds.token {
        signed.push(("x-amz-security-token".into(), token.clone()));
    }
    signed.extend(extra.iter().map(|(n, v)| ((*n).to_owned(), (*v).to_owned())));
    let mut all = signed.clone();
    all.push(("host".into(), host.into()));
    all.sort();
    let canonical_headers: String = all.iter().map(|(n, v)| format!("{n}:{}\n", v.trim())).collect();
    let names = all.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(";");
    let request = format!("GET\n{path}\n\n{canonical_headers}\n{names}\n{EMPTY_SHA256}");
    let scope = format!("{}/{region}/s3/aws4_request", &date[..8]);
    let to_sign = format!("AWS4-HMAC-SHA256\n{date}\n{scope}\n{}", hex(&Sha256::digest(request.as_bytes())));
    let mut key = hmac(format!("AWS4{}", creds.secret).as_bytes(), &date.as_bytes()[..8]);
    for part in [region, "s3", "aws4_request"] {
        key = hmac(&key, part.as_bytes());
    }
    let signature = hex(&hmac(&key, to_sign.as_bytes()));
    signed.push((
        "authorization".into(),
        format!("AWS4-HMAC-SHA256 Credential={}/{scope},SignedHeaders={names},Signature={signature}", creds.key),
    ));
    signed
}

#[cfg(test)]
mod tests {
    use super::{Credentials, amz_date, hex, hmac, sign_get};

    #[test]
    fn hmac_matches_rfc_4231() {
        // Test case 2, and test case 6 for a key longer than a block.
        let got = hex(&hmac(b"Jefe", b"what do ya want for nothing?"));
        assert_eq!(got, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
        let long = hmac(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First");
        assert_eq!(hex(&long), "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    }

    #[test]
    fn dates_are_written_as_sigv4_writes_them() {
        assert_eq!(amz_date(0), "19700101T000000Z");
        assert_eq!(amz_date(1_369_353_600), "20130524T000000Z");
        assert_eq!(amz_date(951_782_400 + 3661), "20000229T010101Z");
        assert_eq!(amz_date(1_798_761_599), "20261231T235959Z");
    }

    #[test]
    fn a_get_is_signed_as_the_aws_s3_documentation_example() {
        // "Example: GET Object" in the S3 API reference's SigV4 header
        // examples.
        let creds = Credentials {
            key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            token: None,
        };
        let headers = sign_get(
            &creds,
            "us-east-1",
            "examplebucket.s3.amazonaws.com",
            "/test.txt",
            &[("range", "bytes=0-9")],
            "20130524T000000Z",
        );
        let auth = &headers.iter().find(|(n, _)| n == "authorization").unwrap().1;
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,\
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,\
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
        assert!(!format!("{creds:?}").contains("EXAMPLEKEY"), "the secret is never printed");
    }
}
