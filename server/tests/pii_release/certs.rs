//! The mTLS fixtures of tls_metrics_deadline.rs (a test server CA and
//! localhost certificate, a test client CA and one client certificate),
//! and the client certificate's fingerprint.

pub const SERVER_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASqgAwIBAgIUAVogSpXO+xbeQ4x9o1xRs9pbHo8wCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMB8xHTAbBgNVBAMMFHR1cmJvLXRlc3Qtc2VydmVy\n\
LWNhMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9sFFiocjmxKXQ0kCtYZ7tmSY\n\
4hYeE1i66YfvRcbSiufRzGTY7h3xjboIymPNxoeEgvgk1231MfGd0vS3Xc7Qk6NC\n\
MEAwHQYDVR0OBBYEFOFpRbGCcyAKmJ0s5z3Ahii6GYDdMA8GA1UdEwEB/wQFMAMB\n\
Af8wDgYDVR0PAQH/BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQCYhj6qYO018b+k\n\
THo9t4rdisRMQVK5qXp1b3TTiqwu0wIgPFW9G8RUUEgj76ET9gx6m2PWA+7QmoHs\n\
czkEzHeFhYU=\n\
-----END CERTIFICATE-----\n";

pub const SERVER_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASugAwIBAgIUMDz+0icDFMibjkseSNPtVqwKB3MwCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMBQxEjAQBgNVBAMMCWxvY2FsaG9zdDBZMBMGByqG\n\
SM49AgEGCCqGSM49AwEHA0IABOi+ACPVzeNQULfW6uWG0DxJwUaESso6RcyeDKLF\n\
i2luj3jexu9SFUiZAXxeNuiYmD2NblgiY75pp/D2SNGD2BajTjBMMAkGA1UdEwQC\n\
MAAwDgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMBoGA1UdEQQT\n\
MBGCCWxvY2FsaG9zdIcEfwAAATAKBggqhkjOPQQDAgNHADBEAiAyBIr/0nJz51E0\n\
0JHXUsmLmZKA3KgLL/mZu33rcPWbYwIgAgeirtnliflpsQbdl0I3BqLVt3Jo8PuU\n\
3eF5aNpH2eI=\n\
-----END CERTIFICATE-----\n";

pub const SERVER_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgXTzexqw3bQIvfaiE\n\
kQgrSussEkBOWa4hXiu6rEivRoahRANCAATovgAj1c3jUFC31urlhtA8ScFGhErK\n\
OkXMngyixYtpbo943sbvUhVImQF8XjbomJg9jW5YImO+aafw9kjRg9gW\n\
-----END PRIVATE KEY-----\n";

pub const CLIENT_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASqgAwIBAgIUCwI4eaomEa2NQucISG+RmGkANgswCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1jbGllbnQtY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMB8xHTAbBgNVBAMMFHR1cmJvLXRlc3QtY2xpZW50\n\
LWNhMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8jVI3Hds8oK1Yg/bFVmhD4on\n\
IicKOIljXpmgDbkWCrdQ+vGIWz+QGzaogoxPSEEgSJh2wiXDGjvFOnZEHhfL/qNC\n\
MEAwHQYDVR0OBBYEFBrce0+O9No7TUNZYFQdhqUK2y16MA8GA1UdEwEB/wQFMAMB\n\
Af8wDgYDVR0PAQH/BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQDjkLhdXJhECicj\n\
CqsrrCAAVxBJs/fN0e1ltw/WdwCFewIgBJYooOLNq6lwZDv3cGk8x22fQIQCMmVK\n\
+48ct49bQE0=\n\
-----END CERTIFICATE-----\n";

pub const CLIENT_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBcDCCARegAwIBAgIUMQ8BSBhmrVSddhRHRzewospjnBUwCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1jbGllbnQtY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMBwxGjAYBgNVBAMMEXR1cmJvLXRlc3QtY2xpZW50\n\
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE4hKHzD/yw8WF02D3olK6XqebknlB\n\
IweseYnw8uJQ42Lureyzoewzgu/abKz2w48EWVasOawgDU92BIetnHODX6MyMDAw\n\
CQYDVR0TBAIwADAOBgNVHQ8BAf8EBAMCB4AwEwYDVR0lBAwwCgYIKwYBBQUHAwIw\n\
CgYIKoZIzj0EAwIDRwAwRAIgIQPpzjXnFD9yNfeudzAerg4a7ec2T1jCANbPtUDw\n\
MKkCIFas6p9vhLm1k80xLmLuJtbOEJFhHhhEaSnv4KGmlTy1\n\
-----END CERTIFICATE-----\n";

pub const CLIENT_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgJi4WvkLcjdwN4Jlb\n\
Os9NOaLaxBg5BVnrCfvk/9S66d+hRANCAATiEofMP/LDxYXTYPeiUrpep5uSeUEj\n\
B6x5ifDy4lDjYu6t7LOh7DOC79psrPbDjwRZVqw5rCANT3YEh62cc4Nf\n\
-----END PRIVATE KEY-----\n";

/// The DER bytes of the first certificate in `pem`.
pub fn der(pem: &str) -> Vec<u8> {
    let b64: String = pem.lines().filter(|l| !l.starts_with("-----")).collect();
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in b64.bytes().filter(|c| *c != b'=') {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        };
        acc = acc << 6 | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

/// The SHA-256 of the client certificate's DER bytes.
pub fn fingerprint_bytes() -> [u8; 32] {
    turbo_sha256::sha256(&der(CLIENT_CERT_PEM)).0
}

/// [`fingerprint_bytes`] as 64 lowercase hexadecimal digits.
pub fn client_fingerprint() -> String {
    fingerprint_bytes().iter().map(|b| format!("{b:02x}")).collect()
}
