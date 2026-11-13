//! The error for an argument no flag claims. It never repeats a value: a
//! misspelled `--flag=value`, or a value whose flag is missing, may be
//! key text pasted in the wrong place. Even after `--`, the name is
//! repeated only when it is a near-miss of a known flag; lowercase hex key
//! material has the same character set as a flag name (#2133).

const KNOWN_FLAGS: &[&str] = &[
    "--admin",
    "--admin-bearer-token",
    "--bearer-token",
    "--bundle-dir",
    "--catalog-data",
    "--catalog-dir",
    "--catalog-index",
    "--catalog-prefetch",
    "--catalog-store",
    "--listen",
    "--max-message-bytes",
    "--max-stream-bytes",
    "--metrics-listen",
    "--model",
    "--pii-audit-endpoint",
    "--pii-audit-timeout-ms",
    "--pii-encryption-key",
    "--pii-encryption-key-active",
    "--pii-grant",
    "--pii-output-allow",
    "--pii-policy",
    "--pii-require-recorded-policy",
    "--pii-token-key",
    "--pii-token-key-active",
    "--sink-documents",
    "--sink-endpoint",
    "--sink-pii-output",
    "--sink-queue-bytes",
    "--sink-queue-items",
    "--sink-typed-offsets",
    "--sink-typed-view",
    "--tls-cert",
    "--tls-client-ca",
    "--tls-key",
    "--training-dir",
    "--vocabulary-dir",
    "--zoo",
];

/// `flag` is the part before any `=`; `inline` says whether a value
/// followed it.
pub(crate) fn unknown_argument(flag: &str, inline: bool) -> String {
    let Some(name) = flag.strip_prefix("--") else {
        return format!("unknown argument ({} bytes, not a --flag, not shown)", flag.len());
    };
    // Every flag is lower case letters, digits and `-`, and only a near
    // typo of a known flag is repeated. Anything else after `--` (a
    // base64 key or a lower case hex secret) is described by length only.
    let flag_shaped = !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if !flag_shaped || !close_to_known_flag(flag) {
        return format!("unknown argument ({} bytes, not shaped like a --flag, not shown)", flag.len());
    }
    match inline {
        true => format!("unknown argument {flag}=... (the value is not shown)"),
        false => format!("unknown argument {flag}"),
    }
}

fn close_to_known_flag(flag: &str) -> bool {
    KNOWN_FLAGS.iter().any(|known| edit_distance_at_most_two(flag, known))
}

fn edit_distance_at_most_two(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len().abs_diff(b.len()) > 2 {
        return false;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()] <= 2
}

#[cfg(test)]
mod tests {
    use super::super::Args;
    use super::KNOWN_FLAGS;

    fn err(args: &[&str]) -> String {
        Args::parse_with(args.iter().map(|a| a.to_string()), |_| None).unwrap_err()
    }

    #[test]
    fn an_unknown_argument_is_named_without_its_value() {
        let key = "q8ZrVx3T0bWmJ2kYp7LdN9fHcE4sGuA6oRiK1tXyQwM=";
        let hex = "9f3a1c8e7b2d4f6a0e5c9b3d7f1a8e2c4b6d0f9a3e5c7b1d";
        assert_eq!(hex.len(), 48);
        let base = ["--listen", "127.0.0.1:0"];
        let cases = [
            (format!("--pii-token-keys=tok:1:{key}"), "unknown argument --pii-token-keys=... (the value is not shown)"),
            (
                format!("--pii-encryption-keys={key}"),
                "unknown argument --pii-encryption-keys=... (the value is not shown)",
            ),
            (key.to_owned(), "unknown argument (44 bytes, not a --flag, not shown)"),
            (format!("tok:1:{key}"), "unknown argument (50 bytes, not a --flag, not shown)"),
            (format!("--{key}"), "unknown argument (45 bytes, not shaped like a --flag, not shown)"),
            (format!("--{hex}"), "unknown argument (50 bytes, not shaped like a --flag, not shown)"),
            (format!("--{hex}=x"), "unknown argument (50 bytes, not shaped like a --flag, not shown)"),
            (format!("--{}", &hex[..16]), "unknown argument (18 bytes, not shaped like a --flag, not shown)"),
        ];
        for (arg, want) in cases {
            let e = err(&[base[0], base[1], &arg]);
            assert_eq!(e, want);
            for secret in [key, hex] {
                for w in secret.as_bytes().windows(4) {
                    assert!(!e.contains(std::str::from_utf8(w).unwrap()), "{e:?} repeats part of the input");
                }
            }
        }
        // A misspelled flag with no value is still named, so a typo is easy
        // to find.
        assert_eq!(err(&["--lisen", "127.0.0.1:0"]), "unknown argument --lisen");
        assert_eq!(err(&["--catalog-date", "/d"]), "unknown argument --catalog-date");
        // Every known flag in KNOWN_FLAGS is claimed by the parser, and
        // every flag the parser matches is in KNOWN_FLAGS so a near-miss
        // typo of it is named.
        let mut parsed_flags = Vec::new();
        for src in
            [include_str!("../config.rs"), include_str!("auth.rs"), include_str!("sink.rs"), include_str!("release.rs")]
        {
            for line in src.lines() {
                if let Some((quoted, _)) = line.trim().split_once("=>")
                    && let Some(flag) = quoted.trim().strip_prefix("\"--").and_then(|s| s.strip_suffix('"'))
                {
                    parsed_flags.push(format!("--{flag}"));
                }
            }
        }
        parsed_flags.sort();
        assert_eq!(parsed_flags, KNOWN_FLAGS);
        for flag in &parsed_flags {
            let e = err(&[flag]);
            assert!(!e.starts_with("unknown argument"), "{flag}: {e}");
            let typo = format!("{flag}z");
            assert_eq!(err(&[&typo]), format!("unknown argument {typo}"));
        }
    }
}
