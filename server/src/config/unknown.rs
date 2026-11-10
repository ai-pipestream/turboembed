//! The error for an argument no flag claims. It never repeats a value: a
//! misspelled `--flag=value`, or a value whose flag is missing, may be
//! key text pasted in the wrong place.

/// `flag` is the part before any `=`; `inline` says whether a value
/// followed it.
pub(crate) fn unknown_argument(flag: &str, inline: bool) -> String {
    let Some(name) = flag.strip_prefix("--") else {
        return format!("unknown argument ({} bytes, not a --flag, not shown)", flag.len());
    };
    // Every flag is lower case letters, digits and `-`. Anything else
    // after `--` (say `--` and a base64 key) is not repeated.
    let flag_shaped = !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if !flag_shaped {
        return format!("unknown argument ({} bytes, not shaped like a --flag, not shown)", flag.len());
    }
    match inline {
        true => format!("unknown argument {flag}=... (the value is not shown)"),
        false => format!("unknown argument {flag}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::Args;

    fn err(args: &[&str]) -> String {
        Args::parse_with(args.iter().map(|a| a.to_string()), |_| None).unwrap_err()
    }

    #[test]
    fn an_unknown_argument_is_named_without_its_value() {
        let key = "q8ZrVx3T0bWmJ2kYp7LdN9fHcE4sGuA6oRiK1tXyQwM=";
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
        ];
        for (arg, want) in cases {
            let e = err(&[base[0], base[1], &arg]);
            assert_eq!(e, want);
            for w in key.as_bytes().windows(4) {
                assert!(!e.contains(std::str::from_utf8(w).unwrap()), "{e:?} repeats part of the input");
            }
        }
        // A misspelled flag with no value is still named, so a typo is easy
        // to find.
        assert_eq!(err(&["--lisen", "127.0.0.1:0"]), "unknown argument --lisen");
    }
}
