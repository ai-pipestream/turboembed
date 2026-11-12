//! The error for a value a flag or variable refused. It never repeats the
//! value: text pasted into the wrong flag or variable may be key material,
//! so the error names the flag and says how long the value was.

/// `name` is the flag or variable, `wanted` what it takes.
pub(crate) fn refused(name: &str, value: &str, wanted: &str) -> String {
    format!("{name}: the value ({} bytes, not shown) is not {wanted}", value.len())
}

/// The `0` to `4294967295` setting `k` of one `--model` value.
pub(super) fn number(k: &str, v: &str) -> Result<u32, String> {
    v.parse().map_err(|_| refused(&format!("--model {k}"), v, "a number from 0 to 4294967295"))
}

/// What a yes or no variable takes.
pub(super) const SWITCH: &str = "one of 1, true, yes, 0, false, no";

/// The settings one `--model` value takes.
pub(super) const MODEL_SETTINGS: &str = "bundle, name, device, precision, max_batch, max_seq or sessions";

#[cfg(test)]
mod tests {
    use super::super::*;

    const KEY: &str = "q8ZrVx3T0bWmJ2kYp7LdN9fHcE4sGuA6oRiK1tXyQwM=";
    const HEX: &str = "9f3a6c1e0b7d42a8e5f19c3b6d0a7e2f4c8b1d5a9e3f6c20";
    const FP: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    const BASE: &str = "--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1";

    fn err(args: &str, env: &[(&str, &str)]) -> String {
        let env = |k: &str| env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        Args::parse_with(args.split_whitespace().map(String::from), env).unwrap_err()
    }

    fn assert_hidden(e: &str, secret: &str, context: &str) {
        assert!(e.contains("not shown"), "{context}: {e:?}");
        for w in secret.as_bytes().windows(8) {
            let w = std::str::from_utf8(w).unwrap();
            assert!(!e.contains(w), "{context}: {e:?} repeats part of the value");
        }
    }

    #[test]
    fn a_refused_flag_value_is_described_by_its_length() {
        let sink = "--sink-endpoint http://a:1";
        let grant = format!(
            "--admin --tls-cert c.pem --tls-key k.pem --tls-client-ca ca.pem \
                             --pii-grant {FP}:token:email --pii-audit-endpoint http://a:9"
        );
        for s in [KEY, HEX] {
            let cases = [
                format!("--model bundle=/b/a,device=0,sessions=1 --listen {s}"),
                format!("{BASE} --metrics-listen {s}"),
                format!("{BASE} --max-message-bytes {s}"),
                format!("{BASE} --max-stream-bytes={s}"),
                format!("{BASE} --catalog-dir /z --catalog-prefetch {s}"),
                format!("--listen 127.0.0.1:0 --model {s}"),
                format!("--listen 127.0.0.1:0 --model {s}=1"),
                format!("--listen 127.0.0.1:0 --model bundle=/b/a,device={s},sessions=1"),
                format!("--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions={s}"),
                format!("--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1,precision={s}"),
                format!("--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1,max_seq={s}"),
                format!("{BASE} {sink} --sink-queue-items {s}"),
                format!("{BASE} {sink} --sink-queue-bytes {s}"),
                format!("{BASE} {sink} --sink-typed-offsets {s}"),
                format!("{BASE} {sink} --sink-typed-offsets utf8 --sink-typed-view {s}"),
                format!("{BASE} {sink} --sink-pii-output {s}"),
                format!("{BASE} {sink} --sink-pii-output detail,{s}"),
                format!("{BASE} --pii-output-allow {s}"),
                format!("{BASE} --pii-output-allow token,{s}"),
                format!("{BASE} {grant} --pii-audit-timeout-ms {s}"),
            ];
            for a in &cases {
                assert_hidden(&err(a, &[]), s, a);
            }
            let vars = [
                (ENV_LISTEN, "--model bundle=/b/a,device=0,sessions=1"),
                (ENV_METRICS_LISTEN, BASE),
                (ENV_MAX_MESSAGE_BYTES, BASE),
                (ENV_MAX_STREAM_BYTES, BASE),
                (ENV_ADMIN, BASE),
                (ENV_ZOO, BASE),
                (ENV_MODELS, "--listen 127.0.0.1:0"),
                (ENV_CATALOG_PREFETCH, "--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1 --catalog-dir /z"),
                (ENV_SINK_DOCUMENTS, &format!("{BASE} {sink}")),
                (ENV_PII_REQUIRE_RECORDED_POLICY, &format!("{BASE} {grant}")),
            ];
            for (k, a) in vars {
                assert_hidden(&err(a, &[(k, s)]), s, k);
            }
        }
    }

    #[test]
    fn a_refused_value_still_names_its_flag_and_what_it_takes() {
        let e = err(&format!("--model bundle=/b/a,device=0,sessions=1 --listen {HEX}"), &[]);
        assert_eq!(e, "--listen: the value (48 bytes, not shown) is not ADDR:PORT");
        let e = err("--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sesions=1", &[]);
        assert_eq!(
            e,
            "--model: a setting (7 bytes, not shown) is not bundle, name, device, precision, max_batch, max_seq or sessions"
        );
        let e = err(BASE, &[(ENV_ZOO, "maybe")]);
        assert_eq!(e, "TURBO_KSERVE_ZOO: the value (5 bytes, not shown) is not one of 1, true, yes, 0, false, no");
        // A setting that parsed is named: it is one of the known ones.
        let e = err("--listen 127.0.0.1:0 --model bundle=/b/a,device=0,device=1,sessions=1", &[]);
        assert_eq!(e, "--model: device given twice");
    }
}
