use super::*;

fn parse(args: &[&str], listen: Option<&str>) -> Result<Probe, String> {
    let listen = listen.map(str::to_string);
    Probe::parse_with(args.iter().map(|s| s.to_string()), |k| if k == ENV_LISTEN { listen.clone() } else { None })
}

#[test]
fn an_unspecified_listen_address_is_checked_on_loopback() {
    assert_eq!(target("0.0.0.0:8081".parse().unwrap()), "127.0.0.1:8081".parse().unwrap());
    assert_eq!(target("[::]:8081".parse().unwrap()), "[::1]:8081".parse().unwrap());
    assert_eq!(target("10.1.2.3:9000".parse().unwrap()), "10.1.2.3:9000".parse().unwrap());
}

#[test]
fn the_address_comes_from_the_flag_else_the_listen_variable() {
    let p = parse(&[], Some("0.0.0.0:8081")).unwrap();
    assert_eq!(
        p,
        Probe { addr: "127.0.0.1:8081".parse().unwrap(), service: String::new(), timeout: DEFAULT_TIMEOUT, tls: None }
    );
    let p = parse(&["--addr", "127.0.0.2:7", "--service=bge", "--timeout-ms", "250"], Some("0.0.0.0:8081")).unwrap();
    assert_eq!(p.addr, "127.0.0.2:7".parse().unwrap());
    assert_eq!(p.service, "bge");
    assert_eq!(p.timeout, Duration::from_millis(250));
}

#[test]
fn bad_arguments_are_refused() {
    assert_eq!(parse(&[], None).unwrap_err(), "--addr or TURBO_KSERVE_LISTEN is required");
    assert_eq!(
        parse(&[], Some("nowhere")).unwrap_err(),
        "TURBO_KSERVE_LISTEN: the value (7 bytes, not shown) is not ADDR:PORT"
    );
    assert_eq!(parse(&["--addr", "x"], None).unwrap_err(), "--addr: the value (1 bytes, not shown) is not ADDR:PORT");
    assert_eq!(parse(&["--addr"], None).unwrap_err(), "--addr needs a value");
    assert_eq!(
        parse(&["--timeout-ms", "0"], Some("127.0.0.1:1")).unwrap_err(),
        "--timeout-ms: the value (1 bytes, not shown) is not a count over 0"
    );
    assert_eq!(parse(&["--service", "a", "--service", "b"], Some("127.0.0.1:1")).unwrap_err(), "--service given twice");
    assert_eq!(
        parse(&["--nope", "1"], Some("127.0.0.1:1")).unwrap_err(),
        "unknown argument (6 bytes, not shown); the flags are --addr, --service, --timeout-ms, --tls-ca, --tls-cert, --tls-key and --tls-name"
    );
}

#[test]
fn a_refused_or_unknown_argument_is_described_by_its_length() {
    let key = "q8ZrVx3T0bWmJ2kYp7LdN9fHcE4sGuA6oRiK1tXyQwM=";
    let hex = "9f3a6c1e0b7d42a8e5f19c3b6d0a7e2f4c8b1d5a9e3f6c20";
    let to = Some("127.0.0.1:1");
    for s in [key, hex] {
        let flag = format!("--{s}");
        let inline = format!("--{s}=x");
        let cases: [(Vec<&str>, Option<&str>); 9] = [
            (vec![s], to),
            (vec!["--addr", "127.0.0.1:1", s], to),
            (vec![&flag], to),
            (vec![&flag, "1"], to),
            (vec![&inline], to),
            (vec!["--addr", s], None),
            (vec!["--timeout-ms", s], to),
            (vec![], Some(s)),
            (vec!["--service", "a", s], to),
        ];
        for (args, env) in cases {
            let e = parse(&args, env).unwrap_err();
            assert!(e.contains("not shown"), "{args:?}: {e:?}");
            for w in s.as_bytes().windows(8) {
                assert!(!e.contains(std::str::from_utf8(w).unwrap()), "{args:?}: {e:?} repeats part of the value");
            }
        }
    }
}

#[tokio::test]
async fn nothing_listening_is_an_error_not_a_hang() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    // Windows retries a refused loopback connect for about two seconds
    // before it fails, so a two second timeout raced the refusal. The
    // refusal still comes long before this one; only a hang reaches it.
    let p = Probe { addr, service: String::new(), timeout: Duration::from_secs(30), tls: None };
    assert!(check(&p).await.unwrap_err().starts_with(&format!("connect {addr}")));
}
