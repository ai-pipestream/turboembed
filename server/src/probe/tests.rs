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
    assert_eq!(p, Probe { addr: "127.0.0.1:8081".parse().unwrap(), service: String::new(), timeout: DEFAULT_TIMEOUT });
    let p = parse(&["--addr", "127.0.0.2:7", "--service=bge", "--timeout-ms", "250"], Some("0.0.0.0:8081")).unwrap();
    assert_eq!(p.addr, "127.0.0.2:7".parse().unwrap());
    assert_eq!(p.service, "bge");
    assert_eq!(p.timeout, Duration::from_millis(250));
}

#[test]
fn bad_arguments_are_refused() {
    assert_eq!(parse(&[], None).unwrap_err(), "--addr or TURBO_KSERVE_LISTEN is required");
    assert_eq!(parse(&[], Some("nowhere")).unwrap_err(), "TURBO_KSERVE_LISTEN=nowhere: not ADDR:PORT");
    assert_eq!(parse(&["--addr", "x"], None).unwrap_err(), "--addr x: not ADDR:PORT");
    assert_eq!(parse(&["--addr"], None).unwrap_err(), "--addr needs a value");
    assert_eq!(parse(&["--timeout-ms", "0"], Some("127.0.0.1:1")).unwrap_err(), "--timeout-ms 0: not a count over 0");
    assert_eq!(parse(&["--service", "a", "--service", "b"], Some("127.0.0.1:1")).unwrap_err(), "--service given twice");
    assert_eq!(parse(&["--nope", "1"], Some("127.0.0.1:1")).unwrap_err(), "unknown argument --nope");
}

#[tokio::test]
async fn nothing_listening_is_an_error_not_a_hang() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    // Windows retries a refused loopback connect for about two seconds
    // before it fails, so a two second timeout raced the refusal. The
    // refusal still comes long before this one; only a hang reaches it.
    let p = Probe { addr, service: String::new(), timeout: Duration::from_secs(30) };
    assert!(check(&p).await.unwrap_err().starts_with(&format!("connect {addr}")));
}
