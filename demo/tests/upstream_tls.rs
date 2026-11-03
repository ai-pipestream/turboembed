//! Real gRPC TLS through the page and the health probe. No downloaded model.
#[path = "tls/certs.rs"]
mod certs;
mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::proto::health::health_check_response::ServingStatus;
use turbo_kserve::tls::TlsConfig;
use turbo_kserve::{Serve, probe};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn health_checks_a_real_tls_server() {
    let dir = certs::dir();
    let s = common::server(
        &["tiny-bert-bundle"],
        Serve {
            tls: Some(TlsConfig { cert: dir.join("cert.pem"), key: dir.join("key.pem"), client_ca: None }),
            ..Serve::default()
        },
    )
    .await;
    let args = vec![
        "--addr".into(),
        s.local_addr().to_string(),
        "--tls-ca".into(),
        dir.join("ca.pem").to_str().unwrap().into(),
    ];
    let p = probe::Probe::parse_with(args, |_| None).expect("TLS probe configuration");
    assert_eq!(probe::check(&p).await.unwrap(), ServingStatus::Serving);
    s.stop().await;
}

#[tokio::test]
async fn demo_cli_calls_a_real_tls_server() {
    let dir = certs::dir();
    let s = common::server(
        &["tiny-bert-bundle"],
        Serve {
            tls: Some(TlsConfig { cert: dir.join("cert.pem"), key: dir.join("key.pem"), client_ca: None }),
            ..Serve::default()
        },
    )
    .await;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let mut child = Process(
        Command::new(env!("CARGO_BIN_EXE_turbo-kserve-demo"))
            .args([
                "--listen",
                &addr.to_string(),
                "--server",
                &format!("https://{}", s.local_addr()),
                "--server-tls-ca",
                dir.join("ca.pem").to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let mut stream = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            assert!(child.0.try_wait().unwrap().is_none(), "demo exited before listening");
            if let Ok(stream) = TcpStream::connect(addr).await {
                break stream;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    stream.write_all(b"POST /inference.GRPCInferenceService/ServerLive HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/proto\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
    let mut reply = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), stream.read_to_end(&mut reply)).await.unwrap().unwrap();
    assert!(reply.starts_with(b"HTTP/1.1 200"), "{}", String::from_utf8_lossy(&reply));
    s.stop().await;
}

#[tokio::test]
async fn demo_and_probe_require_the_right_server_and_client_credentials() {
    use turbo_kserve_demo::{ClientIdentity, Demo, Options, UpstreamTls};
    let dir = certs::dir();
    let s = common::server(
        &["tiny-bert-bundle"],
        Serve {
            tls: Some(TlsConfig {
                cert: dir.join("cert.pem"),
                key: dir.join("key.pem"),
                client_ca: Some(dir.join("client-ca.pem")),
            }),
            ..Serve::default()
        },
    )
    .await;
    // Separate server/client CAs: possessing a server certificate grants no client access.
    for (ca, name, client, succeeds) in [
        ("ca.pem", "localhost", Some("client"), true),
        ("ca.pem", "localhost", None, false),
        ("ca.pem", "localhost", Some("rogue"), false),
        ("client-ca.pem", "localhost", Some("client"), false),
        ("ca.pem", "wrong.example", Some("client"), false),
    ] {
        let tls = UpstreamTls {
            ca: dir.join(ca),
            server_name: Some(name.into()),
            identity: client
                .map(|p| ClientIdentity { cert: dir.join(format!("{p}.pem")), key: dir.join(format!("{p}.key")) }),
        };
        let d = Demo::start_with(
            "127.0.0.1:0".parse().unwrap(),
            &format!("https://{}", s.local_addr()),
            Options { upstream_tls: Some(tls.clone()), ..Options::default() },
        )
        .await
        .unwrap();
        let (reply, body) = tokio::time::timeout(
            Duration::from_secs(20),
            common::unary(&d, "/inference.GRPCInferenceService/ServerLive", &turbo_kserve::proto::ServerLiveRequest {}),
        )
        .await
        .unwrap();
        assert_eq!(reply.status == 200, succeeds, "{ca} {name} {client:?}: {body:?}");
        if succeeds {
            use prost::Message;
            assert!(turbo_kserve::proto::ServerLiveResponse::decode(body.as_slice()).unwrap().live);
        }
        let p = probe::Probe {
            addr: s.local_addr(),
            service: String::new(),
            timeout: Duration::from_secs(20),
            tls: Some(probe::ProbeTls {
                ca: tls.ca,
                server_name: tls.server_name,
                identity: tls.identity.map(|i| probe::ClientIdentity { cert: i.cert, key: i.key }),
            }),
        };
        assert_eq!(probe::check(&p).await.is_ok(), succeeds, "probe {ca} {name} {client:?}");
        d.stop().await;
    }
    s.stop().await;
}

#[tokio::test]
async fn transport_configuration_never_silently_downgrades() {
    use turbo_kserve_demo::{Demo, Options, UpstreamTls};
    let options = Options {
        upstream_tls: Some(UpstreamTls { ca: certs::dir().join("ca.pem"), identity: None, server_name: None }),
        ..Options::default()
    };
    assert!(Demo::start_with("127.0.0.1:0".parse().unwrap(), "http://127.0.0.1:1", options).await.is_err());
    assert!(Demo::start_with("127.0.0.1:0".parse().unwrap(), "https://127.0.0.1:1", Options::default()).await.is_err());
    for (ca, cert, key, name) in [
        (Some("ca"), Some("cert"), None, None),
        (None, Some("cert"), Some("key"), None),
        (None, None, None, Some("name")),
        (Some(" "), None, None, None),
    ] {
        assert!(
            UpstreamTls::from_parts(
                ca.map(str::to_owned),
                cert.map(str::to_owned),
                key.map(str::to_owned),
                name.map(str::to_owned)
            )
            .is_err()
        );
    }
}

#[test]
fn health_tls_environment_and_flag_precedence_are_checked() {
    let env = |k: &str| match k {
        "TURBO_KSERVE_LISTEN" => Some("0.0.0.0:8081".into()),
        "TURBO_KSERVE_HEALTH_TLS_CA" => Some("env-ca".into()),
        "TURBO_KSERVE_HEALTH_TLS_CERT" => Some("cert".into()),
        "TURBO_KSERVE_HEALTH_TLS_KEY" => Some("key".into()),
        "TURBO_KSERVE_HEALTH_TLS_NAME" => Some("localhost".into()),
        _ => None,
    };
    let p = probe::Probe::parse_with(["--tls-ca=flag-ca".into()], env).unwrap();
    assert_eq!(p.addr, "127.0.0.1:8081".parse().unwrap());
    let tls = p.tls.unwrap();
    assert_eq!(tls.ca, std::path::PathBuf::from("flag-ca"));
    assert_eq!(tls.server_name.as_deref(), Some("localhost"));
    assert_eq!(tls.identity.unwrap().key, std::path::PathBuf::from("key"));
    for args in [vec!["--tls-ca="], vec!["--tls-ca=x", "--tls-ca=y"], vec!["--tls-cert=x"], vec!["--tls-name=x"]] {
        let args = [vec!["--addr=127.0.0.1:1"], args].concat().into_iter().map(str::to_owned);
        assert!(probe::Probe::parse_with(args, |_| None).is_err());
    }
}
