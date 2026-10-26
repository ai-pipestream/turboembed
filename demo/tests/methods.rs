//! The demo's method table against the server's protos: every unary and
//! server-streaming method of every service is listed once, with the
//! right kind, and nothing else is. The gRPC health service is not the
//! page's to call.

use std::path::Path;

use turbo_kserve_demo::allow::METHODS;

#[test]
fn the_table_is_every_unary_and_server_streaming_method_of_the_protos() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../server/proto");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".proto"))
        .collect();
    files.sort();
    let fds = protox::compile(&files, [&dir]).unwrap();
    let mut want = Vec::new();
    for file in &fds.file {
        for service in &file.service {
            let full = format!("{}.{}", file.package(), service.name());
            if full == "grpc.health.v1.Health" {
                continue;
            }
            for m in &service.method {
                if m.client_streaming() {
                    continue;
                }
                want.push((format!("/{full}/{}", m.name()), m.server_streaming()));
            }
        }
    }
    want.sort();
    let mut got: Vec<(String, bool)> = METHODS.iter().map(|m| (m.path.to_string(), m.streaming)).collect();
    got.sort();
    assert_eq!(got, want);
}
