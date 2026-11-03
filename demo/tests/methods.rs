//! The demo's method table against the server's protos: every unary,
//! server-streaming and client-streaming method of every service is listed
//! once, with the right kind, and nothing else is: no bidi method. The
//! gRPC health service is not the page's to call.

use std::path::Path;

use turbo_kserve_demo::allow::METHODS;

#[test]
fn the_table_is_every_method_of_the_protos_but_bidi() {
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
                if m.client_streaming() && m.server_streaming() {
                    continue;
                }
                want.push((format!("/{full}/{}", m.name()), m.server_streaming(), m.client_streaming()));
            }
        }
    }
    want.sort();
    let mut got: Vec<(String, bool, bool)> =
        METHODS.iter().map(|m| (m.path.to_string(), m.streaming, m.upload)).collect();
    got.sort();
    assert_eq!(got, want);
}

/// The demo writes its own user into every request that names one
/// (`turbo_kserve_demo::user`). Its table is kept by hand, so a method
/// whose request has a top-level string field `user` and is missing from
/// it would be forwarded with the page's user: this fails first.
#[test]
fn every_request_that_names_a_user_is_rewritten() {
    use protox::prost_reflect::prost_types::field_descriptor_proto::Type;
    use turbo_kserve_demo::user::USER_FIELDS;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../server/proto");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".proto"))
        .collect();
    files.sort();
    let fds = protox::compile(&files, [&dir]).unwrap();
    let mut user_of = std::collections::HashMap::new();
    for file in &fds.file {
        for m in &file.message_type {
            let user = m.field.iter().find(|f| f.name() == "user" && f.r#type() == Type::String);
            if let Some(f) = user {
                user_of.insert(format!(".{}.{}", file.package(), m.name()), f.number() as u32);
            }
        }
    }
    let mut want = Vec::new();
    for file in &fds.file {
        for service in &file.service {
            for m in &service.method {
                if let Some(&n) = user_of.get(m.input_type()) {
                    want.push((format!("/{}.{}/{}", file.package(), service.name(), m.name()), n));
                }
            }
        }
    }
    want.sort();
    let mut got: Vec<(String, u32)> = USER_FIELDS.iter().map(|&(p, n)| (p.to_string(), n)).collect();
    got.sort();
    assert_eq!(got, want);
}
