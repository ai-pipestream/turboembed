//! `DemoService/Describe`, the one call the demo answers itself: the write
//! methods it forwards and whether it has a user, as binary protobuf
//! behind the same gate as a forwarded call, and in step with
//! demo/proto/turbo_demo.proto.

mod common;

use std::path::Path;

use common::*;
use prost::Message;
use turbo_kserve_demo::connect::DESCRIBE;
use turbo_kserve_demo::{Allow, Options};

/// `turbo.demo.v1.DescribeResponse`, as the proto declares it.
#[derive(Clone, PartialEq, Message)]
struct DescribeResponse {
    #[prost(string, repeated, tag = "1")]
    forwarded_writes: Vec<String>,
    #[prost(bool, tag = "2")]
    has_user: bool,
}

const CREATE: &str = "/turbo.search.v1.TurboSearchService/CreateIndex";
const INDEX: &str = "/turbo.search.v1.TurboSearchService/IndexDocuments";

async fn describe(d: &turbo_kserve_demo::Demo) -> DescribeResponse {
    let r = http(d, "POST", DESCRIBE, &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some(UNARY));
    DescribeResponse::decode(r.all().await.as_slice()).unwrap()
}

#[tokio::test]
async fn with_no_allow_and_no_user_nothing_is_listed_and_no_server_is_called() {
    // Nothing listens at the server's address: the demo answers alone.
    let d = demo_at(nowhere().await, &[]).await;
    assert_eq!(describe(&d).await, DescribeResponse::default());
}

#[tokio::test]
async fn the_allowed_writes_and_the_user_are_reported() {
    let options = Options {
        allow: Allow::with(&[INDEX.into(), CREATE.into()]).unwrap(),
        user: Some("alice".into()),
        ..Options::default()
    };
    let d = demo_with(nowhere().await, options).await;
    let r = describe(&d).await;
    assert_eq!(r.forwarded_writes, [CREATE, INDEX]);
    assert!(r.has_user);
}

#[tokio::test]
async fn it_is_a_unary_binary_post_behind_the_token() {
    let options = Options { token: Some("s3cret".into()), ..Options::default() };
    let d = demo_with(nowhere().await, options).await;
    let r = http(&d, "POST", DESCRIBE, &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 401);
    let r = http(&d, "POST", DESCRIBE, &[("Content-Type", UNARY), ("Authorization", "Bearer s3cret")], b"").await;
    assert_eq!(r.status, 200);
    for kind in ["application/json", "application/connect+json", STREAM] {
        let r = http(&d, "POST", DESCRIBE, &[("Content-Type", kind), ("Authorization", "Bearer s3cret")], b"").await;
        assert_eq!(r.status, 415, "{kind}");
    }
    let r = http(&d, "GET", DESCRIBE, &[], b"").await;
    assert_eq!(r.status, 405);
}

#[test]
fn the_path_and_fields_are_the_protos() {
    use protox::prost_reflect::prost_types::field_descriptor_proto::{Label, Type};
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("proto");
    let fds = protox::compile(["turbo_demo.proto"], [&dir]).unwrap();
    let file = &fds.file[0];
    let paths: Vec<String> = file
        .service
        .iter()
        .flat_map(|s| s.method.iter().map(move |m| format!("/{}.{}/{}", file.package(), s.name(), m.name())))
        .collect();
    assert_eq!(paths, [DESCRIBE]);
    let reply = file.message_type.iter().find(|m| m.name() == "DescribeResponse").unwrap();
    let fields: Vec<(&str, i32, Type, Label)> =
        reply.field.iter().map(|f| (f.name(), f.number(), f.r#type(), f.label())).collect();
    assert_eq!(
        fields,
        [("forwarded_writes", 1, Type::String, Label::Repeated), ("has_user", 2, Type::Bool, Label::Optional)]
    );
}
