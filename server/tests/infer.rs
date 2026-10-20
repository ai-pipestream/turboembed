//! The Open Inference Protocol against the server in-process on the CPU
//! with testdata/tiny-bert-bundle: every RPC, against the bundle's
//! reference vectors and the statuses docs/kserve.md gives.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;

#[tokio::test]
async fn capacity() {
    let (s, mut c) = serve(1).await;
    let many: Vec<&str> = vec!["a"; 65];
    let e = refused(&mut c, texts(&many)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    assert!(e.message().starts_with("TURBO_E_CAPACITY: "), "{}", e.message());
    assert_eq!(embed(&mut c, texts(&many[..64])).await.len(), 64);

    let row = vec![101, 7592, 102];
    let rows = vec![row.clone(); 65];
    // Typed rows go to the library, which refuses them.
    let e = refused(&mut c, typed_tokens(&rows, false)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    // Raw rows larger than the session's buffers are the server's refusal.
    let e = refused(&mut c, raw_tokens(&rows, true)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    assert!(e.message().contains("max_batch 64"), "{}", e.message());
    let mut long = row.clone();
    long.resize(65, 1000);
    for r in [raw_tokens(std::slice::from_ref(&long), false), typed_tokens(std::slice::from_ref(&long), false)] {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    }
    // An oversized raw request with an option the library would refuse is
    // told CAPACITY.
    let e = refused(&mut c, with(raw_tokens(&rows, false), "output_dim", int(16))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    s.stop().await;
}

/// sessions=1, and the one session held through Server::hold, which takes
/// it from the pool exactly as a ModelInfer does and keeps it until dropped.
/// While it is held every ModelInfer is BUSY at once, the calls that take
/// no session are answered, and once it is dropped the next request runs.
#[tokio::test]
async fn busy_when_every_session_is_held() {
    let (s, mut c) = serve(1).await;
    assert_eq!(embed(&mut c, texts(&["a"])).await.len(), 1);
    let held = s.hold(NAME).unwrap();
    for r in [texts(&["a"]), raw_tokens(&[vec![101, 102]], false)] {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), (Code::Unavailable, 769, 0));
        assert!(e.message().starts_with("TURBO_E_BUSY: "), "{}", e.message());
    }
    assert_eq!(s.hold(NAME).err().map(|f| f.code), Some(769));
    assert!(
        c.model_ready(ModelReadyRequest { name: NAME.into(), version: String::new() })
            .await
            .unwrap()
            .into_inner()
            .ready
    );
    assert!(c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap();
    c.server_metadata(ServerMetadataRequest {}).await.unwrap();
    // A request is checked before a session is taken: only one the server
    // would send to the library is told BUSY.
    let e = refused(&mut c, with(texts(&["a"]), "precision", st("PRECISION_EXACT"))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 0));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![2];
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::InvalidArgument, 260, 0));
    let e = refused(&mut c, raw_tokens(&vec![vec![101, 102]; 65], false)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    drop(held);
    assert_eq!(embed(&mut c, texts(&["a"])).await.len(), 1);
    s.stop().await;
}

/// A request message larger than --max-message-bytes is refused by gRPC with
/// RESOURCE_EXHAUSTED before it is read, and carries no turbo-code.
#[tokio::test]
async fn a_message_over_the_limit() {
    let s =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), 1)], 4096, Serve::default()).await.unwrap();
    s.load().await.unwrap();
    let mut c = client(&s).await;
    let big = "a ".repeat(4096);
    let e = refused(&mut c, texts(&[&big])).await;
    assert_eq!(e.code(), Code::ResourceExhausted, "{e:?}");
    assert!(e.metadata().get("turbo-code").is_none() && e.metadata().get("turbo-field").is_none(), "{e:?}");
    assert_eq!(embed(&mut c, texts(&[&big[..1000]])).await.len(), 1);
    s.stop().await;
}

/// Two sessions serve two requests at once.
#[tokio::test]
async fn one_request_per_session() {
    let (s, c) = serve(2).await;
    let held = s.hold(NAME).unwrap();
    let mut c1 = c.clone();
    assert_eq!(embed(&mut c1, texts(&["a", "b"])).await.len(), 2);
    let _second = s.hold(NAME).unwrap();
    assert_eq!(refusal(&refused(&mut c1, texts(&["a"])).await).1, 769);
    drop(held);
    assert_eq!(embed(&mut c1, texts(&["a"])).await.len(), 1);
    s.stop().await;
}

#[tokio::test]
async fn unknown_model_or_version() {
    let (s, mut c) = serve(1).await;
    let e = c.model_ready(ModelReadyRequest { name: "tiny".into(), version: String::new() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    assert!(e.message().starts_with("TURBO_E_BUNDLE_NOT_FOUND: "), "{}", e.message());
    let e = c.model_metadata(ModelMetadataRequest { name: "tiny".into(), version: String::new() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    let e = c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: "0".into() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_name = "Tiny-Bert-Bundle".into();
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_version = "2".into();
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_version = "1".into();
    assert_eq!(embed(&mut c, r).await.len(), 1);
    s.stop().await;
}

#[tokio::test]
async fn input_refusals() {
    let (s, mut c) = serve(1).await;
    let arg = (Code::InvalidArgument, 256, 0);
    let shape = (Code::InvalidArgument, 260, 0);
    let rows = vec![vec![101, 7592, 102]];

    let mut cases: Vec<(&str, ModelInferRequest, (Code, i32, u32))> = Vec::new();
    cases.push(("no inputs", request(vec![], vec![]), arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].datatype = "INT64".into();
    r.inputs[0].contents = Some(InferTensorContents { int64_contents: vec![101, 7592, 102], ..Default::default() });
    cases.push(("INT64 ids", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].datatype = "STRING".into();
    cases.push(("texts not BYTES", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs.remove(1);
    cases.push(("ids without mask", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs.push(texts(&["a"]).inputs.remove(0));
    cases.push(("texts and ids", r, arg));
    let mut r = typed_tokens(&rows, true);
    r.inputs[2].name = "mask".into();
    cases.push(("a name given twice", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].name = "text".into();
    cases.push(("an unknown input", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].parameters.insert("x".into(), InferParameter { parameter_choice: Some(int(1)) });
    cases.push(("input parameters", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor { name: "embeddings".into(), parameters: HashMap::new() }];
    cases.push(("an unknown output", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor { name: "vectors".into(), parameters: HashMap::new() }; 2];
    cases.push(("vectors twice", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor {
        name: "vectors".into(),
        parameters: HashMap::from([("x".into(), InferParameter { parameter_choice: Some(int(1)) })]),
    }];
    cases.push(("output parameters", r, arg));
    let mut r = texts(&["a"]);
    r.raw_input_contents = vec![vec![1, 0, 0, 0, b'a']];
    cases.push(("raw and typed", r, arg));
    let mut r = raw_tokens(&rows, false);
    r.raw_input_contents.pop();
    cases.push(("a raw entry short", r, arg));
    let mut r = raw_texts(&[b"a"]);
    r.raw_input_contents[0].push(9);
    cases.push(("malformed raw BYTES", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].contents.as_mut().unwrap().int_contents = vec![1];
    cases.push(("texts in int_contents too", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs[1].contents = Some(InferTensorContents { uint_contents: vec![1, 1, 1], ..Default::default() });
    cases.push(("mask in uint_contents", r, arg));

    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![1, 1];
    cases.push(("texts of rank 2", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].shape = vec![3];
    cases.push(("ids of rank 1", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![-1];
    cases.push(("a negative extent", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![1 << 32];
    cases.push(("an extent over uint32_t", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![2];
    cases.push(("fewer texts than the shape", r, shape));
    let mut r = raw_texts(&[b"a", b"b"]);
    r.inputs[0].shape = vec![1];
    cases.push(("more raw texts than the shape", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[1].shape = vec![3, 1];
    cases.push(("mask shaped unlike ids", r, shape));
    let mut r = typed_tokens(&rows, true);
    r.inputs[2].shape = vec![1, 2];
    cases.push(("types shaped unlike ids", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].contents.as_mut().unwrap().int_contents.pop();
    cases.push(("typed ids short", r, shape));
    let mut r = raw_tokens(&rows, false);
    r.raw_input_contents[1].extend([0, 0, 0, 0]);
    cases.push(("raw mask long", r, shape));
    // 4 x batch x seq is 2^64 here, which wraps to 0 in a uint64_t: the
    // empty contents must not pass for it.
    for e in [1i64 << 31, u32::MAX as i64] {
        let mut r = raw_tokens(&rows, false);
        for i in &mut r.inputs {
            i.shape = vec![e, e];
        }
        r.raw_input_contents = vec![vec![], vec![]];
        cases.push(("raw rows past a uint64_t of bytes", r, shape));
    }

    cases.push(("not UTF-8", raw_texts(&[b"\xff\xfe"]), (Code::InvalidArgument, 258, 0)));

    for (what, r, want) in cases {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), want, "{what}: {}", e.message());
    }
    s.stop().await;
}
