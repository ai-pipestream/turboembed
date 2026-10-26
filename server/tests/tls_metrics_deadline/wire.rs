//! Read server deadline status without tonic's earlier client-local timer.
use std::time::Duration;

pub async fn zero_budget_status<M: prost::Message>(server: &turbo_kserve::Server, path: &str, message: M) -> String {
    let socket = tokio::net::TcpStream::connect(server.local_addr()).await.unwrap();
    let (mut sender, connection) = h2::client::handshake(socket).await.unwrap();
    let connection = tokio::spawn(connection);
    let request = tonic::codegen::http::Request::builder()
        .method("POST")
        .uri(format!("http://{}{path}", server.local_addr()))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("grpc-timeout", "0n")
        .body(())
        .unwrap();
    let encoded = message.encode_to_vec();
    let mut frame = vec![0];
    frame.extend_from_slice(&u32::try_from(encoded.len()).unwrap().to_be_bytes());
    frame.extend_from_slice(&encoded);
    let (response, mut body) = sender.send_request(request, false).unwrap();
    body.send_data(frame.into(), true).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(2), response).await.unwrap().unwrap();
    let status = response.headers().get("grpc-status").unwrap().to_str().unwrap().to_owned();
    drop(body);
    drop(response);
    drop(sender);
    connection.abort();
    status
}
