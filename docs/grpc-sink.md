# The document sink over gRPC

`turbo.sink.TurboSinkService` (`server/proto/turbo_sink.proto`) carries
analyzed documents from one `turbo-kserve` to the next stage of an
ingestion pipeline: another server, or a sink that writes them to an
index. A server forwards to a sink when started with `--sink-endpoint`,
and a program embedding the server can also serve the sink itself.

## The contract

`StreamDocuments` is client-streaming: the sender streams
`DocumentSinkItem`s, and the sink answers once the sender closes the
stream with a `DocumentSinkSummary` of what it received, accepted and
rejected.

An item carries a document graph (binary `DocumentGraph` bytes, as
`RoundTripDocumentGraph` takes them, with its layers and views listed),
an `AnalyzeStream` answer, or both. Its `sequence` is the sender's, and
strictly increases within a stream. Fields 6 and 7 are reserved for the
typed document projection (#1219) and the document's analysis time.

## Forwarding

With `--sink-endpoint URI`, every answer an `AnalyzeStream` call
completes is offered to the forwarder as the client is handed it, as an
item carrying that answer. A failed document is not a completed one and
is not offered. The offer never waits: the queue holds at most
`--sink-queue-items` items (default 1024) and `--sink-queue-bytes`
encoded bytes (default 64 MiB), and an answer that does not fit is
dropped and counted. A slow or absent sink therefore never holds back
the stream that produced the answer.

One task drains the queue into a long-lived `StreamDocuments` call. The
call takes each item from the queue as it sends it, so items waiting on
a slow sink stay in the queue and in its counts, and each is numbered
then: sequences follow the order items are sent in, whatever stream
completed them. A call that fails is counted as it fails, also while no
item is waiting, and a new one opens half a second later; the items it
had taken are lost with it. `Server::sink_stats` gives the counts:
offered, forwarded (taken by a call to send), dropped, queued items and bytes, failed calls with the last
error, and the last summary a sink answered with. When the server
stops, it closes the queue and waits up to 10 seconds for what the queue
holds to be sent and the sink's summary to arrive; `Server::close_sink`
does that on its own and answers with the final counts.

## Serving the sink

`Serve { sink_receiver: Some(SinkReceiver::new(on_item)), .. }` serves
`TurboSinkService` on the server's port. Each item is checked and, when
accepted, handed to `on_item` in the order its stream sent it. An item is
rejected when:

- its sequence is not over the last accepted item's in the stream;
- it carries neither a graph nor an answer;
- its graph is over `max_graph_bytes` (64 MiB), or does not decode and
  import with the server's codec registry.

The summary lists the first 100 reasons, each naming the item's
sequence. A server started without a receiver answers `StreamDocuments`
with `UNIMPLEMENTED`.
