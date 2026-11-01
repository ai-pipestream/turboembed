# The document sink over gRPC

`turbo.sink.v1.TurboSinkService` (`server/proto/turbo_sink.proto`) carries
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
strictly increases within a stream. An item carrying a graph can also
carry `typed` (field 6), the graph's `TypedDocumentResult` exactly as
`InspectDocumentGraph` answers it for the item's `doc_id` and the view
and offset mode the sender is configured with, so a sink in any language
reads the annotations without decoding the graph or calling back. Field 7
is reserved for the document's analysis time.

## Forwarding

With `--sink-endpoint URI`, every answer an `AnalyzeStream` call
completes is offered to the forwarder as the client is handed it, as an
item carrying that answer. Answers a client never takes, because it went
away first, are not offered. A failed document is not a completed one and
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
offered, forwarded (taken by a call to send), dropped (refused by a full
queue), refused (document graphs refused before the queue), abandoned (left in the queue when a close stopped the drain),
queued items and bytes, failed calls with the last
error, and the last summary a sink answered with. When the server
stops, it closes the queue and waits up to 10 seconds for what the queue
holds to be sent and the sink's summary to arrive, then stops the drain
and counts what the queue still holds as abandoned, so a sink that is down is not retried after the server stops; `Server::close_sink`
does that on its own and answers with the final counts.

## Forwarding document graphs

With `--sink-documents` as well (`SinkConfig::documents`), every
`AnalyzeDocumentProgressive` call that ends with its `Complete` event, the
document's whole graph, forwards that graph once the client has been
handed the event and the reply has ended. The item's `doc_id` is the
request's `doc_id`, as the call's own events carry it, empty when the
request gives none, as an `AnalyzeStream` answer carries the id its
document gave. A call that ends in an error, runs out of time, or whose
client went away before the `Complete` event is not forwarded. Forwarding
never fails or delays the call: a full queue drops and counts the item as
it does an `AnalyzeStream` answer. Without the flag, progressive calls
forward nothing.

A program embedding the server forwards a document graph itself with
`Server::forward_document(doc_id, graph)`, which answers whether the queue
took it, and refuses, with nothing offered, a graph for a server without
a sink.

Either way, the graph is imported and re-exported, and the item carries
the re-exported bytes with their layers and views listed, as
`RoundTripDocumentGraph` answers them. It goes through the same queue,
counts and numbering as an `AnalyzeStream` answer. A graph that does not
import, or one the typed projection cannot be built for, is refused
before the queue and counted `refused` in `Server::sink_stats`, with the
reason as the last error.

## The typed projection

`--sink-typed-offsets utf8|utf16|code-point` asks for the typed
projection on every item carrying a graph, with `--sink-typed-view N`
(default 0, the source view) choosing the view; both need
`--sink-endpoint`, and the view needs the offsets. In the library it is
`SinkConfig::typed`, `Some(SinkTyped { view, offsets })`, where `view`
(`u32`) and `offsets` (`OffsetMode`) mean what they mean on
`InspectDocumentGraphRequest`. The projection is built from the item's
own graph bytes by the projection `InspectDocumentGraph` uses, so it
equals what that call answers for the same graph and options. A view
over 65535 refuses the start; a view a document does not have refuses
that document, which is counted `refused`. Without it, the default, items
carry no projection and are what they were before it existed. Items
carrying an `AnalyzeStream` answer never carry one: they have no graph.

## Serving the sink

`Serve { sink_receiver: Some(SinkReceiver::new(on_item)), .. }` serves
`TurboSinkService` on the server's port. Each item is checked and, when
accepted, handed to `on_item` in the order its stream sent it. An item is
rejected when:

- its sequence is not over the last accepted item's in the stream;
- it carries neither a graph nor an answer;
- its graph is over `max_graph_bytes` (64 MiB), or does not decode and
  import with the server's codec registry;
- it carries a typed projection without a graph, or one that
  `RoundTripTypedDocument` refuses (span bounds, UTF-16 surrogate halves,
  unknown offset modes, model references, index bounds).

The summary lists the first 100 reasons, each naming the item's
sequence. A server started without a receiver answers `StreamDocuments`
with `UNIMPLEMENTED`.
