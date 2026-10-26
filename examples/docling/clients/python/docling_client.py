#!/usr/bin/env python3
"""The docling example from Python, against a running turbo-kserve.

Sends the docling structure (data/report.seed.graph, which
`docling-client seed` writes) as AnalyzeDocumentProgressive's
initial_graph, lets the server analyze, chunk and embed it, then asks
InspectDocumentWindow for the heat and ProjectChunks for the chunks.
Offsets are asked for in code points, which Python strings index.
Prints the summary and, given the expected file, exits 1 unless it
matches.

    python3 docling_client.py HOST:PORT SEED [EXPECTED]

Needs only the turboembed_proto wheel (clients/python) and grpcio.
"""

from __future__ import annotations

import sys

import grpc
from turboembed_proto import turbo_analysis_pb2 as pb
from turboembed_proto import turbo_analysis_pb2_grpc as rpc
from turboembed_proto.turbo_typed_annotations_pb2 import OFFSET_MODE_CODE_POINT

MODEL = "tiny-bert-bundle"
STEPS = [
    "turbo:normalize:whitespace",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo:subwords",
    "turbo-geo:names",
    "turbo:structure-boundaries",
    "turbo:chunker",
    "turbo:levels",
    "turbo:chunk-vectors",
]
HEAT_BUCKETS = 16
ENTITIES = "opennlp:entities"
TIMEOUT = 60.0


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def analyze(stub: rpc.AnalysisServiceStub, seed: bytes) -> tuple[int, bytes]:
    request = pb.AnalyzePipelineRequest(
        doc_id="report",
        offsets=OFFSET_MODE_CODE_POINT,
        steps=STEPS,
        model=MODEL,
        initial_graph=seed,
    )
    completed, graph = 0, None
    for event in stub.AnalyzeDocumentProgressive(request, timeout=TIMEOUT):
        kind = event.WhichOneof("event")
        if kind == "layers_ready":
            completed += 1
        elif kind == "step_failed":
            raise SystemExit(f"step {event.step_failed.step_id} failed: {event.step_failed.failure.message}")
        elif kind == "complete":
            graph = event.complete.graph
    if graph is None:
        raise SystemExit("the stream ended without the document")
    return completed, graph


def summary(completed: int, window: pb.DocumentWindowResponse, chunks: pb.ProjectChunksResponse) -> str:
    text = window.window.text
    lines = [f"steps {completed}"]
    for p in chunks.projection.points:
        vector = "no" if p.empty else "yes"
        lines.append(f"chunk {p.item} level {p.level} vector {vector} {quote(text[p.start:p.end])}")
    entities: dict[int, list] = {}
    for seg in window.segments:
        for a in seg.annotations:
            if a.layer == ENTITIES:
                e = entities.setdefault(a.item, [a.label, seg.start, seg.end])
                e[1], e[2] = min(e[1], seg.start), max(e[2], seg.end)
    for item in sorted(entities):
        label, start, end = entities[item]
        lines.append(f"entity {item} {label} {quote(text[start:end])}")
    for lane in window.heatmap_lanes:
        counts = " ".join(str(b.count) for b in lane.buckets)
        lines.append(f"heat {lane.id} {lane.kind} {counts}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: docling_client.py HOST:PORT SEED [EXPECTED]")
    with open(sys.argv[2], "rb") as f:
        seed = f.read()
    with grpc.insecure_channel(sys.argv[1]) as channel:
        stub = rpc.AnalysisServiceStub(channel)
        completed, graph = analyze(stub, seed)
        window = stub.InspectDocumentWindow(
            pb.DocumentWindowRequest(graph=graph, offsets=OFFSET_MODE_CODE_POINT, heatmap_buckets=HEAT_BUCKETS),
            timeout=TIMEOUT,
        )
        chunks = stub.ProjectChunks(
            pb.ProjectChunksRequest(graph=graph, offsets=OFFSET_MODE_CODE_POINT), timeout=TIMEOUT
        )
    got = summary(completed, window, chunks)
    sys.stdout.write(got)
    if len(sys.argv) == 4:
        with open(sys.argv[3], encoding="utf-8") as f:
            if f.read() != got:
                raise SystemExit(f"python docling client: the summary differs from {sys.argv[3]}")
        print("python docling client ok")


if __name__ == "__main__":
    main()
