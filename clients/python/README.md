# turboembed-proto

Generated Protobuf and gRPC Python client stubs for `turbo-kserve` (`turbo.*.v1` and the Open Inference Protocol).

## Usage

```python
import grpc
from turboembed_proto import turbo_analysis_pb2, turbo_analysis_pb2_grpc

with grpc.insecure_channel("localhost:8081") as channel:
    stub = turbo_analysis_pb2_grpc.AnalysisServiceStub(channel)
    caps = stub.Capabilities(turbo_analysis_pb2.CapabilitiesRequest())
```
