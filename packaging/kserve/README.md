# Inferstream on KServe

Two manifests: a `ClusterServingRuntime` that says how to run the
Inferstream image, and an `InferenceService` that points it at a bundle.

```sh
scripts/inferstream-image.sh                     # turbo-inferstream:cpu
docker tag turbo-inferstream:cpu registry.example/turbo-inferstream:cpu
docker push registry.example/turbo-inferstream:cpu
# set the image in servingruntime.yaml, then
kubectl apply -f packaging/kserve/servingruntime.yaml
kubectl apply -f packaging/kserve/inferenceservice.yaml
kubectl get inferenceservice qwen -w
```

The runtime fixes the provider (`ggml` in the CPU image) and the server
options, and KServe fills in the model name and the bundle: the storage
initializer copies `storageUri` to `/mnt/models`, and the container
starts `inferstream --model name=<InferenceService name>,bundle=/mnt/models,...`.
A load that fails (a bundle the provider cannot run, a missing artifact)
fails the pod with the Turbo status in its log; the pod never reports
ready with a half-loaded model.

Once ready, the predictor answers the Open Inference Protocol v2 at
`/v2/...` (what `kserve.InferenceRESTClient` and `kserve.InferenceGRPCClient`
speak), the OpenAI-shaped routes at `/v1/...`, and the repository
extension at `/v2/repository/...`, which loads other bundles the pod can
reach (for example other directories on the same claim mounted through a
`volumes` entry on the runtime). The gRPC listener is on the pod's 8081;
`grpcurl -plaintext <pod>:8081 list` works without a proto file because
the server publishes reflection.

A GPU runtime is the same `ClusterServingRuntime` on
`turbo-inferstream:cuda` (`scripts/inferstream-image.sh cuda`) with
`provider=ggml,ordinal=0` in the model argument and an `nvidia.com/gpu: 1`
limit; the CUDA and OpenVINO ONNX providers need their SDKs in the image
and are not built by the Dockerfile today (`docs/packaging.md`).
