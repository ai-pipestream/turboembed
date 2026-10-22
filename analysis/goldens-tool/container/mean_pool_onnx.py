"""A mean-pooled copy of a sentence-transformers BERT ONNX export, for the
SentenceVectorsDL goldens. `turbo-goldens` runs it in the bundle tool's
pinned reference image, with no network, on the upstream export it fetched
and hashed; neither the library nor the generator reads the upstream file.

OpenNLP's SentenceVectorsDL returns `output[0][0]` of the graph's first
output. On the upstream export that is the [CLS] row of
`last_hidden_state`. This copy puts a new first output in front:
`sentence_embedding`, shaped [batch, 1, hidden], the mean of
`last_hidden_state` over the positions `attention_mask` keeps (the mean
sentence-transformers pools with), so `output[0][0]` is the pooled vector.
`last_hidden_state` stays as the second output. No other node changes.

Arguments: the upstream ONNX file, the file to write, and the JSON file
that says what ran.
"""

import hashlib
import json
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx import TensorProto, helper


def main():
    src, out, report = sys.argv[1:4]
    m = onnx.load(src)
    g = m.graph
    hidden = next(o for o in g.output if o.name == "last_hidden_state")
    nodes = [
        helper.make_node("Cast", ["attention_mask"], ["pool_mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["pool_mask_f", "pool_axis_2"], ["pool_mask_3d"]),
        helper.make_node("Mul", ["last_hidden_state", "pool_mask_3d"], ["pool_masked"]),
        helper.make_node("ReduceSum", ["pool_masked", "pool_axis_1"], ["pool_sum"], keepdims=1),
        helper.make_node("ReduceSum", ["pool_mask_3d", "pool_axis_1"], ["pool_count"], keepdims=1),
        helper.make_node("Max", ["pool_count", "pool_eps"], ["pool_count_safe"]),
        helper.make_node("Div", ["pool_sum", "pool_count_safe"], ["sentence_embedding"]),
    ]
    g.initializer.extend([
        helper.make_tensor("pool_axis_1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("pool_axis_2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("pool_eps", TensorProto.FLOAT, [], [1e-9]),
    ])
    g.node.extend(nodes)
    dims = hidden.type.tensor_type.shape.dim
    pooled = helper.make_tensor_value_info(
        "sentence_embedding", TensorProto.FLOAT, [dims[0].dim_param or dims[0].dim_value, 1, dims[2].dim_value]
    )
    outputs = [pooled] + list(g.output)
    del g.output[:]
    g.output.extend(outputs)
    onnx.checker.check_model(m)
    onnx.save(m, out)

    # Check the pooled output against the mean computed here from the
    # second output, on a fixed input with padding, with onnx's reference
    # evaluator (the image has no onnxruntime).
    s = ReferenceEvaluator(m)
    ids = np.array([[101, 7592, 2088, 2003, 1037, 3231, 102, 0, 0]], dtype=np.int64)
    mask = (ids != 0).astype(np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if any(i.name == "token_type_ids" for i in g.input):
        feeds["token_type_ids"] = np.zeros_like(ids)
    pooled_out, hidden_out = s.run(["sentence_embedding", "last_hidden_state"], feeds)
    want = (hidden_out * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    diff = float(np.abs(pooled_out[:, 0, :] - want).max())
    if diff > 1e-5:
        sys.exit(f"pooled output differs from the mean by {diff}")

    with open(report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "mean_pool_onnx.py",
                "onnx": onnx.__version__,
                "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
                "output_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
                "check_max_abs_diff": diff,
            },
            f,
        )


if __name__ == "__main__":
    main()
