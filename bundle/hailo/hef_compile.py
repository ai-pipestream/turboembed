"""A HEF for a Hailo device from a bundle's ONNX export of a BERT encoder.
The bundle tool runs it in the Dataflow Compiler container, with no
network, on files it fetched and hashed. The library never reads the ONNX
file; it loads the HEF this writes.

The HEF takes what the hailo backend gives it (docs/bundle.md, "Graph
inputs"): the word-embedding rows the ids select, [seq, hidden], and an
additive attention bias per head, [heads, seq, seq] as the graph has it,
0 for a key the mask keeps and MASKED for one it drops. It gives the last
layer's hidden states, [seq, hidden]. Everything between the gather and
the output is in the HEF: the position and token type add, the
embeddings LayerNorm and every layer.

The export is cut to that shape first:
  - the graph's inputs become constants of the fixed shape [1, seq]:
    input_ids and token_type_ids zeros, attention_mask ones, so the
    position and token type (type 0) lookups fold into constants;
  - the word-embedding Gather's output, the one on input_ids, becomes the
    input word_rows;
  - the tensor every attention Softmax adds to its scores becomes the
    input attn_bias, one bias per head.
The cut graph is checked against the export under onnxruntime on every
calibration text, on the positions the mask keeps, before it is compiled.

The compiler then parses the cut graph for the target, quantizes it to
8 bits on the calibration texts (tokenized by the bundle's tokenizer,
their word rows gathered and their bias built as the backend does), and
compiles it.

Arguments: the ONNX file, the HEF to write, the JSON file that says what
ran, then --tokenizer, --calibration (JSON lines, one {"text": ...} each),
--target, --seq and --heads.
"""

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify
from tokenizers import Tokenizer

# The bias for a dropped key: exp(-100) underflows to 0 in the softmax.
# The hailo backend writes the same value (core/hailo/embed.cpp).
MASKED = -100.0

# The model script. equalization and matmul correction as Hailo's own
# MiniLM compile uses them, at optimization level 0 with no compression:
# on the Hailo-10H, levels 1 and 2, 16-bit layers and quantization-aware
# finetuning did not come closer to the F32 vectors than this.
MODEL_SCRIPT = [
    "model_optimization_config(calibration, batch_size=8, calibset_size={calibset})",
    "pre_quantization_optimization(equalization, policy=enabled)",
    "pre_quantization_optimization(ew_add_fusing, policy=disabled)",
    "model_optimization_flavor(optimization_level=0, compression_level=0)",
    "pre_quantization_optimization(matmul_correction, layers={{matmul*}}, correction_type=zp_comp_block)",
    "model_optimization_config(negative_exponent, layers={{*}}, rank=0)",
]

# How far the cut graph may be from the export on a kept position.
CUT_TOLERANCE = 1e-4


def producers(graph):
    return {o: n for n in graph.node for o in n.output}


def word_gather(graph):
    """The Gather on input_ids: its data is the word-embedding table."""
    found = [n for n in graph.node if n.op_type == "Gather" and n.input[1] == "input_ids"]
    if len(found) != 1:
        sys.exit(f"hef_compile: {len(found)} Gathers read input_ids; a BERT export has one, the word embeddings")
    return found[0]


def attention_bias(graph):
    """The tensor every attention Softmax adds to its scores."""
    made = producers(graph)
    biases = set()
    for sm in (n for n in graph.node if n.op_type == "Softmax"):
        add = made.get(sm.input[0])
        if add is None or add.op_type != "Add":
            sys.exit(f"hef_compile: Softmax {sm.name} does not read an Add of the scores and a bias")
        scores = [i for i in add.input if i in made and made[i].op_type == "MatMul"]
        others = [i for i in add.input if i not in scores]
        if len(scores) != 1 or len(others) != 1:
            sys.exit(f"hef_compile: Add {add.name} before a Softmax is not scores plus one bias")
        biases.add(others[0])
    if len(biases) != 1:
        sys.exit(f"hef_compile: the Softmaxes add {len(biases)} different biases; a BERT export adds one mask")
    return biases.pop()


def cut(model, seq, hidden, heads):
    """The export with word_rows and attn_bias as its inputs, fixed at [1, seq]."""
    g = model.graph
    word = word_gather(g)
    table = next(t for t in g.initializer if t.name == word.input[0])
    bias = attention_bias(g)
    for name in ["input_ids", "attention_mask", "token_type_ids"]:
        inp = [i for i in g.input if i.name == name]
        if not inp:
            if name == "token_type_ids":
                continue
            sys.exit(f"hef_compile: the export has no input {name}")
        g.input.remove(inp[0])
        value = np.ones((1, seq), np.int64) if name == "attention_mask" else np.zeros((1, seq), np.int64)
        g.initializer.append(numpy_helper.from_array(value, name))
    g.input.append(helper.make_tensor_value_info("word_rows", TensorProto.FLOAT, [1, seq, hidden]))
    g.input.append(helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, [1, heads, seq, seq]))
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == word.output[0]:
                n.input[k] = "word_rows"
            elif i == bias:
                n.input[k] = "attn_bias"
    del g.value_info[:]
    simplified, ok = simplify(model)
    if not ok:
        sys.exit("hef_compile: onnxsim could not check the cut graph")
    return simplified, numpy_helper.to_array(table)


def frames(tok, table, texts, seq, heads):
    """Each text's word rows, per-head bias and mask, as the backend makes them."""
    rows, bias, masks = [], [], []
    for t in texts:
        e = tok.encode(t)
        ids = np.array(e.ids)
        m = np.array(e.attention_mask)
        rows.append(table[ids][None].astype(np.float32))
        keys = np.where(m == 1, 0.0, MASKED).astype(np.float32)
        bias.append(np.broadcast_to(keys, (heads, seq, seq)).astype(np.float32))
        masks.append(m)
    return np.stack(rows), np.stack(bias), np.stack(masks)


def check_cut(export, cut_path, tok, texts, rows, bias, masks):
    """The cut graph gives the export's hidden states on every kept position."""
    full = ort.InferenceSession(export)
    part = ort.InferenceSession(cut_path)
    typed = any(i.name == "token_type_ids" for i in full.get_inputs())
    worst = 0.0
    for i, t in enumerate(texts):
        e = tok.encode(t)
        ids = np.array([e.ids], np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask], np.int64)}
        if typed:
            feed["token_type_ids"] = np.zeros_like(ids)
        a = full.run(None, feed)[0][0]
        b = part.run(None, {"word_rows": rows[i], "attn_bias": bias[i][None]})[0][0]
        live = masks[i] == 1
        worst = max(worst, float(np.abs(a[live] - b[live]).max()))
    if worst > CUT_TOLERANCE:
        sys.exit(f"hef_compile: the cut graph is {worst} from the export on a kept position, over {CUT_TOLERANCE}")
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("out")
    p.add_argument("report")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    a = p.parse_args()

    from hailo_sdk_client import ClientRunner, __version__ as dfc_version
    from hailo_sdk_client.exposed_definitions import Dims

    texts = [json.loads(line)["text"] for line in open(a.calibration, encoding="utf-8") if line.strip()]
    tok = Tokenizer.from_file(a.tokenizer)
    tok.enable_truncation(a.seq)
    tok.enable_padding(length=a.seq, pad_id=tok.token_to_id("[PAD]") or 0)

    export = onnx.load(a.onnx)
    word = word_gather(export.graph)
    hidden = numpy_helper.to_array(next(t for t in export.graph.initializer if t.name == word.input[0])).shape[1]
    cut_model, table = cut(export, a.seq, hidden, a.heads)
    cut_path = "/tmp/cut.onnx"
    onnx.save(cut_model, cut_path)

    rows, bias, masks = frames(tok, table, texts, a.seq, a.heads)
    worst = check_cut(a.onnx, cut_path, tok, texts, rows, bias, masks)

    runner = ClientRunner(hw_arch=a.target)
    fmt = {
        "word_rows": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
        "attn_bias": [Dims.BATCH, Dims.GROUPS, Dims.WIDTH, Dims.CHANNELS],
    }
    runner.translate_onnx_model(cut_path, "encoder", net_input_format=fmt)
    script = [line.format(calibset=len(texts)) for line in MODEL_SCRIPT]
    runner.load_model_script("\n".join(script) + "\n")
    # The compiler's layout of the bias: [seq, heads * seq], each query's
    # keys once per head.
    hailo_bias = bias.transpose(0, 2, 1, 3).reshape(len(texts), 1, a.seq, a.heads * a.seq)
    runner.optimize({"encoder/input_layer1": rows, "encoder/input_layer2": hailo_bias})
    hef = runner.compile()
    with open(a.out, "wb") as f:
        f.write(hef)

    report = {
        "tool": "hailo-dataflow-compiler",
        "tool_version": dfc_version,
        "settings": [
            f"hw_arch={a.target}",
            f"seq={a.seq}",
            f"heads={a.heads}",
            f"masked={MASKED!r}",
            f"calibration_texts={len(texts)}",
            f"cut_max_abs_diff={worst:.3g}",
        ]
        + script,
    }
    with open(a.report, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
