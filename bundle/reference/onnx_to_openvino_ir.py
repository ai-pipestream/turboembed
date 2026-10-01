"""A static-shape OpenVINO IR from the upstream F32 ONNX export, for the
npu backend (docs/npu.md): the Intel NPU driver's own compiler builds the
graph on the machine from the IR's two files, so the library links no
OpenVINO; this conversion runs only while a bundle is made, in the
reference container, with no network.

The export's inputs (token ids, the attention mask, and token type ids
where the export takes them) are reshaped to one fixed [batch, seq],
because the NPU compiles static shapes: a dynamic IR fails in the
driver's compiler. The weights are compressed to FP16 (compress_to_fp16=True). That pass
rewrites Constant weights only. The Result stays FP32, and the npu
backend's build flags copy that port into --outputs_precisions, so the
driver's compiler reports an FP32 output and model_load refuses a
manifest compute_dtype of DTYPE_F16. The hidden states are converted to
f16 before the Result, the same PrePostProcessor step OpenVINO's NPU
compile tool uses for an FP16 output, so the port and the flag are
FP16. The ids stay integer. The script then refuses a Result that is
not FP16.

convert_model fuses attention into ScaledDotProductAttention, which is
opset 13. The Arrow Lake driver reports maxOVOpsetVersionSupported 11,
and the backend refuses an IR whose highest layer opset is above that.
The Python package does not wrap
ov::pass::ScaledDotProductAttentionDecomposition, so this script builds
the same subgraph that pass builds in OpenVINO 2026.3 (ops at opset 8
or below) and then refuses the saved xml if any layer is still above
the cap. Retagging a layer's version would leave an op the compiler
does not implement.

With --cut embeddings the export is cut the way the Hailo HEF is
(bundle/hailo/hef_compile.py) before that conversion. The graph's inputs
become word_rows [batch, seq, hidden] and attn_bias [batch, heads, seq, seq].
input_ids and token_type_ids become zeros and attention_mask becomes ones,
so the position lookup and the type-0 lookup fold into constants. The word
Gather's output becomes word_rows. The bias every attention Softmax adds
becomes attn_bias, 0 for a key the mask keeps and -100 for one it drops.
The cut is checked against the export on CPU, on the positions the mask
keeps, and the script exits if they differ by more than 1e-4. --heads is
required for that cut and refused without it. The gathered inputs stay
FP32, which is the dtype of the host word table. The Result is still FP16.

Arguments: the source ONNX file, the output xml, the output bin, the
produced_by report path, then --seq N --batch B, and optionally
--max-opset N (default 11). --cut embeddings --heads N selects the cut.
"""

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import openvino as ov
import openvino.utils as ov_utils
from openvino import opset1, opset3, opset4, opset8
from openvino.passes import ConstantFolding, Manager

COMPRESS_TO_FP16 = True
DEFAULT_MAX_OPSET = 11


def _i32(value):
    return opset1.constant(np.int32(value))


def _opset(node):
    version = node.get_type_info().version_id
    if not version.startswith("opset"):
        sys.exit(f"{node.get_friendly_name()}: type version {version!r} is not an opset")
    return int(version[len("opset") :])


def _decompose_sdpa(node):
    """The subgraph ov::pass::ScaledDotProductAttentionDecomposition::decompose
    builds in OpenVINO 2026.3 for query, key and value, an optional mask and
    an optional scale. A boolean mask keeps the True positions; a float mask
    is added to the scores. A sink input (the sixth) is refused: the Python
    package cannot build that node, so this conversion does not guess it."""
    n_in = node.get_input_size()
    if n_in < 3 or n_in > 5:
        sys.exit(
            f"ScaledDotProductAttention {node.get_friendly_name()}: {n_in} inputs; "
            "the decomposition covers 3 to 5 (query, key, value, an optional mask, an optional scale)"
        )
    attrs = node.get_attributes()
    if "causal" not in attrs:
        sys.exit(f"ScaledDotProductAttention {node.get_friendly_name()}: no causal attribute")
    causal = bool(attrs["causal"])

    query = node.input_value(0)
    key = node.input_value(1)
    value = node.input_value(2)
    q_shape = opset3.shape_of(query, "i32")
    k_shape = opset3.shape_of(key, "i32")
    minus_one = _i32(-1)
    minus_two = _i32(-2)
    zero_i = _i32(0)
    one_i = _i32(1)
    one_f = opset1.convert_like(one_i, query)
    zero_f = opset1.convert_like(zero_i, query)

    if n_in < 5:
        dim = _i32(-1)
        gathered = opset8.gather(q_shape, dim, zero_i)
        scale = opset1.convert_like(gathered, query)
        scale = opset1.divide(one_f, opset1.sqrt(scale))
    else:
        scale = node.input_value(4)

    k_rank = opset3.shape_of(k_shape, "i32")
    k_last_dim = opset1.add(k_rank, minus_one)
    k_next_dim = opset1.add(k_rank, minus_two)
    keep_dim_last = opset1.squeeze(k_next_dim, zero_i)
    k_dims_before_transpose = opset4.range(zero_i, keep_dim_last, one_i)
    transpose_dims = opset1.concat([k_dims_before_transpose, k_last_dim, k_next_dim], 0)
    k_transposed = opset1.transpose(key, transpose_dims)

    atten = opset1.matmul(query, k_transposed, False, False)
    scaled_atten = opset1.multiply(atten, scale)
    minus_inf = opset1.convert_like(_i32_f32_inf(), scaled_atten)

    if causal or n_in > 3:
        if not causal:
            mask = node.input_value(3)
            if mask.get_element_type() == ov.Type.boolean:
                atten_mask = opset1.select(mask, zero_f, minus_inf)
            else:
                atten_mask = mask
        else:
            target_s_len = opset8.gather(q_shape, _i32(-2), zero_i)
            source_s_len = opset8.gather(k_shape, _i32(-2), zero_i)
            ssl = opset1.unsqueeze(source_s_len, zero_i)
            tsl = opset1.unsqueeze(target_s_len, zero_i)
            mask_shape = opset1.concat([tsl, ssl], 0)
            mask = opset1.broadcast(minus_inf, mask_shape)
            horizontal = opset1.unsqueeze(opset4.range(zero_i, source_s_len, one_i), zero_i)
            vertical = opset1.unsqueeze(opset4.range(one_i, opset1.add(target_s_len, one_i), one_i), one_i)
            triu = opset1.greater_equal(horizontal, vertical)
            atten_mask = opset1.select(triu, mask, zero_f)
        scaled_atten = opset1.add(scaled_atten, atten_mask)

    scaled_atten = opset8.softmax(scaled_atten, -1)

    result = opset1.matmul(scaled_atten, value, False, False)
    result.set_friendly_name(node.get_friendly_name())
    return result


def _i32_f32_inf():
    return opset1.constant(np.float32(-np.inf))


def lower_above_max_opset(model, cap):
    """Replace ScaledDotProductAttention nodes whose opset is above `cap`.
    Anything else still above the cap is left for the xml check to name."""
    hot = [op for op in model.get_ordered_ops() if _opset(op) > cap and op.get_type_name() == "ScaledDotProductAttention"]
    for node in hot:
        ov_utils.replace_node(node, _decompose_sdpa(node))
    if hot:
        model.validate_nodes_and_infer_types()
        manager = Manager()
        manager.register_pass(ConstantFolding())
        manager.run_passes(model)
    return len(hot)


def layer_opsets(xml_path):
    """(type, opset) for every layer, the same version attribute the npu
    backend reads. The <net version> is the IR file version, not a layer."""
    found = []
    for el in ET.parse(xml_path).iter():
        if not (el.tag == "layer" or el.tag.endswith("}layer")):
            continue
        version = el.attrib.get("version", "")
        if not version.startswith("opset"):
            continue
        found.append((el.attrib.get("type", "?"), int(version[len("opset") :])))
    return found


OUTPUT_PRECISION = "FP16"


def f16_outputs(model):
    """Make each Result f16. compress_to_fp16 does not. The compiler's
    output argument precision is this port, via --outputs_precisions."""
    ppp = ov.preprocess.PrePostProcessor(model)
    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f16)
    return ppp.build()


def result_precisions(xml_path):
    """(name, precision) for every Result port, the attribute build_flags
    copies into --outputs_precisions."""
    found = []
    for el in ET.parse(xml_path).iter():
        if not (el.tag == "layer" or el.tag.endswith("}layer")):
            continue
        if el.attrib.get("type") != "Result":
            continue
        for port in el.iter():
            if port is el or not (port.tag == "port" or port.tag.endswith("}port")):
                continue
            prec = port.attrib.get("precision")
            if prec:
                found.append((el.attrib.get("name", "?"), prec))
    return found


def refuse_output_precision(xml_path, want):
    found = result_precisions(xml_path)
    if not found:
        sys.exit(f"{xml_path}: the IR xml has no Result precision")
    bad = [f"{name} {prec}" for name, prec in found if prec != want]
    if bad:
        sys.exit(f"IR Result precision is not {want}: " + ", ".join(bad))
    return found


def refuse_above(xml_path, cap):
    layers = layer_opsets(xml_path)
    if not layers:
        sys.exit(f"{xml_path}: the IR xml has no layer opset")
    over = {}
    for kind, opset in layers:
        if opset > cap:
            over[(kind, opset)] = over.get((kind, opset), 0) + 1
    if over:
        parts = [f"{kind} opset{opset} x{count}" for (kind, opset), count in sorted(over.items())]
        sys.exit(f"IR layer opset exceeds {cap}: " + ", ".join(parts))
    return max(opset for _, opset in layers)


# The bias a dropped key gets. exp(-100) underflows to 0 in the softmax.
# The same value the Hailo backend writes (core/hailo/embed.cpp).
MASKED = -100.0

# How far the cut graph may be from the export on a kept position, in f32,
# before the IR is saved. The same tolerance hef_compile.py uses.
CUT_TOLERANCE = 1e-4

# The CPU plugin's default hint lowers ScaledDotProductAttention. The
# static cut and the dynamic export then disagree by about 1e-2 on a
# kept position. f32 is the precision the tolerance is about.
CHECK_COMPILE = {"INFERENCE_PRECISION_HINT": "f32"}


def _producers(graph):
    return {o: n for n in graph.node for o in n.output}


def _word_gather(graph):
    """The Gather on input_ids: its data is the word-embedding table."""
    found = [n for n in graph.node if n.op_type == "Gather" and len(n.input) > 1 and n.input[1] == "input_ids"]
    if len(found) != 1:
        sys.exit(
            f"onnx_to_openvino_ir: {len(found)} Gathers read input_ids; a BERT export has one, the word embeddings"
        )
    return found[0]


def _attention_bias(graph):
    """The tensor every attention Softmax adds to its scores."""
    made = _producers(graph)
    biases = set()
    softmaxes = [n for n in graph.node if n.op_type == "Softmax"]
    if not softmaxes:
        sys.exit("onnx_to_openvino_ir: the export has no Softmax; the embeddings cut replaces the attention bias")
    for sm in softmaxes:
        add = made.get(sm.input[0]) if sm.input else None
        if add is None or add.op_type != "Add":
            sys.exit(f"onnx_to_openvino_ir: Softmax {sm.name} does not read an Add of the scores and a bias")
        scores = [i for i in add.input if i in made and made[i].op_type == "MatMul"]
        others = [i for i in add.input if i not in scores]
        if len(scores) != 1 or len(others) != 1:
            sys.exit(f"onnx_to_openvino_ir: Add {add.name} before a Softmax is not scores plus one bias")
        biases.add(others[0])
    if len(biases) != 1:
        sys.exit(
            f"onnx_to_openvino_ir: the Softmaxes add {len(biases)} different biases; a BERT export adds one mask"
        )
    return biases.pop()


def _prune(model):
    """Drop nodes and initializers that do not feed an output.

    The word Gather and the original mask are dead after the cut. Leaving
    them would keep the word table in the IR.
    """
    g = model.graph
    produced = _producers(g)
    live = set()
    stack = [o.name for o in g.output]
    while stack:
        name = stack.pop()
        if not name or name in live:
            continue
        live.add(name)
        node = produced.get(name)
        if node is None:
            continue
        for i in node.input:
            stack.append(i)
    kept = [n for n in g.node if any(o in live for o in n.output)]
    del g.node[:]
    g.node.extend(kept)
    kept_init = [t for t in g.initializer if t.name in live]
    del g.initializer[:]
    g.initializer.extend(kept_init)
    del g.value_info[:]


def cut_embeddings(src, seq, heads):
    """The export with word_rows and attn_bias as its inputs, fixed at [1, seq].

    The same cut bundle/hailo/hef_compile.py makes. Returns the cut model
    and the F32 word table the check gathers from.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(src)
    g = model.graph
    word = _word_gather(g)
    table_init = next((t for t in g.initializer if t.name == word.input[0]), None)
    if table_init is None:
        sys.exit("onnx_to_openvino_ir: the word Gather's table is not an initializer")
    table = numpy_helper.to_array(table_init).astype(np.float32)
    if table.ndim != 2:
        sys.exit(f"onnx_to_openvino_ir: the word table has rank {table.ndim}; it is [vocab, hidden]")
    hidden = int(table.shape[1])
    bias = _attention_bias(g)
    for name in ["input_ids", "attention_mask", "token_type_ids"]:
        inp = [i for i in g.input if i.name == name]
        if not inp:
            if name == "token_type_ids":
                continue
            sys.exit(f"onnx_to_openvino_ir: the export has no input {name}")
        if any(t.name == name for t in g.initializer):
            sys.exit(f"onnx_to_openvino_ir: {name} is already an initializer")
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
    _prune(model)
    onnx.checker.check_model(model)
    return model, table


def _compiled_output(compiled, values):
    feed = {}
    for inp in compiled.inputs:
        name = inp.get_any_name()
        if name not in values:
            sys.exit(f"onnx_to_openvino_ir: the cut check did not feed input {name}")
        feed[name] = values[name]
    result = compiled(feed)
    return np.array(result[compiled.output(0)])


def check_cut(export_path, cut_path, table, seq, heads):
    """The cut graph gives the export's hidden states on every kept position."""
    full = ov.convert_model(export_path)
    for inp in full.inputs:
        rank = len(inp.get_partial_shape())
        if rank != 2:
            sys.exit(f"input {inp.any_name}: rank {rank}; the export's inputs are [batch, seq]")
    full.reshape({inp.any_name: ov.PartialShape([1, seq]) for inp in full.inputs})
    full_c = ov.compile_model(full, "CPU", CHECK_COMPILE)
    part = ov.convert_model(cut_path)
    part_c = ov.compile_model(part, "CPU", CHECK_COMPILE)

    live_n = min(6, seq)
    if live_n < 1:
        sys.exit(f"--seq {seq}: the cut check needs a sequence")
    ids = np.zeros((1, seq), np.int64)
    ids[0, :live_n] = np.arange(1, live_n + 1)
    if int(ids.max()) >= table.shape[0]:
        sys.exit("onnx_to_openvino_ir: the cut check's ids are outside the word table")
    mask = np.zeros((1, seq), np.int64)
    mask[0, :live_n] = 1
    if live_n > 3:
        mask[0, 3] = 0
    full_names = {inp.any_name for inp in full.inputs}
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in full_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    a = _compiled_output(full_c, feed)
    rows = table[ids[0]].astype(np.float32)[None, :, :]
    keys = np.where(mask[0] == 1, 0.0, MASKED).astype(np.float32)
    bias = np.broadcast_to(keys, (1, heads, seq, seq)).astype(np.float32)
    b = _compiled_output(part_c, {"word_rows": rows, "attn_bias": np.array(bias)})
    kept = mask[0] == 1
    if not kept.any():
        sys.exit("onnx_to_openvino_ir: the cut check kept no position")
    worst = float(np.abs(a[0][kept] - b[0][kept]).max())
    if worst > CUT_TOLERANCE:
        sys.exit(
            f"onnx_to_openvino_ir: the cut graph is {worst} from the export on a kept position, over {CUT_TOLERANCE}"
        )
    return worst


def _convert(src, seq, batch, cut, heads):
    """The model to lower, and the extra produced_by settings for a cut."""
    if cut is None:
        if heads is not None:
            sys.exit("onnx_to_openvino_ir: --heads is only for --cut embeddings")
        model = ov.convert_model(src)
        shapes = {}
        for inp in model.inputs:
            rank = len(inp.get_partial_shape())
            if rank != 2:
                sys.exit(f"input {inp.any_name}: rank {rank}; the export's inputs are [batch, seq]")
            shapes[inp.any_name] = ov.PartialShape([batch, seq])
        model.reshape(shapes)
        return model, []
    if cut != "embeddings":
        sys.exit(f"onnx_to_openvino_ir: --cut {cut} is not embeddings")
    if heads is None or heads < 1:
        sys.exit("onnx_to_openvino_ir: --cut embeddings needs --heads of at least 1")

    import onnx

    cut_model, table = cut_embeddings(src, seq, heads)
    hidden = int(table.shape[1])
    # The source is mounted read-only. The cut file is a scratch copy.
    fd, cut_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        onnx.save(cut_model, cut_path)
        worst = check_cut(src, cut_path, table, seq, heads)
        model = ov.convert_model(cut_path)
    finally:
        os.remove(cut_path)
    shapes = {}
    seen = set()
    for inp in model.inputs:
        name = inp.any_name
        seen.add(name)
        rank = len(inp.get_partial_shape())
        if name == "word_rows":
            if rank != 3:
                sys.exit(f"input word_rows: rank {rank}; the cut's word rows are [batch, seq, hidden]")
            shapes[name] = ov.PartialShape([batch, seq, hidden])
        elif name == "attn_bias":
            if rank != 4:
                sys.exit(f"input attn_bias: rank {rank}; the cut's bias is [batch, heads, seq, seq]")
            shapes[name] = ov.PartialShape([batch, heads, seq, seq])
        else:
            sys.exit(f"input {name}: an embeddings cut takes word_rows and attn_bias")
    if seen != {"word_rows", "attn_bias"}:
        sys.exit(f"onnx_to_openvino_ir: the cut converted to {sorted(seen)}; it takes word_rows and attn_bias")
    model.reshape(shapes)
    return model, [
        "cut=embeddings",
        f"heads={heads}",
        f"masked={MASKED}",
        f"cut_max_abs_diff={worst:.3g}",
    ]


def main(src, out_xml, out_bin, produced_by_path, *rest):
    if len(rest) % 2 != 0:
        sys.exit("onnx_to_openvino_ir: arguments after the report path come in --name value pairs")
    opts = dict(zip(rest[0::2], rest[1::2]))
    unknown = [k for k in opts if k not in ("--seq", "--batch", "--max-opset", "--cut", "--heads")]
    if unknown:
        sys.exit("onnx_to_openvino_ir: unknown arguments " + " ".join(unknown))
    if "--seq" not in opts or "--batch" not in opts:
        sys.exit("onnx_to_openvino_ir: --seq and --batch are required")
    seq = int(opts["--seq"])
    batch = int(opts["--batch"])
    cap = int(opts.get("--max-opset", DEFAULT_MAX_OPSET))
    if cap < 1:
        sys.exit(f"--max-opset {cap}: the cap is a layer opset, at least 1")
    heads = int(opts["--heads"]) if "--heads" in opts else None
    model, extra = _convert(src, seq, batch, opts.get("--cut"), heads)
    lower_above_max_opset(model, cap)
    model = f16_outputs(model)

    # save_model writes the weights beside the xml under the xml's stem,
    # so it runs in a scratch directory on the same filesystem and the
    # two files are then moved to the paths the tool asked for.
    out_dir = os.path.dirname(os.path.abspath(out_xml))
    with tempfile.TemporaryDirectory(dir=out_dir) as d:
        xml = os.path.join(d, "model.xml")
        ov.save_model(model, xml, compress_to_fp16=COMPRESS_TO_FP16)
        highest = refuse_above(xml, cap)
        refuse_output_precision(xml, OUTPUT_PRECISION)
        os.replace(xml, out_xml)
        os.replace(os.path.join(d, "model.bin"), out_bin)

    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "openvino.save_model",
                "tool_version": ov.get_version(),
                "settings": [
                    f"seq={seq}",
                    f"batch={batch}",
                    f"max_opset={cap}",
                    f"compress_to_fp16={COMPRESS_TO_FP16}",
                    f"output_precision={OUTPUT_PRECISION}",
                    *extra,
                ],
            },
            f,
        )
    return highest


if __name__ == "__main__":
    main(*sys.argv[1:])
