"""A static-shape OpenVINO IR from the upstream F32 ONNX export, for the
npu backend (docs/npu.md): the Intel NPU driver's own compiler builds the
graph on the machine from the IR's two files, so the library links no
OpenVINO; this conversion runs only while a bundle is made, in the
reference container, with no network.

The export's inputs (token ids, the attention mask, and token type ids
where the export takes them) are reshaped to one fixed [batch, seq],
because the NPU compiles static shapes: a dynamic IR fails in the
driver's compiler. The weights are compressed to FP16
(compress_to_fp16=True, the artifact's DTYPE_F16); the inputs and the
output keep their types, so the ids stay integer and the hidden states
come back F32.

convert_model fuses attention into ScaledDotProductAttention, which is
opset 13. The Arrow Lake driver reports maxOVOpsetVersionSupported 11,
and the backend refuses an IR whose highest layer opset is above that.
The Python package does not wrap
ov::pass::ScaledDotProductAttentionDecomposition, so this script builds
the same subgraph that pass builds in OpenVINO 2026.3 (ops at opset 8
or below) and then refuses the saved xml if any layer is still above
the cap. Retagging a layer's version would leave an op the compiler
does not implement.

Arguments: the source ONNX file, the output xml, the output bin, the
produced_by report path, then --seq N --batch B, and optionally
--max-opset N (default 11).
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


def main(src, out_xml, out_bin, produced_by_path, *rest):
    opts = dict(zip(rest[0::2], rest[1::2]))
    seq = int(opts["--seq"])
    batch = int(opts["--batch"])
    cap = int(opts.get("--max-opset", DEFAULT_MAX_OPSET))
    if cap < 1:
        sys.exit(f"--max-opset {cap}: the cap is a layer opset, at least 1")

    model = ov.convert_model(src)
    shapes = {}
    for inp in model.inputs:
        rank = len(inp.get_partial_shape())
        if rank != 2:
            sys.exit(f"input {inp.any_name}: rank {rank}; the export's inputs are [batch, seq]")
        shapes[inp.any_name] = ov.PartialShape([batch, seq])
    model.reshape(shapes)
    lower_above_max_opset(model, cap)

    # save_model writes the weights beside the xml under the xml's stem,
    # so it runs in a scratch directory on the same filesystem and the
    # two files are then moved to the paths the tool asked for.
    out_dir = os.path.dirname(os.path.abspath(out_xml))
    with tempfile.TemporaryDirectory(dir=out_dir) as d:
        xml = os.path.join(d, "model.xml")
        ov.save_model(model, xml, compress_to_fp16=COMPRESS_TO_FP16)
        highest = refuse_above(xml, cap)
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
                ],
            },
            f,
        )
    return highest


if __name__ == "__main__":
    main(*sys.argv[1:])
