"""An F16 copy of an F32 ONNX graph, for the reference programs that build
F16 only from a strongly typed graph (TensorRT's trtexec from 10 on, with
--stronglyTyped). The bundle tool runs it in the reference container, with
no network, on the ONNX file the bundle carries. The library never reads
either file.

Every float initializer and value becomes float16, except in the ops on
onnxconverter-common's default block list, which stay float32 with a
cast on each side. Converting clamps the constants (initializers and
Constant nodes): one above max_finite_val (1e4) in magnitude becomes
+-1e4, and a nonzero one below min_positive_val (1e-7) becomes +-1e-7.
Both are onnxconverter-common's defaults, passed here so the settings
name them. The inputs (token ids) and the output keep their types, with
a cast at each end, so a program feeds and reads the F16 graph as it
does the F32 one.

The converter leaves a Cast to FLOAT from an integer input as it was, so
a graph that casts the int64 attention mask to float and subtracts it
from 1.0 comes out as Sub(FLOAT16, FLOAT), which a strongly typed build
refuses. Each Cast to FLOAT that feeds converted ops is made a Cast to
FLOAT16 for them. Then the script checks the result: after shape
inference, no op but a Cast or one on the block list may take both FLOAT
and FLOAT16 inputs, or it fails naming the node.

Arguments: the F32 ONNX file, the file to write, and the JSON file that
says what ran.
"""

import json
import sys

import onnx
import onnxconverter_common
from onnxconverter_common import float16

KEEP_IO_TYPES = True
MAX_FINITE_VAL = 1e4
MIN_POSITIVE_VAL = 1e-7
SETTINGS = [
    f"keep_io_types={KEEP_IO_TYPES}",
    f"max_finite_val={MAX_FINITE_VAL!r}",
    f"min_positive_val={MIN_POSITIVE_VAL!r}",
    "float_casts_into_f16_ops=FLOAT16",
]

FLOAT = onnx.TensorProto.FLOAT
FLOAT16 = onnx.TensorProto.FLOAT16
# No op_block_list is passed, so the converter blocks its default list.
BLOCKED = set(float16.DEFAULT_OP_BLOCK_LIST)


def converted_op(node):
    """An op the converter made F16: not a Cast and not on the block list."""
    return node.op_type != "Cast" and node.op_type not in BLOCKED


def retype_float_casts(model):
    """Make each Cast to FLOAT that feeds converted ops a Cast to FLOAT16
    for them. A graph output keeps its FLOAT Cast; when some consumers
    are converted and some are not, the converted ones get a Cast of
    their own. Returns the number of Casts changed or added."""
    graph = model.graph
    outputs = {o.name for o in graph.output}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    changed = 0
    added = []
    for node in graph.node:
        to = next((a for a in node.attribute if a.name == "to"), None)
        if node.op_type != "Cast" or to is None or to.i != FLOAT:
            continue
        out = node.output[0]
        users = consumers.get(out, [])
        f16 = [u for u in users if converted_op(u)]
        if not f16:
            continue
        if out not in outputs and len(f16) == len(users):
            to.i = FLOAT16
            for vi in graph.value_info:
                if vi.name == out:
                    vi.type.tensor_type.elem_type = FLOAT16
        else:
            name = f"{out}_as_float16"
            added.append((out, onnx.helper.make_node("Cast", [node.input[0]], [name], name=name, to=FLOAT16)))
            for u in f16:
                for i, x in enumerate(u.input):
                    if x == out:
                        u.input[i] = name
        changed += 1
    for out, cast in added:
        at = next(i for i, n in enumerate(graph.node) if out in n.output)
        graph.node.insert(at + 1, cast)
    return changed


def mixed_float_inputs(model):
    """Each converted op that takes both FLOAT and FLOAT16 inputs, after
    shape inference, as a line naming it. The inference starts from no
    value_info, since the converter writes FLOAT16 there for values a
    Cast to FLOAT still makes."""
    bare = onnx.ModelProto()
    bare.CopyFrom(model)
    del bare.graph.value_info[:]
    g = onnx.shape_inference.infer_shapes(bare).graph
    types = {}
    for v in list(g.input) + list(g.value_info) + list(g.output):
        types[v.name] = v.type.tensor_type.elem_type
    for t in g.initializer:
        types[t.name] = t.data_type
    bad = []
    for i, node in enumerate(g.node):
        if not converted_op(node):
            continue
        if len({types.get(x) for x in node.input if x} & {FLOAT, FLOAT16}) > 1:
            kinds = [onnx.TensorProto.DataType.Name(types[x]) if x in types else "?" for x in node.input]
            bad.append(f"node {i} {node.name or node.op_type} ({node.op_type}): inputs {', '.join(kinds)}")
    return bad


def main(src, out, produced_by_path):
    model = onnx.load(src)
    converted = float16.convert_float_to_float16(
        model,
        min_positive_val=MIN_POSITIVE_VAL,
        max_finite_val=MAX_FINITE_VAL,
        keep_io_types=KEEP_IO_TYPES,
    )
    retype_float_casts(converted)
    bad = mixed_float_inputs(converted)
    if bad:
        sys.exit("the F16 graph mixes FLOAT and FLOAT16 inputs in an op:\n" + "\n".join(bad))
    onnx.checker.check_model(converted)
    onnx.save(converted, out)
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "onnxconverter-common",
                "tool_version": f"{onnxconverter_common.__version__} (onnx {onnx.__version__})",
                "settings": SETTINGS,
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
