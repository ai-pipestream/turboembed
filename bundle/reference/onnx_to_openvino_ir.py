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

Arguments: the source ONNX file, the output xml, the output bin, the
produced_by report path, then --seq N --batch B.
"""

import json
import os
import sys
import tempfile

import openvino as ov

COMPRESS_TO_FP16 = True


def main(src, out_xml, out_bin, produced_by_path, *rest):
    opts = dict(zip(rest[0::2], rest[1::2]))
    seq = int(opts["--seq"])
    batch = int(opts["--batch"])

    model = ov.convert_model(src)
    shapes = {}
    for inp in model.inputs:
        rank = len(inp.get_partial_shape())
        if rank != 2:
            sys.exit(f"input {inp.any_name}: rank {rank}; the export's inputs are [batch, seq]")
        shapes[inp.any_name] = ov.PartialShape([batch, seq])
    model.reshape(shapes)

    # save_model writes the weights beside the xml under the xml's stem,
    # so it runs in a scratch directory on the same filesystem and the
    # two files are then moved to the paths the tool asked for.
    out_dir = os.path.dirname(os.path.abspath(out_xml))
    with tempfile.TemporaryDirectory(dir=out_dir) as d:
        xml = os.path.join(d, "model.xml")
        ov.save_model(model, xml, compress_to_fp16=COMPRESS_TO_FP16)
        os.replace(xml, out_xml)
        os.replace(os.path.join(d, "model.bin"), out_bin)

    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "openvino.save_model",
                "tool_version": ov.get_version(),
                "settings": [f"seq={seq}", f"batch={batch}", f"compress_to_fp16={COMPRESS_TO_FP16}"],
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
