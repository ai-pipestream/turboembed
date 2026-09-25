"""An F16 copy of an F32 ONNX graph, for the reference programs that build
F16 only from a strongly typed graph (TensorRT's trtexec from 10 on, with
--stronglyTyped). The bundle tool runs it in the reference container, with
no network, on the ONNX file the bundle carries. The library never reads
either file.

Every float initializer and value becomes float16; the inputs (token ids)
and the output keep their types, with a cast at each end, so a program
feeds and reads the F16 graph as it does the F32 one.

Arguments: the F32 ONNX file, the file to write, and the JSON file that
says what ran.
"""

import json
import sys

import onnx
import onnxconverter_common
from onnxconverter_common import float16

SETTINGS = ["keep_io_types=True"]


def main(src, out, produced_by_path):
    model = onnx.load(src)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
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
