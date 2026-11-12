"""Run in the pinned DFC image: python -m unittest test_cut_validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models

from hef_compile import check_cut


def graph(path, values, cut):
    """Actual ONNX graphs exercise both runtime sessions, without a shim."""
    inputs = (
        [("word_rows", TensorProto.FLOAT, [1, 1, 3]),
         ("attn_bias", TensorProto.FLOAT, [1, 1, 1, 1])]
        if cut else
        [("input_ids", TensorProto.INT64, [1, 1]),
         ("attention_mask", TensorProto.INT64, [1, 1])]
    )
    values = np.asarray(values, dtype=np.float32)
    node = helper.make_node("Constant", [], ["states"], value=numpy_helper.from_array(values))
    model = helper.make_model(helper.make_graph(
        [node], "validation", [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info("states", TensorProto.FLOAT, list(values.shape))],
    ), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


class CutValidation(unittest.TestCase):
    def test_empty_calibration_cannot_report_success(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2, 3]]], [[[1, 2, 3]]], count=0, row_count=0)

    def test_mismatched_calibration_counts_refuse(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2, 3]]], [[[1, 2, 3]]], row_count=2)

    def compare(self, reference, candidate, live=True, count=1, row_count=1):
        with tempfile.TemporaryDirectory() as directory:
            full = str(Path(directory) / "full.onnx")
            cut = str(Path(directory) / "cut.onnx")
            graph(full, reference, False)
            graph(cut, candidate, True)
            tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))
            return check_cut(full, cut, tok, ["x"] * count, np.zeros((row_count, 1, 1, 3), np.float32),
                             np.zeros((count, 1, 1, 1), np.float32), np.full((count, 1), int(live)))

    def test_finite_identical_states_pass(self):
        self.assertEqual(self.compare([[[1, 2, 3]]], [[[1, 2, 3]]]), 0.0)

    def test_nan_reference_is_not_zero_difference(self):
        with self.assertRaises(SystemExit):
            self.compare([[[float("nan"), 2, 3]]], [[[1, 2, 3]]])

    def test_nan_candidate_is_not_zero_difference(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2, 3]]], [[[float("nan"), 2, 3]]])

    def test_matching_infinities_do_not_pass(self):
        with self.assertRaises(SystemExit):
            self.compare([[[float("inf"), 2, 3]]], [[[float("inf"), 2, 3]]])

    def test_logits_cannot_masquerade_as_hidden_states(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2]]], [[[1, 2]]])

    def test_extra_batch_is_not_silently_discarded(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2, 3]], [[4, 5, 6]]], [[[1, 2, 3]], [[4, 5, 6]]])

    def test_no_live_positions_refuses_explicitly(self):
        with self.assertRaises(SystemExit) as refusal:
            self.compare([[[1, 2, 3]]], [[[1, 2, 3]]], live=False)
        self.assertIn("no kept positions", str(refusal.exception))

    def test_finite_difference_still_obeys_existing_bound(self):
        with self.assertRaises(SystemExit):
            self.compare([[[1, 2, 3]]], [[[1, 2, 4]]])

    def test_large_finite_difference_does_not_overflow_comparison(self):
        with self.assertRaises(SystemExit) as refusal:
            self.compare([[[3e38, 2, 3]]], [[[-3e38, 2, 3]]])
        self.assertIn("over", str(refusal.exception))
        self.assertNotIn("inf", str(refusal.exception))


if __name__ == "__main__":
    unittest.main()
