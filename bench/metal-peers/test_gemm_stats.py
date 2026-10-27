"""Arithmetic regressions for the receipt writer; no MLX or device substitute."""

import unittest

from mlx_gemm import stats


class TimingStats(unittest.TestCase):
    def test_twenty_samples_use_the_average_of_the_middle_pair(self):
        got = stats(range(1, 21))
        self.assertEqual(got["median"], 10.5)
        self.assertEqual(got["mean"], 10.5)
        self.assertEqual(got["least"], 1)
        self.assertEqual(got["p99"], 20)

    def test_odd_and_single_samples(self):
        self.assertEqual(stats([9, 1, 5])["median"], 5)
        self.assertEqual(stats([3])["p99"], 3)

    def test_nearest_rank_p99_is_not_always_the_maximum(self):
        self.assertEqual(stats(range(1, 201))["p99"], 198)

    def test_samples_keep_original_order_and_precision(self):
        values = [0.0000234567, 0.0000123456]
        got = stats(values)
        self.assertEqual(got["samples"], values)
        self.assertEqual(got["least"], values[1])
        values.append(99)
        self.assertEqual(len(got["samples"]), 2)

    def test_invalid_timings_are_refused(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stats(values)


if __name__ == "__main__":
    unittest.main()
