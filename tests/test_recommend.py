import math
import unittest

from flexft import MethodBenchmark, MethodRecommendation, recommend_method


class RecommendMethodTests(unittest.TestCase):
    def test_estimate_prefers_direct_for_one_output(self):
        recommendation = recommend_method(4096, 1, cache=False)

        self.assertIsInstance(recommendation, MethodRecommendation)
        self.assertEqual(recommendation.method, "direct")
        self.assertEqual(recommendation.mode, "estimate")
        self.assertEqual(recommendation.device, "not benchmarked")
        self.assertIsInstance(recommendation.results["direct"], MethodBenchmark)
        self.assertEqual(recommendation.results["direct"].score_unit, "work_units")

    def test_estimate_prefers_bluestein_for_square_transform(self):
        recommendation = recommend_method(256, 256, cache=False)
        self.assertEqual(recommendation.method, "bluestein")

    def test_direct_memory_guard_is_reported(self):
        recommendation = recommend_method(16, 4, max_direct_bytes=0, cache=False)

        direct = recommendation.results["direct"]
        self.assertEqual(recommendation.method, "bluestein")
        self.assertTrue(math.isinf(direct.score))
        self.assertIn("max_direct_bytes", direct.skipped_reason)

    def test_identical_recommendations_are_cached(self):
        first = recommend_method(37, 3)
        second = recommend_method(37, 3)
        fresh = recommend_method(37, 3, cache=False)

        self.assertIs(first, second)
        self.assertIsNot(first, fresh)

    def test_benchmark_reports_synchronized_timings(self):
        recommendation = recommend_method(
            16,
            3,
            mode="benchmark",
            expected_calls=2,
            warmup=1,
            repeats=2,
            cache=False,
        )

        self.assertIn(recommendation.method, ("direct", "bluestein"))
        self.assertNotEqual(recommendation.device, "not benchmarked")
        for result in recommendation.results.values():
            self.assertIsNone(result.skipped_reason)
            self.assertEqual(result.score_unit, "seconds")
            self.assertGreaterEqual(result.setup_time, 0.0)
            self.assertGreaterEqual(result.compilation_time, 0.0)
            self.assertGreater(result.execution_time, 0.0)
            self.assertGreaterEqual(result.score, 2 * result.execution_time)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            recommend_method(8, mode="automatic")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            recommend_method(8, batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_direct_bytes"):
            recommend_method(8, max_direct_bytes=-1)
        with self.assertRaisesRegex(TypeError, "dtype"):
            recommend_method(8, dtype="int32")


if __name__ == "__main__":
    unittest.main()
