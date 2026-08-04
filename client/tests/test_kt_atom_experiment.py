import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kt_atom_experiment import (PreparedRequest, histogram_delta,  # noqa: E402
                                histogram_quantile, parse_prometheus,
                                percentile, select_compatible_requests,
                                state_features)


class KtAtomExperimentTest(unittest.TestCase):

    def test_parses_and_deltas_labeled_vllm_histogram(self):
        start = parse_prometheus("""
process_start_time_seconds 10
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="1.0"} 2
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="2.0"} 4
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="+Inf"} 5
vllm:e2e_request_latency_seconds_sum{model_name="q"} 6
vllm:e2e_request_latency_seconds_count{model_name="q"} 5
""")
        end = parse_prometheus("""
process_start_time_seconds 10
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="1.0"} 3
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="2.0"} 7
vllm:e2e_request_latency_seconds_bucket{model_name="q",le="+Inf"} 9
vllm:e2e_request_latency_seconds_sum{model_name="q"} 13
vllm:e2e_request_latency_seconds_count{model_name="q"} 9
""")

        delta = histogram_delta(start, end, "vllm:e2e_request_latency_seconds")
        estimate, lower, upper = histogram_quantile(delta, 0.50)

        self.assertEqual(delta["count"], 4)
        self.assertEqual(delta["sum"], 7)
        self.assertAlmostEqual(estimate, 1.5)
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 2.0)
        self.assertEqual(delta["buckets"][math.inf], 4)

    def test_client_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_state_features_match_paper_and_proxy_percentile_convention(self):
        state = state_features(list(range(1, 101)))
        self.assertEqual(state["inflight_count"], 100)
        self.assertEqual(state["inflight_p99_tokens"], 100)
        self.assertEqual(state["inflight_p90_tokens"], 91)
        self.assertEqual(state["inflight_p25_tokens"], 26)

    def test_oversized_prompts_are_skipped_and_ids_are_reassigned(self):
        requests = [
            PreparedRequest(0, 10, [], "a", prompt_tokens=100),
            PreparedRequest(1, 11, [], "b", prompt_tokens=50000),
            PreparedRequest(2, 12, [], "c", prompt_tokens=200),
        ]

        selected, skipped = select_compatible_requests(
            requests, total=2, max_model_len=40960, max_tokens=64)

        self.assertEqual([item.dataset_index for item in selected], [10, 12])
        self.assertEqual([item.request_id for item in selected], [0, 1])
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
