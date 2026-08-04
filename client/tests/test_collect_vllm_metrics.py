import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_vllm_metrics import snapshot_from_text  # noqa: E402


class CollectVllmMetricsTest(unittest.TestCase):

    def test_builds_snapshot_and_sums_labeled_counters(self):
        metrics = """
process_start_time_seconds 123.5
vllm:num_requests_running{engine="0"} 3.0
vllm:num_requests_waiting{engine="0"} 7.0
vllm:kv_cache_usage_perc{engine="0"} 0.125
vllm:prompt_tokens_total{engine="0"} 100.0
vllm:generation_tokens_total{engine="0"} 200.0
vllm:request_success_total{finished_reason="stop"} 4.0
vllm:request_success_total{finished_reason="length"} 5.0
vllm:num_preemptions_total{engine="0"} 2.0
"""

        snapshot = snapshot_from_text(metrics)

        self.assertEqual(snapshot["replica_id"], "123.500")
        self.assertEqual(snapshot["num_running"], 3.0)
        self.assertEqual(snapshot["num_waiting"], 7.0)
        self.assertEqual(snapshot["kv_cache_usage_perc"], 0.125)
        self.assertEqual(snapshot["request_success_total"], 9.0)
        self.assertEqual(snapshot["num_preemptions_total"], 2.0)


if __name__ == "__main__":
    unittest.main()
