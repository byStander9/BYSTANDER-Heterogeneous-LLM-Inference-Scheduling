import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_server import ProxyServer  # noqa: E402


class MetricsCollectionTest(unittest.IsolatedAsyncioTestCase):

    def make_proxy(self) -> ProxyServer:
        proxy = ProxyServer.__new__(ProxyServer)
        proxy.backend_servers = [{
            "name": "gpu-1",
            "host": "127.0.0.1",
            "port": 8000,
        }]
        proxy.routing_algorithm = "slm_adaptive"
        proxy.inflight_tokens = {"gpu-1": deque(maxlen=500)}
        proxy._metrics_fail_log_count = 0
        return proxy

    async def test_failure_preserves_snapshot_until_collection_recovers(self):
        proxy = self.make_proxy()
        proxy._fetch_metrics = AsyncMock(side_effect=[
            {
                "num_requests_running": 100,
                "num_requests_waiting": 0,
                "inflight_prompt_token_lengths": list(range(100)),
            },
            None,
            {
                "num_requests_running": 20,
                "num_requests_waiting": 0,
                "inflight_prompt_token_lengths": list(range(20)),
            },
        ])

        first = await proxy._collect_all_metrics_direct()
        self.assertEqual(first["gpu-1"], {
            "num_requests_running": 100,
            "num_requests_waiting": 0,
        })
        self.assertEqual(len(proxy.inflight_tokens["gpu-1"]), 100)

        failed = await proxy._collect_all_metrics_direct()
        self.assertEqual(failed["gpu-1"], {
            "num_requests_running": -1,
            "num_requests_waiting": -1,
        })
        self.assertEqual(len(proxy.inflight_tokens["gpu-1"]), 100)

        recovered = await proxy._collect_all_metrics_direct()
        self.assertEqual(recovered["gpu-1"], {
            "num_requests_running": 20,
            "num_requests_waiting": 0,
        })
        self.assertEqual(list(proxy.inflight_tokens["gpu-1"]),
                         list(range(20)))
        self.assertEqual(proxy._fetch_metrics.await_count, 3)

    async def test_fetch_accepts_complete_vllm3_payload(self):
        proxy = self.make_proxy()
        response = Mock(status_code=200)
        response.json.return_value = {
            "status": "success",
            "engine_running_requests": 3.0,
            "engine_waiting_requests": 2.0,
            "inflight_prompt_token_lengths": [128, 256],
        }
        client = Mock()
        client.get = AsyncMock(return_value=response)
        proxy.get_metrics_http_client = AsyncMock(return_value=client)

        result = await proxy._fetch_metrics(proxy.backend_servers[0],
                                            quiet=True)

        self.assertEqual(result, {
            "num_requests_running": 3,
            "num_requests_waiting": 2,
            "inflight_prompt_token_lengths": [128, 256],
        })

    async def test_fetch_rejects_incomplete_or_error_payloads(self):
        invalid_payloads = [
            {
                "status": "error",
                "engine_running_requests": 0,
                "engine_waiting_requests": 0,
                "inflight_prompt_token_lengths": [],
            },
            {
                "status": "success",
                "engine_running_requests": 0,
                "engine_waiting_requests": 0,
            },
            {
                "status": "success",
                "engine_running_requests": 0,
                "engine_waiting_requests": 0,
                "inflight_prompt_token_lengths": {},
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                proxy = self.make_proxy()
                response = Mock(status_code=200)
                response.json.return_value = payload
                client = Mock()
                client.get = AsyncMock(return_value=response)
                proxy.get_metrics_http_client = AsyncMock(return_value=client)

                result = await proxy._fetch_metrics(proxy.backend_servers[0],
                                                    quiet=True)

                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
