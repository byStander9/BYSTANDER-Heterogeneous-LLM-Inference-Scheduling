import asyncio
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kt_rr_proxy import RoundRobinRouter  # noqa: E402


class RoundRobinRouterTest(unittest.TestCase):

    def test_selects_alternating_endpoints(self):
        router = RoundRobinRouter(["https://npu-1/", "https://npu-2/"])

        selections = [asyncio.run(router.select()) for _ in range(4)]

        self.assertEqual([row[1] for row in selections], [0, 1, 0, 1])
        self.assertEqual(router.forwarded, [2, 2])

    def test_explicit_request_id_preserves_client_rr_assignment(self):
        router = RoundRobinRouter(["https://npu-1", "https://npu-2"])

        request_id, index, endpoint = asyncio.run(router.select("7"))

        self.assertEqual(request_id, 7)
        self.assertEqual(index, 1)
        self.assertEqual(endpoint, "https://npu-2")


if __name__ == "__main__":
    unittest.main()
