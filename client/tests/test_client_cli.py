import asyncio
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_request_qps import (RequestResult,  # noqa: E402
                               build_argument_parser,
                               build_chat_completions_url, run_experiment,
                               save_results)


class ClientCliTest(unittest.TestCase):

    def test_dataset_and_legacy_sharegpt_flags_share_the_same_destination(self):
        parser = build_argument_parser()

        dataset_args = parser.parse_args(["--dataset", "lmsys"])
        legacy_args = parser.parse_args(["--sharegpt", "sharegpt.json"])

        self.assertEqual(dataset_args.dataset, "lmsys")
        self.assertEqual(legacy_args.dataset, "sharegpt.json")

    def test_rejects_invalid_qps_and_request_counts(self):
        parser = build_argument_parser()
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--qps", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--total", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--max-concurrent", "-1"])

    def test_base_url_builds_https_chat_completions_url(self):
        parser = build_argument_parser()
        args = parser.parse_args([
            "--base-url",
            "https://qwen3-4b.proxy.ainexus.ktcloud.com/",
        ])

        self.assertEqual(
            build_chat_completions_url(args),
            "https://qwen3-4b.proxy.ainexus.ktcloud.com/v1/chat/completions",
        )

    def test_accepts_positive_max_tokens(self):
        parser = build_argument_parser()
        args = parser.parse_args(["--max-tokens", "256"])

        self.assertEqual(args.max_tokens, 256)

    def test_dry_run_validates_dataset_without_contacting_proxy(self):
        record = {
            "conversations": [{
                "from": "human",
                "value": "hello",
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "sharegpt.json"
            dataset_path.write_text(json.dumps([record]), encoding="utf-8")
            args = build_argument_parser().parse_args([
                "--dataset",
                str(dataset_path),
                "--total",
                "1",
                "--dry-run",
            ])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                asyncio.run(run_experiment(args))

        self.assertIn("프록시에는 요청을 보내지 않았습니다", stdout.getvalue())

    def test_result_writer_creates_parent_directory(self):
        result = RequestResult(
            request_id=1,
            http_status=200,
            start_time="2026-01-01 00:00:00",
            end_time="2026-01-01 00:00:01",
            latency_e2e_ms=1000,
            latency_ttft_ms=100,
            tokens_generated=10,
            prompt_tokens=5,
            prompt="[]",
            error=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "result.csv"
            save_results([result], str(output_path), "start", "end", 1.0)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
