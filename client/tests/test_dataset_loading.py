import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_request_qps import (load_dataset_auto,  # noqa: E402
                               resolve_dataset_path)


class DatasetLoadingTest(unittest.TestCase):

    def test_loads_sharegpt_json_array_with_existing_slice_semantics(self):
        records = [{
            "id": index,
            "conversations": [{
                "from": "human",
                "value": f"prompt-{index}",
            }],
        } for index in range(5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "sharegpt.json"
            dataset_path.write_text(json.dumps(records), encoding="utf-8")

            loaded = load_dataset_auto(str(dataset_path),
                                       start_index=2,
                                       limit=2)

        self.assertEqual([item["id"] for item in loaded], [2, 3])

    def test_loads_lmsys_jsonl_and_skips_invalid_lines(self):
        valid = {
            "conversation_id": "abc",
            "conversation": [{
                "role": "user",
                "content": "hello",
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "lmsys.jsonl"
            dataset_path.write_text(
                "not-json\n" + json.dumps(valid) + "\n", encoding="utf-8")

            loaded = load_dataset_auto(str(dataset_path))

        self.assertEqual(loaded, [valid])

    def test_resolves_dataset_alias_from_configured_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "lmsys_english_shuffled.json"
            dataset_path.write_text("[]", encoding="utf-8")

            resolved = resolve_dataset_path("lmsys", dataset_dir=temp_dir)

        self.assertEqual(resolved, dataset_path.resolve())

    def test_missing_dataset_lists_checked_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError,
                                        "BYSTANDER_DATASET_DIR"):
                resolve_dataset_path("missing.json", dataset_dir=temp_dir)


if __name__ == "__main__":
    unittest.main()
