import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from prepare import load_pairs, main, normalize, prepare, split_pairs


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.rows = [{"instruction": f"Question {index}", "output": f"Answer {index}"}
                     for index in range(10)]
        self.write(self.rows)

    def write(self, rows):
        self.source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_split_is_disjoint_complete_and_sized(self):
        train, validation = split_pairs(self.rows)
        self.assertEqual((len(train), len(validation)), (8, 2))
        self.assertEqual({row["instruction"] for row in train + validation},
                         {row["instruction"] for row in self.rows})
        self.assertFalse({normalize(row["instruction"]) for row in train}
                         & {normalize(row["instruction"]) for row in validation})

    def test_seed_and_input_order(self):
        self.assertEqual(split_pairs(self.rows), split_pairs(list(reversed(self.rows))))
        self.assertNotEqual(split_pairs(self.rows, seed=1), split_pairs(self.rows, seed=2))

    def test_unicode_deduplication_is_deterministic(self):
        rows = [{"instruction": " Ｈｅｌｌｏ  World ", "output": "same"},
                {"instruction": "hello world", "output": "same"}, self.rows[0]]
        self.write(rows)
        expected = load_pairs(self.source)
        self.write(list(reversed(rows)))
        self.assertEqual(load_pairs(self.source), expected)
        self.assertEqual(expected[1], 1)

    def test_conflicting_answers_rejected(self):
        self.write([self.rows[0], {**self.rows[0], "output": "different"}])
        with self.assertRaises(ValueError):
            load_pairs(self.source)

    def test_invalid_schema_types_and_empty_fields(self):
        for row in ([], {}, {"instruction": "x", "output": "y", "extra": 1},
                    {"instruction": 42, "output": "y"},
                    {"instruction": " ", "output": "y"},
                    {"instruction": "x", "output": ""},
                    {"instruction": "x", "output": "\ud800"}):
            with self.subTest(row=repr(row)):
                self.write([row, self.rows[0]])
                with self.assertRaises(ValueError):
                    load_pairs(self.source)

    def test_duplicate_json_keys_and_constants(self):
        for content in ('{"instruction":"x","instruction":"y","output":"z"}',
                        '{"instruction":"x","output":NaN}', '{broken'):
            self.source.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pairs(self.source)

    def test_blank_lines_and_bom(self):
        self.source.write_text("\ufeff\n" + "\n\n".join(json.dumps(row) for row in self.rows), encoding="utf-8")
        self.assertEqual(len(load_pairs(self.source)[0]), 10)

    def test_small_dataset_always_keeps_both_sets(self):
        for fraction in (0.001, 0.999):
            train, validation = split_pairs(self.rows[:2], fraction)
            self.assertEqual((len(train), len(validation)), (1, 1))

    def test_invalid_split_settings(self):
        for fraction in (-1, 0, 1, float("nan"), float("inf"), True, "0.2"):
            with self.assertRaises(ValueError):
                split_pairs(self.rows, fraction)
        with self.assertRaises(ValueError):
            split_pairs(self.rows, seed=True)
        with self.assertRaises(ValueError):
            split_pairs(self.rows + [self.rows[0]])

    def test_length_limit_and_insufficient_data(self):
        with self.assertRaises(ValueError):
            load_pairs(self.source, max_chars=2)
        self.write([self.rows[0]])
        with self.assertRaises(ValueError):
            load_pairs(self.source)

    def test_output_hashes_match_files_and_repeat(self):
        report = prepare(self.source, self.root / "first")
        prepare(self.source, self.root / "second")
        for name in ("train.jsonl", "validation.jsonl", "report.json"):
            data = (self.root / "first" / name).read_bytes()
            self.assertEqual(data, (self.root / "second" / name).read_bytes())
            if name in report["sha256"]:
                self.assertEqual(hashlib.sha256(data).hexdigest(), report["sha256"][name])

    def test_output_never_overwrites_existing_directory(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare(self.source, output)
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(len(list(output.iterdir())), 1)

    def test_invalid_input_creates_no_output_or_content_leak(self):
        self.source.write_text('{"instruction":"PRIVATE_EXAMPLE","output":null}', encoding="utf-8")
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main([str(self.source), "--out", str(self.root / "failed")])
        self.assertEqual(code, 2)
        self.assertNotIn("PRIVATE_EXAMPLE", error.getvalue())
        self.assertFalse((self.root / "failed").exists())

    def test_cli_success(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.source), "--out", str(self.root / "result")]), 0)
        self.assertEqual(json.loads(output.getvalue())["unique_records"], 10)


if __name__ == "__main__":
    unittest.main()
