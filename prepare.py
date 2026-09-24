"""Check instruction pairs and make repeatable train/validation splits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-standard JSON constant")


def load_pairs(path, max_chars=20000):
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    records, seen, duplicates = [], {}, 0
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=strict_object,
                                    parse_constant=reject_constant)
                if not isinstance(record, dict) or set(record) != {"instruction", "output"}:
                    raise ValueError("expected exactly instruction and output")
                for field in ("instruction", "output"):
                    text = record[field]
                    if not isinstance(text, str) or not normalize(text):
                        raise ValueError(f"{field} must be a non-empty string")
                    if len(text) > max_chars:
                        raise ValueError(f"{field} exceeds max_chars")
                    if any(unicodedata.category(char) == "Cs" for char in text):
                        raise ValueError("unpaired Unicode surrogate")
                key = normalize(record["instruction"])
                if key in seen:
                    if record["output"] != seen[key]["output"]:
                        raise ValueError("same normalized instruction has conflicting outputs")
                    duplicates += 1
                    if record["instruction"] < seen[key]["instruction"]:
                        seen[key] = record
                else:
                    seen[key] = record
            except (ValueError, TypeError, RecursionError) as error:
                raise ValueError(f"line {line_number}: invalid instruction pair ({type(error).__name__})") from error
    records = [seen[key] for key in sorted(seen)]
    if len(records) < 2:
        raise ValueError("at least two distinct instructions are required")
    return records, duplicates


def split_pairs(records, validation_fraction=0.2, seed=42):
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise ValueError("validation_fraction must be a number between 0 and 1")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if len(records) < 2:
        raise ValueError("at least two records are required")
    keys = [normalize(record["instruction"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("deduplicate instructions before splitting")
    def rank(record):
        key = normalize(record["instruction"])
        digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest()
        return digest, key
    ranked = sorted(records, key=rank)
    count = max(1, min(len(ranked) - 1, math.floor(len(ranked) * validation_fraction)))
    return ranked[count:], ranked[:count]


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def prepare(input_path, output_dir, validation_fraction=0.2, seed=42, max_chars=20000):
    records, duplicates = load_pairs(input_path, max_chars)
    train, validation = split_pairs(records, validation_fraction, seed)
    payloads = {
        "train.jsonl": "".join(encode_json(row) + "\n" for row in train),
        "validation.jsonl": "".join(encode_json(row) + "\n" for row in validation),
    }
    report = {
        "schema_version": 1, "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "unique_records": len(records), "duplicates_removed": duplicates,
        "train_records": len(train), "validation_records": len(validation),
        "normalized_instruction_overlap": 0,
        "sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                   for name, text in payloads.items()},
    }
    payloads["report.json"] = encode_json(report) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, text in payloads.items():
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(text)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=20000)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.input, args.out, args.validation_fraction, args.seed, args.max_chars)
    except (ValueError, OSError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
