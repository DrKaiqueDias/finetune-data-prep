# Fine-tune Data Prep

[![Python checks](https://github.com/DrKaiqueDias/finetune-data-prep/actions/workflows/tests.yml/badge.svg)](https://github.com/DrKaiqueDias/finetune-data-prep/actions)

Good training starts with examples you can trust. This is a small Python project for the step before fine-tuning: checking instruction/answer pairs, catching conflicting examples and keeping repeated prompts out of the validation set.

I kept the format simple so the rules are easy to inspect. It runs locally with Python 3.11+ and the standard library.

## Run it

```sh
python prepare.py examples/instructions.jsonl --out runs/demo --seed 42
python -m unittest discover -v
```

The example has 12 unique pairs. The default split gives **10 training examples and 2 validation examples**. The output folder must be new; use a different folder name for another run.

## Input and output

Each JSONL line has exactly two fields:

```json
{"instruction": "Explain why a validation set is useful.", "output": "It helps compare model choices using examples outside the training set."}
```

The command writes `train.jsonl`, `validation.jsonl` and `report.json`. The report records the seed, counts and SHA-256 hashes of the two output files.

- Empty values, unexpected fields, duplicate JSON keys and conflicting answers stop the run.
- Instructions are compared after Unicode NFKC normalization, case folding and whitespace normalization.
- Repeated instructions with exactly the same answer are kept once. Answer differences are sent back for review, not silently resolved.
- The same normalized instruction cannot appear in both splits.
- The seed and normalized instruction determine a SHA-256 sort order. Reordering the input leaves the split unchanged.
- Validation size is the requested fraction rounded down, with at least one record in each set.

`--validation-fraction 0.2`, `--seed 42` and `--max-chars 20000` are the defaults. The character limit applies to each field and counts Python string characters, not model tokens. Exit code **0** means success; **2** means invalid input or a file error.

## Where I would use it

Before converting a reviewed instruction dataset into a model-specific training format. This project prepares data; it does not submit a training job or claim compatibility with every provider.

The checks catch exact normalized prompt overlap, not paraphrases, related documents or examples from the same person. For those cases, split by source or group and review the result. The split is not stratified. Adding examples can change which records fall into each set.

Everything is held in memory. An interrupted disk write may leave a partial output folder; choose a new folder when retrying. Output datasets retain their original text, so review them before sharing. The report contains counts and hashes rather than example text.

## Example data

The examples in this repository are synthetic, written to exercise the workflow. They contain no employer datasets or customer conversations. They are too small to support conclusions about model quality.

## Changes

Open an issue with a small synthetic example, or send a pull request with a test for the behavior you changed. Run the tests before submitting. Code and bundled examples use the [MIT license](LICENSE).
