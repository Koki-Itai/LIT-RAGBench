# Benchmarking Generator Capabilities of Large Language Models in Retrieval-Augmented Generation

## Overview

This repository provides the supplementary materials for our research on evaluating the generation capabilities of Large Language Models (LLMs) in Retrieval-Augmented Generation.

## Repository Structure

### 1. `datasets/`

Contains the evaluation datasets in JSONL format:

- **`ja.jsonl`**: Japanese evaluation dataset
- **`en.jsonl`**: English evaluation dataset


Each instance includes:
- `question`
- `answer`: Ground truth reference answer
- `qa_type`: Question type classification
- `positive_chunk_list`: Relevant chunks that contain evidence for generating the answer
- `negative_chunk_list`: Irelevant chunks that do not contain evidence
- `reasoning_content`: Detailed reasoning process for deriving the answer

### 2. `prompts/`

#### `dataset_constructoin/`

- **`step1.txt`**: Initial scenario generation prompt
- **`step2.txt`**: Context generation prompt
- **`step3.txt`**: Question-answer pair generation prompt

#### `prompts/evaluation/`

- **`generate_en.txt`** / **`generate_ja.txt`**: System prompts for instructing the evaluated LLM to generate answers based on retrieved documents

- **`judge_en.txt`** / **`judge_ja.txt`**: System prompts for the evaluator model to assess answer correctness

The generation prompts instruct models to synthesize information from provided document chunks and cite sources appropriately. The judge prompts guide an LLM evaluator to score generated answers against reference answers on a binary scale (correct/incorrect).

### 3. `src/`

Reference implementation for running the evaluation:

Set the `OPENAI_API_KEY` environment variable or create a `.env` file with your API key.

```bash
uv sync
uv run python src/run.py --lang en --num-tasks 5
```

