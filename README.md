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

Prompts used to assist in dataset construction with LLM (GPT-5).

- **`qa_scenario.txt`**: Initial QA scenario generation prompt
- **`q_c_a.txt`**: Question, positive chunks, reasoning, and answer generation prompt
- **`negative_chunks.txt`**: Irrelevant document generation prompt

#### `prompts/evaluation/`

Prompts for evaluating LLM performance in this benchmark via LLM-as-a-Judge.

- **`generate_en.txt`** / **`generate_ja.txt`**: System prompts for instructing the evaluated LLM to generate answers based on retrieved documents

- **`judge_en.txt`** / **`judge_ja.txt`**: System prompts for the judge model to assess answer correctness

### 3. `src/`

Example code for running the benchmark evaluation:

Set the `OPENAI_API_KEY` environment variable or create a `.env` file with your API key.

```bash
uv sync
uv run python src/run.py --lang en
```

