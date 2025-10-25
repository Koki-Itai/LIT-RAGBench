import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent

GENERATE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"


class RAGTask:
    def __init__(
        self,
        question: str,
        reference_answer: str,
        positive_context: list[str],
        negative_context: list[str],
        qa_type: list[str],
        reasoning_content: str = "",
        language: str = "ja",
    ) -> None:
        self.question = question
        self.reference_answer = reference_answer
        self.positive_context = positive_context
        self.negative_context = negative_context
        self.qa_type = qa_type
        self.reasoning_content = reasoning_content
        self.generated_answer = ""
        self.language = language

        self._load_prompts()

    def _load_prompts(self) -> None:
        prompt_dir = PROJECT_ROOT / "prompts" / "evaluation"
        generate_prompt_path = prompt_dir / f"generate_{self.language}.txt"
        with open(generate_prompt_path, "r", encoding="utf-8") as f:
            self.generate_prompt = f.read()

        judge_prompt_path = prompt_dir / f"judge_{self.language}.txt"
        with open(judge_prompt_path, "r", encoding="utf-8") as f:
            self.judge_prompt = f.read()

    def create_generate_messages(self) -> list[dict[str, str]]:
        combined_doc = self.positive_context + self.negative_context
        random.shuffle(combined_doc)
        combined_doc = [f"[cite:{i + 1}]{item}" for i, item in enumerate(combined_doc)]

        document_prompt = "<DOCUMENTS>" + "\n".join(combined_doc) + "</DOCUMENTS>"
        user_question = f"\n\n<QUESTION>:\n{self.question}\n</QUESTION>"
        user_prompt = document_prompt + user_question

        return [
            {"role": "system", "content": self.generate_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def create_evaluate_messages(self) -> list[dict[str, str]]:
        output_format = {
            "score": "Score to evaluate the accuracy of the answer (int, 0 or 1)",
            "evaluation_reason": "String explaining the reason for the evaluation (str)",
        }

        newline = "\n"
        user_prompt = (
            f"<QUESTION>\n{self.question}\n\n"
            f"<REFERENCE_REASONING_CONTENT>\n{self.reasoning_content}\n\n"
            f"<REFERENCE_ANSWER>\n{self.reference_answer}\n\n"
            f"<POSITIVE_CONTEXT>\n{newline.join(self.positive_context)}\n\n"
            "--- The following is the answer to be evaluated ---\n"
            f"<GENERATED_ANSWER>\n{self.generated_answer}\n\n"
        )

        judge_prompt_with_format = (
            f"{self.judge_prompt}\n"
            f"<OUTPUT_FORMAT>\n"
            f"Output in the following JSON format:\n"
            f"{json.dumps(output_format, indent=2, ensure_ascii=False)}\n"
            f"</OUTPUT_FORMAT>\n"
        )

        return [
            {"role": "system", "content": judge_prompt_with_format},
            {"role": "user", "content": user_prompt},
        ]


def load_tasks_from_local(language: str = "ja") -> list[RAGTask]:
    dataset_path = PROJECT_ROOT / "datasets" / f"{language}.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")

    tasks = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                positive_context = row.get("positive_chunk_list", [])
                if positive_context and isinstance(positive_context[0], dict):
                    positive_context = [
                        item.get("content", "") for item in positive_context
                    ]
                negative_context = row.get("negative_chunk_list", [])
                if negative_context and isinstance(negative_context[0], dict):
                    negative_context = [
                        item.get("content", "") for item in negative_context
                    ]

                task = RAGTask(
                    question=row.get("question", ""),
                    reference_answer=row.get("answer", ""),
                    positive_context=positive_context,
                    negative_context=negative_context,
                    qa_type=row.get("qa_type", []),
                    reasoning_content=row.get("reasoning_content", ""),
                    language=language,
                )
                tasks.append(task)

    print(f"✓ Loaded {len(tasks)} tasks")
    return tasks


def generate_answer(client: OpenAI, task: RAGTask) -> str:
    messages = task.create_generate_messages()

    response = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=4096,
    )

    answer = response.choices[0].message.content
    return answer


def evaluate_answer(client: OpenAI, task: RAGTask) -> dict[str, Any]:
    messages = task.create_evaluate_messages()

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    eval_result = json.loads(response.choices[0].message.content)
    return eval_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to run RAG benchmark evaluation"
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["ja", "en"],
        default="ja",
        help="Language to use (ja: Japanese, en: English). Default: ja",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks to evaluate. Default: 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    print(f"Language: {args.lang}")
    tasks = load_tasks_from_local(language=args.lang)

    num_tasks_to_evaluate = min(args.num_tasks, len(tasks))
    print(f"\nEvaluating {num_tasks_to_evaluate} tasks...")
    print(f"Generation model: {GENERATE_MODEL}")
    print(f"Judge model: {JUDGE_MODEL}")
    print("=" * 80)

    results = []
    total_score = 0

    for task in tqdm(tasks[:num_tasks_to_evaluate], desc="Evaluating tasks"):
        generated_answer = generate_answer(client, task)
        task.generated_answer = generated_answer

        eval_result = evaluate_answer(client, task)
        score = eval_result.get("score", 0)
        reason = eval_result.get("evaluation_reason", "")

        results.append(
            {
                "question": task.question,
                "qa_type": task.qa_type,
                "reference_answer": task.reference_answer,
                "generated_answer": generated_answer,
                "score": score,
                "evaluation_reason": reason,
            }
        )

        total_score += score

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print(
        f"Total score: {total_score}/{num_tasks_to_evaluate} ({total_score / num_tasks_to_evaluate * 100:.1f}%)"
    )
    print("=" * 80)

    results_dir = PROJECT_ROOT / "results" / GENERATE_MODEL
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"{args.lang}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
