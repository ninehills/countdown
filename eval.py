#!/usr/bin/env python3
"""
python eval.py --provider <provider> --data_path <data_path> --model_name <model_name>
"""
import re
import os
import argparse
from typing import Tuple
from dotenv import load_dotenv
from bespokelabs import curator
from datasets import load_dataset

from pydantic import BaseModel

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."""

USER_PROMPT_TPL = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once, and you must use all the numbers. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>. Think step by step inside <think> tags."""

REASONING_USER_PROMPT_TPL = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once, and you must use all the numbers. Return the final equation in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."""


class EvalResult(BaseModel):
    """Result of the judge's evaluation."""

    correct: bool
    correct_reason: str
    target: int
    numbers: list[int]
    prompt: str
    completion: str
    reasoning: str
    solution: str


class Reasoner(curator.LLM):
    """Curator class for Countdown dataset."""

    return_completions_object = True

    def __init__(self, **kwargs):
        # 是否为原生推理模型，不同的模型的 Prompt 不同
        self.is_reasoning = kwargs.get("is_reasoning", False)
        del kwargs["is_reasoning"]
        super().__init__(**kwargs)

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        if self.is_reasoning:
            return [
                {
                    "role": "user",
                    "content": REASONING_USER_PROMPT_TPL.format(
                        numbers=", ".join(map(str, input["numbers"])),
                        target=input["target"],
                    ),
                },
            ]
        else:
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TPL.format(
                        numbers=", ".join(map(str, input["numbers"])),
                        target=input["target"],
                    ),
                },
            ]

    def correct(self, numbers, target, solution) -> Tuple[bool, str]:
        """Check if the LLM response is correct."""
        if not solution:
            # 如果 solution 为空，则认为不正确
            return False, "Solution is empty"
        try:
            if '=' in solution:
                # 删除掉等号以及等号后的内容
                solution = solution.split('=')[0]
            if not eval(solution) == target:
                # 如果 solution 计算结果不等于 target，则认为不正确
                return False, "Solution is not equal to target"
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, solution):
                # 如果 solution 不符合要求，则认为不正确
                return False, "Solution is not allowed"
            # 检查 solution 是否使用了所有数字
            used_numbers = [int(n) for n in re.findall(r'\d+', solution)]
            if sorted(used_numbers) != sorted(numbers):
                # 如果 solution 没有使用所有数字没有仅使用一次，则认为不正确
                return False, "Solution is not using all numbers or each number is not used only once"
            return True, "Solution is correct"
        except Exception as e:
            # 计算失败，则认为不正确
            return False, f"Solution calculation failed: {e}"

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        content = response["choices"][0]["message"]["content"]
        # 解析 <think> 和 <answer> 标签

        if self.is_reasoning:
            if "reasoning_content" in response["choices"][0]["message"]:
                # deepseek api
                reasoning = response["choices"][0]["message"]["reasoning_content"]
            else:
                # openrouter api
                reasoning = response["choices"][0]["message"]["reasoning"]
            solution_str = re.search(
                r"<answer>(.*?)</answer>", content, re.DOTALL
            )
            solution = solution_str.group(1) if solution_str else ""
            completion = f"<think>{reasoning}</think>{content}"
        else:
            reasoning_str = re.search(
                r"<think>(.*?)</think>", content, re.DOTALL
            )
            reasoning = reasoning_str.group(1) if reasoning_str else ""
            solution_str = re.search(
                r"<answer>(.*?)</answer>", content, re.DOTALL
            )
            solution = solution_str.group(1) if solution_str else ""
            completion = content

        correct, correct_reason = self.correct(input["numbers"], input["target"], solution)
        return [
            {
                "target": input["target"],
                "numbers": input["numbers"],
                "prompt": self.prompt(input)[-1]["content"],
                "completion": completion,
                "reasoning": reasoning or "",
                "solution": solution or "",
                "correct": correct,
                "correct_reason": correct_reason,
            }
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--is_reasoning", type=bool, default=False)
    args = parser.parse_args()

    print(f">>> All args: {args}")
    unique_model_name = (
        f"{args.provider}__{args.model_name.replace('/', '_').replace(':', '_')}"
    )

    llm = Reasoner(
        model_name=args.model_name,
        backend="openai",
        generation_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        backend_params={
            "api_key": os.getenv(f"{args.provider.upper()}_API_KEY"),
            "base_url": os.getenv(f"{args.provider.upper()}_BASE_URL"),
            "invalid_finish_reasons": [],  # 默认是 length 和 content_filter，但是这两个reason 重试大概率还是会失败，所以没必要重试。
        },
        is_reasoning=args.is_reasoning,
    )
    problems = load_dataset("json", data_files=args.data_path)["train"]
    print(f">>> Load {len(problems)} problems from {args.data_path}")
    print(problems[0])

    print(f">>> Start run eval {unique_model_name}")
    response = llm(problems)
    response.save_to_disk(f"data/test_{unique_model_name}_results")

    total_count = len(response)
    correct_count = len(response.filter(lambda x: x["correct"]))
    accuracy = correct_count / total_count
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.2%})")
