SYSTEM_PROMPT = """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."""

USER_PROMPT_TPL = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once, and you must use all the numbers. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>. Think step by step inside <think> tags."""

REASONING_USER_PROMPT_TPL = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once, and you must use all the numbers. Return the final equation in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."""


def parse_user_prompt(tpl, numbers, target):
    return tpl.format(
        numbers=", ".join(map(str, numbers)),
        target=target,
    )
