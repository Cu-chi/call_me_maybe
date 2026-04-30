from llm_sdk import Small_LLM_Model
from src.constrainer import JSONConstrained, JSONState
import numpy as np
from pydantic import BaseModel
from typing import Any


def pre_prompt(input: str, functions: list[dict[str, Any]]):
    prompt: str = f"""
You are an expert in categorization.
Given the following list of function,
generate a valid JSON that follow the function format and parameters
which answer to this specific question '{input}':
"""
    for function in functions:
        prompt += f"{function['name']}: {function['description']} "
        prompt += "parameters are"
        for param in function["parameters"]:
            prompt += f" '{param}' ({function['parameters'][param]['type']})"
        prompt += "\n"
    return prompt


def generate_function(prompt: str, llm: Small_LLM_Model,
                      reversed_vocab: dict[int, str],
                      models: dict[str, BaseModel]) -> str:
    input_ids: list[int] = llm.encode(prompt)[0].tolist()
    max_tokens: int = 20
    generated_text: str = ""
    constrainer: JSONConstrained = JSONConstrained(models)
    for _ in range(max_tokens):
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)

        allowed_chars: str = constrainer.get_allowed_chars()

        masked_logit = [float("-inf")] * len(logits)
        for token_id, token_str in reversed_vocab.items():
            if not token_str:
                continue
            if token_str[0] in allowed_chars and token_id < len(logits):
                constrainer_tester: JSONConstrained = constrainer.clone()
                if constrainer_tester.update_state(token_str):
                    masked_logit[token_id] = logits[token_id]

        next_token_id: int = int(np.argmax(masked_logit))
        input_ids.append(next_token_id)
        next_token_str = reversed_vocab[next_token_id]
        generated_text += next_token_str
        constrainer.update_state(next_token_str)
        print(generated_text)
        if generated_text.endswith("}") or not constrainer.state:
            break
    return generated_text
