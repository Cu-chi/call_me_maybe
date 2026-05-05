from llm_sdk import Small_LLM_Model
from src.constrainer import SchemaConstrainer, State
import numpy as np
from pydantic import BaseModel
from typing import Any, Type
import json


def pre_prompt(input: str, functions: list[dict[str, Any]]):
    tools_schema = json.dumps(functions, indent=2)
    prompt: str = "You are an expert AI routing agent."
    prompt += "Your task is to map the user's natural "
    prompt += "language input to the exact function that can answer it."
    prompt += f"\n\nAVAILABLE FUNCTIONS:\n{tools_schema}\n\n"
    prompt += "RULES:\n"
    prompt += "1. Analyze the USER INPUT and select the most appropriate "
    prompt += "function from the list above.\n"
    prompt += "2. Extract the required parameters from the USER INPUT.\n"
    prompt += "3. If no function matches the request, output \"unknown\" "
    prompt += "for the function name.\n"
    prompt += "4. Output strictly valid JSON. Do not write explanations, "
    prompt += "markdown formatting, or any text outside the JSON object.\n"
    prompt += f"""
EXPECTED JSON OUTPUT FORMAT:
{{
  "prompt": "The original natural-language request",
  "name": "The exact name of the selected function (or 'unknown')",
  "parameters": {{
    "param_1": "extracted_value"
  }}
}}

USER INPUT:
"{input}"
"""
    return prompt


def generate_function(input: str, llm: Small_LLM_Model,
                      reversed_vocab: dict[int, str],
                      functions: list[dict[str, Any]],
                      models: dict[str, Type[BaseModel]]) -> str:
    prompt = pre_prompt(input, functions)
    input_ids: list[int] = llm.encode(prompt)[0].tolist()
    max_tokens: int = 200
    generated_text: str = ""
    constrainer = SchemaConstrainer(models, input)
    cur_state: State | None = constrainer.initial_state()
    # each state has an associated token ids list
    cache: dict[State, list[int]] = {}
    for _ in range(max_tokens):
        if cur_state is None:
            return generated_text
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)

        masked_logit = [float("-inf")] * len(logits)
        if cur_state not in cache:
            valid_tokens = []
            for token_id, token_str in reversed_vocab.items():
                if not token_str or token_id >= len(logits):
                    continue
                if constrainer.update_state(cur_state, token_str) is not None:
                    valid_tokens.append(token_id)
            cache[cur_state] = valid_tokens

        for tid in cache[cur_state]:
            masked_logit[tid] = logits[tid]

        next_token_id: int = int(np.argmax(masked_logit))
        if masked_logit[next_token_id] == float("-inf"):
            break
        input_ids.append(next_token_id)
        next_token_str = reversed_vocab[next_token_id]
        generated_text += next_token_str
        cur_state = constrainer.update_state(cur_state, next_token_str)
        print(generated_text)
        if cur_state is None or cur_state[0] == "DONE":
            break
    return generated_text
