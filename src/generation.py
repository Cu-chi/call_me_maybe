from llm_sdk import Small_LLM_Model
from src.constrainer import JSONConstrained, JSONState
import numpy as np
from pydantic import BaseModel
from typing import Any
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


def generate_function(prompt: str, llm: Small_LLM_Model,
                      reversed_vocab: dict[int, str],
                      models: dict[str, BaseModel]) -> str:
    input_ids: list[int] = llm.encode(prompt)[0].tolist()
    max_tokens: int = 50
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
