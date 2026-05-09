from llm_sdk import Small_LLM_Model
from src.constrainer import SchemaConstrainer, State
import numpy as np
from pydantic import BaseModel
from typing import Type
import json
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax


def pre_prompt(input: str, functions: str) -> str:
    prompt: str = "You are an expert AI routing agent.\n"
    prompt += "Your task is to map the user's natural "
    prompt += "language input to the exact function that can answer it."
    prompt += f"\n\nAVAILABLE FUNCTIONS:\n{functions}\n\n"
    prompt += "RULES:\n"
    prompt += "1. Analyze the USER INPUT and select the most appropriate "
    prompt += "function from the list above.\n"
    prompt += "2. Extract the required parameters from the USER INPUT.\n"
    prompt += "3. Output strictly valid JSON. Do not write explanations, "
    prompt += "markdown formatting, or any text outside the JSON object.\n"
    prompt += "4. If a string is given, extract it from the quotes\n"
    prompt += f"""
EXPECTED JSON OUTPUT FORMAT:
{{
  "prompt": "The original natural-language request",
  "name": "The exact name of the selected function",
  "parameters": {{
    "param_1": "extracted_value"
  }}
}}

USER INPUT:
"{input}"\n
"""
    return prompt


def generate_function(input: str, llm: Small_LLM_Model,
                      reversed_vocab: dict[int, str],
                      functions: str,
                      models: dict[str, Type[BaseModel]]) -> str:
    prompt = pre_prompt(input, functions)
    generated_text: str = f"{{\"prompt\":\"{json.dumps(
        input)[1:-1]}\",\"name\":\""
    input_ids: list[int] = llm.encode(prompt + generated_text)[0].tolist()
    max_tokens: int = 200
    constrainer = SchemaConstrainer(models, input)
    cur_state: State | None = constrainer.post_name_state()
    syntax = Syntax(generated_text, "json", theme="monokai", word_wrap=True)
    panel: Panel = Panel(syntax,
                         title="Generating...",
                         border_style="blue")
    with Live(panel, refresh_per_second=20) as live:
        for _ in range(max_tokens):
            if cur_state is None:
                return generated_text
            logits: list[float] = llm.get_logits_from_input_ids(input_ids)

            masked_logit = [float("-inf")] * len(logits)

            allowed_first: str | None = constrainer.get_allowed_first_chars(
                    cur_state[0])

            for token_id, token_str in reversed_vocab.items():
                if cur_state[0] == "IN_NUMBER_VAL":
                    if token_id > 92:
                        break
                if allowed_first is not None\
                   and token_str[0] not in allowed_first:
                    continue
                if constrainer.update_state(cur_state, token_str) is not None:
                    masked_logit[token_id] = logits[token_id]

            next_token_id: int = int(np.argmax(masked_logit))
            if masked_logit[next_token_id] == float("-inf"):
                break
            input_ids.append(next_token_id)
            next_token_str = reversed_vocab[next_token_id]
            generated_text += next_token_str
            cur_state = constrainer.update_state(cur_state, next_token_str)

            syntax.code = generated_text
            live.update(panel)
            if cur_state is None or cur_state[0] == "DONE":
                panel.title = "[bold green]Generation done[/bold green]"
                panel.border_style = "green"
                live.update(panel)
                break
    print()
    return generated_text
