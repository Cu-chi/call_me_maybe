from llm_sdk import Small_LLM_Model
from src.constrainer import JSONConstrained
import numpy as np


def generate_function(input: str, llm: Small_LLM_Model,
                      reversed_vocab: dict[int, str]) -> str:
    input_ids: list[int] = llm.encode(input)[0].tolist()
    max_tokens: int = 50
    generated_text: str = ""
    constrainer: JSONConstrained = JSONConstrained()
    for _ in range(max_tokens):
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)

        allowed_chars: str = constrainer.get_allowed_chars()

        masked_logit = [float("-inf")] * len(logits)
        for token_id, token_str in reversed_vocab.items():
            if not token_str:
                continue
            if token_str[0] in allowed_chars:
                if token_id < len(logits):
                    masked_logit[token_id] = logits[token_id]

        next_token_id: int = int(np.argmax(masked_logit))
        input_ids.append(next_token_id)
        generated_text += reversed_vocab[next_token_id]
        constrainer.update_state(generated_text)
        print(generated_text)
        if generated_text.endswith("}"):
            break
    return generated_text
