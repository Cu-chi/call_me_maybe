from src.definitions import create_models_from_json, get_functions_json
from llm_sdk import Small_LLM_Model
from src.utils import load_vocab
from src.generation import generate_function, pre_prompt
import json
from typing import Any


def main() -> None:
    model = Small_LLM_Model()
    functions: list[dict[str, Any]] = get_functions_json(
        "./data/input/functions_definition.json")
    pydantic_models = create_models_from_json(functions)
    reversed_vocab: dict[int, str] = load_vocab(model)
    prompt: str = pre_prompt("What is the sum of 2 and 3?", functions)
    generated_text = generate_function(prompt, model, reversed_vocab,
                                       pydantic_models)
    try:
        print(generated_text)
        data = json.loads(generated_text)
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Format JSON valide")
    except ValueError:
        print("Format JSON invalide")

    # ids = model._tokenizer.encode("build a json data object:", add_special_tokens=False)
    # for _ in range(50):
    #     logits = model.get_logits_from_input_ids(ids)
    #     ids.append(logits.index(max(logits)))
    # print(model.decode(ids))


if __name__ == "__main__":
    main()
