from src.definitions import create_models_from_json, get_functions_json
from llm_sdk import Small_LLM_Model
from src.utils import load_vocab
from src.generation import generate_function
import json
from typing import Any


def main() -> None:
    model = Small_LLM_Model()
    functions: list[dict[str, Any]] = get_functions_json(
        "./data/input/functions_definition.json")
    pydantic_models = create_models_from_json(functions)
    reversed_vocab: dict[int, str] = load_vocab(model)
    generated_text = generate_function("What is the sum of 3.15533333 and 2",
                                       model, reversed_vocab, functions,
                                       pydantic_models)
    try:
        data = json.loads(generated_text)
        print(json.dumps(data, indent=4))
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Format JSON valide")
    except ValueError:
        print("Format JSON invalide")


if __name__ == "__main__":
    main()
