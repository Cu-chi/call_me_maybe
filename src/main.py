from src.definitions import create_models_from_json, \
    get_functions_json, get_prompts_json
from llm_sdk import Small_LLM_Model
from src.utils import load_vocab
from src.generation import generate_function
import json
from typing import Any


def main() -> None:
    model = Small_LLM_Model()
    functions: list[dict[str, Any]] = get_functions_json(
        "./data/input/functions_definition.json")
    prompts: list[str] = get_prompts_json(
        "./data/input/function_calling_tests.json")
    pydantic_models = create_models_from_json(functions)
    reversed_vocab: dict[int, str] = load_vocab(model)

    output: list[dict[str, Any]] = []
    missed: int = 0
    for prompt in prompts:
        print(f"Generating JSON for: '{prompt}'...")
        result = generate_function(prompt, model, reversed_vocab,
                                   functions, pydantic_models)
        try:
            data = json.loads(result)
            output.append(data)
        except ValueError:
            missed += 1
            print(f"Invalid JSON format for prompt: '{prompt}', got: {result}")
    with open('data/output/function_calling_results.json', 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Done generating function calling results (missed: {missed})")


if __name__ == "__main__":
    main()
