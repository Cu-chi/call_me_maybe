from src.definitions import create_models_from_json, \
    get_minified_functions_json, get_functions_json,  \
    get_prompts_json, SchemaError, SchemaTypeError
from llm_sdk import Small_LLM_Model
from src.utils import load_vocab
from src.generation import generate_function
import json
import sys
import os
from typing import Any
from src.parsing import parse
from argparse import Namespace


def main() -> None:
    try:
        args: Namespace = parse()
        model = Small_LLM_Model()
        functions: str = get_functions_json(args.functions_definition)
        minified_functions: str = get_minified_functions_json(functions)
        prompts: list[str] = get_prompts_json(args.input)
        pydantic_models = create_models_from_json(functions)
        reversed_vocab: dict[int, str] = load_vocab(model)
    except SchemaError as e:
        print(f"Error: key '{e.key}' is not in the function definition",
              file=sys.stderr)
        return
    except SchemaTypeError as e:
        print(f"Error: unknown type '{e.type}' in function definition",
              file=sys.stderr)
        return
    except KeyError:
        print("Error: format error in function definition", file=sys.stderr)
        return
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    output: list[dict[str, Any]] = []
    generated: int = len(prompts)
    try:
        for prompt in prompts:
            print(f"Generating JSON for: '{prompt}'...")
            result = generate_function(prompt, model, reversed_vocab,
                                       minified_functions, pydantic_models)
            try:
                data = json.loads(result)
                output.append(data)
            except ValueError:
                generated -= 1
                print(f"Invalid JSON format, got: {result}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w+') as f:
            json.dump(output, f, indent=4)
        print(f"File {args.output} created "
              f"({generated}/{len(prompts)} functions)")
    except PermissionError:
        print(f"Error: not enought permissions for writing on '{args.output}'",
              file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
