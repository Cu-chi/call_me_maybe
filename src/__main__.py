from src.definitions import create_models_from_json, \
    get_minified_functions_json, get_functions_json,  \
    get_prompts_json, SchemaError, SchemaTypeError
from src.utils import load_vocab
from src.generation import generate_function
from src.constrainer import State
from src.model import Model
import json
import sys
import os
from typing import Any, Type
from pydantic import BaseModel
from rich.prompt import Prompt
from src.parsing import parse
from argparse import Namespace


def save_outputs(args: Namespace, output: list[dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w+') as f:
            json.dump(output, f, indent=4)
    except PermissionError:
        print(f"Error: not enought permissions for writing on '{args.output}'",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def generate_and_save(prompt: str,
                      model: Model,
                      reversed_vocab: dict[int, str],
                      minified_functions: str,
                      pydantic_models: dict[str, Type[BaseModel]],
                      cache: dict[State, list[int]],
                      output: list[dict[str, Any]]
                      ) -> bool:
    print(f"Generating JSON for: '{prompt}'...")
    res, cache = generate_function(prompt, model, reversed_vocab,
                                   minified_functions,
                                   pydantic_models, cache)
    try:
        data = json.loads(res)
        output.append(data)
        return True
    except ValueError:
        print(f"Invalid JSON format, got: {res}")
        return False


def main() -> None:
    try:
        args: Namespace = parse()
        model = Model(model_name=args.model)
        functions: list[dict[str, Any]] = get_functions_json(
            args.functions_definition)
        minified_functions: str = get_minified_functions_json(functions)
        if not args.interactive:
            prompts: list[str] = get_prompts_json(args.input)
        pydantic_models = create_models_from_json(functions)
        reversed_vocab: dict[int, str] = load_vocab(model, args.model)
    except SchemaError as e:
        print(f"Error: key '{e.key}' is not in the function definition",
              file=sys.stderr)
        sys.exit(1)
    except SchemaTypeError as e:
        print(f"Error: unknown type '{e.type}' in function definition",
              file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print("Error: format error in function definition", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    output: list[dict[str, Any]] = []
    try:
        cache: dict[State, list[int]] = {}
        if args.interactive:
            while True:
                user_input = Prompt.ask("[bold blue]Prompt[/]")
                generate_and_save(user_input, model, reversed_vocab,
                                  minified_functions,
                                  pydantic_models, cache, output)
        else:
            generated: int = len(prompts)
            for prompt in prompts:
                if not generate_and_save(prompt, model, reversed_vocab,
                                         minified_functions,
                                         pydantic_models, cache, output):
                    generated -= 1
            save_outputs(args, output)
            print(f"File {args.output} created "
                  f"({generated}/{len(prompts)} functions)")
    except (KeyboardInterrupt, EOFError):
        print("\nProgram exited, saving functions to file...")
        save_outputs(args, output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
