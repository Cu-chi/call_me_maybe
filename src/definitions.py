import json
import sys
from pydantic import BaseModel, create_model
from typing import Type, Any


class SchemaError(Exception):
    def __init__(self, key: str, *args: tuple[Any]) -> None:
        self.key = key
        super().__init__(*args)


class SchemaTypeError(Exception):
    def __init__(self, type: str, *args: tuple[Any]) -> None:
        self.type = type
        super().__init__(*args)


def get_python_types(type_str: str) -> Type[Any]:
    if type_str == "number":
        return float
    elif type_str == "integer":
        return int
    elif type_str == "string":
        return str
    elif type_str == "boolean":
        return bool
    raise SchemaTypeError(type_str)


def get_minified_functions_json(functions: list[dict[str, Any]]) -> str:
    try:
        minified: str = ""
        for function in functions:
            minified += f"{function["name"]}: {function["description"]}\n"
        return minified
    except KeyError:
        print("Error: A function definition is missing the key 'name' or "
              "'description'",
              file=sys.stderr)
        sys.exit(1)


def get_functions_json(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r") as f:
            functions: list[dict[str, Any]] = json.load(f)
            return functions
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: File '{path}' is not a valid JSON, syntax error at:",
              file=sys.stderr)
        print(f"line {e.lineno}, column {e.colno}: {e.msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while reading '{path}':\n{e}",
              file=sys.stderr)
        sys.exit(1)


def get_prompts_json(path: str) -> list[str]:
    prompts_list: list[str] = []
    try:
        with open(path, "r") as f:
            prompts: list[dict[str, Any]] = json.load(f)
            for prompt_dict in prompts:
                prompts_list.append(prompt_dict["prompt"])
            return prompts_list
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: File '{path}' is not a valid JSON, syntax error at:",
              file=sys.stderr)
        print(f"line {e.lineno}, column {e.colno}: {e.msg}", file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f"Error: File '{path}': key 'prompt' is not in the dict",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error while reading '{path}':\n{e}",
              file=sys.stderr)
        sys.exit(1)


def create_models_from_json(functions: list[dict[str, Any]])\
     -> dict[str, Type[BaseModel]]:
    models: dict[str, Type[BaseModel]] = {}
    for func in functions:
        func_name: str | None = func.get("name", None)
        if func_name is None:
            raise SchemaError("name")
        params: dict[str, Any] | None = func.get('parameters', None)
        if params is None:
            raise SchemaError("parameters")
        fields: dict[str, Any] = {
            param_name: (get_python_types(details["type"]), ...)
            for param_name, details in params.items()
        }
        models[func_name] = create_model(f"{func_name}_Model", **fields)
    return models
