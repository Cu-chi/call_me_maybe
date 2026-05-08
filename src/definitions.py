import json
from pydantic import BaseModel, create_model
from typing import Type, Any


class SchemaError(Exception):
    def __init__(self, key, *args):
        self.key = key
        super().__init__(*args)


class SchemaTypeError(Exception):
    def __init__(self, type, *args):
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


def get_functions_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        functions: list[dict[str, Any]] = json.load(f)
    return functions


def get_prompts_json(path: str) -> list[str]:
    prompts_list: list[str] = []
    with open(path, "r") as f:
        prompts: list[dict[str, Any]] = json.load(f)
    for prompt_dict in prompts:
        prompts_list.append(prompt_dict["prompt"])
    return prompts_list


def create_models_from_json(functions: list[dict[str, Any]])\
     -> dict[str, Type[BaseModel]]:
    models: dict[str, Type[BaseModel]] = {}
    for func in functions:
        func_name: str = func.get("name", None)
        if func_name is None:
            return SchemaError("name")
        params: dict[str, Any] | None = func.get('parameters', None)
        if params is None:
            raise SchemaError("parameters")
        fields: dict[str, Any] = {
            param_name: (get_python_types(details["type"]), ...)
            for param_name, details in params.items()
        }
        models[func_name] = create_model(f"{func_name}_Model", **fields)
    return models
