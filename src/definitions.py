import json
from pydantic import BaseModel, create_model
from typing import Type, Any


class SchemaTypeError(Exception):
    pass


def get_python_types(type_str: str) -> Type:
    if type_str == "number":
        return float
    elif type_str == "integer":
        return int
    elif type_str == "string":
        return str
    elif type_str == "boolean":
        return bool
    raise SchemaTypeError(f"type JSON '{type_str}' is not supported")


def create_models_from_json(path: str) -> dict[str, Type[BaseModel]]:
    with open(path, "r") as f:
        functions: list[dict[str, Any]] = json.load(f)

    models: dict[str, Type[BaseModel]] = {}
    for func in functions:
        func_name: str = func["name"]
        params: dict[str, Any] = func.get('parameters', {})

        fields: dict[str, Any] = {
            param_name: (get_python_types(details["type"]), ...)
            for param_name, details in params.items()
        }
        models[func_name] = create_model(f"{func_name}_Model", **fields)
    return models
