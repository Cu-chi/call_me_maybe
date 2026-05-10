import sys
import json
import math
import re
from typing import Any


def fn_add_numbers(a: float, b: float) -> float:
    return a + b


def fn_greet(name: str) -> str:
    return f"Hi {name}!"


def fn_reverse_string(s: str) -> str:
    return s[::-1]


def fn_get_square_root(a: float) -> float:
    return math.sqrt(a)


def fn_substitute_string_with_regex(source_string: str,
                                    regex: str,
                                    replacement: str) -> str:
    return re.sub(regex, replacement, source_string)


def main() -> None:
    with open("./data/output/function_calling_results.json", "r") as f:
        output: list[dict[str, Any]] = json.load(f)
    for function in output:
        print(f"{function["prompt"]}")
        if function["name"] == "fn_add_numbers":
            print(f"result: {fn_add_numbers(function['parameters']['a'],
                                            function['parameters']['b'])}")
        elif function["name"] == "fn_greet":
            print("result: " + fn_greet(function['parameters']['name']))
        elif function["name"] == "fn_reverse_string":
            print("result: " + fn_reverse_string(function['parameters']['s']))
        elif function["name"] == "fn_get_square_root":
            print(f"result: {fn_get_square_root(function['parameters']['a'])}")
        elif function["name"] == "fn_substitute_string_with_regex":
            print("result: " + fn_substitute_string_with_regex(
                    function['parameters']['source_string'],
                    function['parameters']['regex'],
                    function['parameters']['replacement']))
        else:
            print("Unknown function")
        print()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: output file not found", file=sys.stderr)
    except PermissionError:
        print("Error: permission error on output file", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
