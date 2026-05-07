from pydantic import BaseModel
from typing import Type
import json
import string

from src.utils import from_model_get_dict_fields

State = tuple[str, str, str, str, bool, frozenset[str]]
"""Tuple that describes the current state of the schema
0: State Key
1: buffer (used during key/value generation)
2: function name
3: param name
4: escaped (used to change behavior of char in string value)
5: parsed params (params that are already set in the dict)
"""

"""
All states, in order:
START

EXPECT_PROMPT_KEY
In_PROMPT_KEY
EXPECT_COLON_PROMPT
EXPECT_PROMPT_VAL
IN_PROMPT_VAL
EXPECT_COMMA_PROMPT

EXPECT_NAME_KEY
IN_NAME_KEY
EXPECT_COLON_NAME
EXPECT_NAME_VAL
IN_NAME_VAL
EXPECT_COMMA_NAME

EXPECT_PARAMS_KEY
IN_PARAMS_KEY
EXPECT_COLON_PARAMS

EXPECT_PARAM_START
EXPECT_PARAM_KEY_OR_END
IN_PARAM_KEY
EXPECT_COLON_PARAM

EXPECT_STRING_VAL or EXPECT_NUMBER_VAL or EXPECT_BOOL_VAL
IN_STRING or IN_NUMBER or IN_BOOL
EXPECT_PARAM_COMMA_OR_END
EXPECT_PARAM_KEY
etc...

EXPECT_END
"""


class SchemaConstrainer:
    def __init__(self, models: dict[str, Type[BaseModel]], input_str: str):
        self.schemas = models
        self.schemas_fields = {}
        for func_name in self.schemas:
            self.schemas_fields.update({
                func_name: from_model_get_dict_fields(self.schemas[func_name])
                })
        self.prompt = json.dumps(input_str)[1:-1]

    def initial_state(self) -> State:
        return ("START", "", "", "", False, frozenset())

    def update_state(self, current: State, token: str) -> State | None:
        state, buffer, func_name, param_name, escaped, parsed_params = current
        for char in token:
            res = self._consume(state, buffer, func_name, param_name,
                                escaped, parsed_params, char)
            if res is None:
                return None
            state, buffer, func_name, param_name, escaped, parsed_params = res
        return (state, buffer, func_name, param_name, escaped, parsed_params)

    def _consume(self, state: str, buffer: str, func_name: str,
                 param_name: str, escaped: bool,
                 parsed_params: frozenset[str], char: str):
        if state == "START":
            if char == "{":
                return ("EXPECT_PROMPT_KEY", buffer, func_name,
                        param_name, escaped, parsed_params)
        elif state == "EXPECT_PROMPT_KEY":
            if char == "\"":
                return ("IN_PROMPT_KEY", buffer, func_name,
                        param_name, escaped, parsed_params)
        elif state == "IN_PROMPT_KEY":
            if char == "\"":
                if buffer == "prompt":
                    return ("EXPECT_COLON_PROMPT", "", func_name, param_name,
                            escaped, parsed_params)
                return None
            if "prompt".startswith(buffer + char):
                return (state, buffer + char, func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_COLON_PROMPT":
            if char == ":":
                return ("EXPECT_PROMPT_VAL", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_PROMPT_VAL":
            if char == "\"":
                return ("IN_PROMPT_VAL", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "IN_PROMPT_VAL":
            if escaped:
                if not self.prompt.startswith(buffer + char):
                    return None
                return ("IN_PROMPT_VAL", buffer + char, func_name,
                        param_name, False, parsed_params)
            if char == "\\":
                if not self.prompt.startswith(buffer + char):
                    return None
                return ("IN_PROMPT_VAL", buffer + char, func_name,
                        param_name, True, parsed_params)
            if char == "\"":
                if buffer == self.prompt:
                    return ("EXPECT_COMMA_PROMPT", "", func_name, param_name,
                            escaped, parsed_params)
                return None
            if self.prompt.startswith(buffer + char):
                return (state, buffer + char, func_name, param_name,
                        False, parsed_params)
        elif state == "EXPECT_COMMA_PROMPT":
            if char == ",":
                return ("EXPECT_NAME_KEY", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_NAME_KEY":
            if char == "\"":
                return ("IN_NAME_KEY", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "IN_NAME_KEY":
            if char == "\"":
                if buffer == "name":
                    return ("EXPECT_COLON_NAME", "", func_name, param_name,
                            escaped, parsed_params)
                return None
            if "name".startswith(buffer + char):
                return (state, buffer + char, func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_COLON_NAME":
            if char == ":":
                return ("EXPECT_NAME_VAL", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_NAME_VAL":
            if char == "\"":
                return ("IN_NAME_VAL", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "IN_NAME_VAL":
            if char == "\"":
                if buffer in self.schemas:
                    return ("EXPECT_COMMA_NAME", "", buffer, param_name,
                            escaped, parsed_params)
                return None
            if any(fname.startswith(buffer + char) for fname in self.schemas):
                return (state, buffer + char, func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_COMMA_NAME":
            if char == ",":
                return ("EXPECT_PARAMS_KEY", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_PARAMS_KEY":
            if char == "\"":
                return ("IN_PARAMS_KEY", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "IN_PARAMS_KEY":
            if char == "\"":
                if buffer == "parameters":
                    return ("EXPECT_COLON_PARAMS", "", func_name, param_name,
                            escaped, parsed_params)
                return None
            if "parameters".startswith(buffer + char):
                return (state, buffer + char, func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_COLON_PARAMS":
            if char == ":":
                return ("EXPECT_PARAM_START", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_PARAM_START":
            if char == "{":
                return ("EXPECT_PARAM_KEY_OR_END", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_PARAM_KEY_OR_END":
            if char == "}" \
               and all(key in parsed_params or key == param_name
                       for key in self.schemas_fields[func_name].keys()):
                return ("EXPECT_END", "", func_name, param_name,
                        escaped, parsed_params)
            if char == "\"":
                return ("IN_PARAM_KEY", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "IN_PARAM_KEY":
            if char == "\"":
                if buffer in self.schemas_fields[func_name].keys() \
                   and buffer not in parsed_params:
                    return ("EXPECT_COLON_PARAM", "", func_name, buffer,
                            escaped, parsed_params)
            if any(pname.startswith(buffer + char)
                   for pname in self.schemas_fields[func_name].keys()):
                return ("IN_PARAM_KEY", buffer + char, func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_COLON_PARAM":
            if char == ":":
                ptype = self.schemas_fields[func_name][param_name]
                if ptype is str:
                    return ("EXPECT_STRING_VAL", "", func_name, param_name,
                            escaped, parsed_params)
                elif ptype is int or ptype is float:
                    return ("EXPECT_NUMBER_VAL", "", func_name, param_name,
                            escaped, parsed_params)
                elif ptype is bool:
                    return ("EXPECT_BOOL_VAL", "", func_name, param_name,
                            escaped, parsed_params)
        elif state == "EXPECT_STRING_VAL":
            if char == "\"":
                return ("IN_STRING_VAL", "", func_name, param_name,
                        False, parsed_params)
        elif state == "IN_STRING_VAL":
            if escaped:
                return ("IN_STRING_VAL", buffer, func_name,
                        param_name, False, parsed_params)
            if char == "\\":
                return ("IN_STRING_VAL", buffer, func_name,
                        param_name, True, parsed_params)
            if char == "\"":
                return ("EXPECT_PARAM_COMMA_OR_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
            return ("IN_STRING_VAL", buffer, func_name,
                    param_name, False, parsed_params)
        elif state == "EXPECT_NUMBER_VAL":
            if char in "-0123456789":
                return ("IN_NUMBER_VAL", char, func_name, param_name,
                        False, parsed_params)
        elif state == "IN_NUMBER_VAL":
            if char in "0123456789.":
                return ("IN_NUMBER_VAL", buffer + char, func_name,
                        param_name, False, parsed_params)
            if buffer == "-":
                return None
            if char in " \n\t\r":
                return ("EXPECT_PARAM_COMMA_OR_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
            if char == ",":
                return ("EXPECT_PARAM_KEY_OR_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
            if char == "}" \
                and all(key in parsed_params or key == param_name
                        for key in self.schemas_fields[func_name].keys()):
                return ("EXPECT_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
        elif state == "EXPECT_BOOL_VAL":
            if char in "tf":
                return ("IN_BOOL_VAL", char, func_name, param_name,
                        False, parsed_params)
        elif state == "IN_BOOL_VAL":
            if char in "truefals":
                if not "true".startswith(buffer + char) \
                   and not "false".startswith(buffer + char):
                    return None
                return ("IN_BOOL_VAL", buffer + char, func_name,
                        param_name, False, parsed_params)
            if char in string.whitespace:
                if buffer == "true" or buffer == "false":
                    return ("EXPECT_PARAM_COMMA_OR_END", "", func_name,
                            param_name, escaped,
                            parsed_params | frozenset([param_name]))
            if char == ",":
                return ("EXPECT_PARAM_KEY_OR_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
            if char == "}" \
                and all(key in parsed_params or key == param_name
                        for key in self.schemas_fields[func_name].keys()):
                return ("EXPECT_END", "", func_name, param_name,
                        escaped, parsed_params | frozenset([param_name]))
        elif state == "EXPECT_PARAM_COMMA_OR_END":
            if char == ",":
                return ("EXPECT_PARAM_KEY_OR_END", "", func_name,
                        param_name, False, parsed_params)
            if char == "}" \
                and all(key in parsed_params
                        for key in self.schemas_fields[func_name].keys()):
                return ("EXPECT_END", "", func_name, param_name,
                        escaped, parsed_params)
        elif state == "EXPECT_END":
            if char == "}":
                return ("DONE", "", func_name, param_name,
                        escaped, parsed_params)
        return None
