from enum import Enum, auto
from pydantic import BaseModel
import string


class JSONState(Enum):
    """Enum used to represent a state of
    the JSON constrained decoding
    """
    EXPECT_OBJECT_START = auto()
    EXPECT_OBJECT_END = auto()
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_COMMA = auto()

    EXPECT_STRING_VALUE = auto()
    EXPECT_NUMBER_VALUE = auto()
    EXPECT_BOOLEAN_VALUE = auto()

    IN_STRING = auto()
    IN_NUMBER = auto()
    IN_BOOLEAN = auto()


class JSONConstrained:
    def __init__(self, models: dict[str, BaseModel]) -> None:
        self.state: list[JSONState] = [JSONState.EXPECT_OBJECT_START]
        self.models = models
        self.current_func_name = ""

        self.is_escaped = False
        self.current_key = ""
        self.buffer = ""

    def get_allowed_chars(self) -> str:
        if not self.state:
            return ""
        cur_state: JSONState = self.state[-1]
        if cur_state == JSONState.EXPECT_OBJECT_START:
            return "{"
        elif cur_state == JSONState.EXPECT_OBJECT_END:
            return "}"
        elif cur_state == JSONState.EXPECT_KEY:
            return "\"}"
        elif cur_state == JSONState.EXPECT_COLON:
            return ":"
        elif cur_state == JSONState.EXPECT_COMMA:
            return ",}"
        elif cur_state == JSONState.EXPECT_STRING_VALUE:
            return "\""
        elif cur_state == JSONState.EXPECT_NUMBER_VALUE:
            return "-0123456789"
        elif cur_state == JSONState.EXPECT_BOOLEAN_VALUE:
            return "truefals"
        elif cur_state == JSONState.IN_STRING:
            return string.printable
        elif cur_state == JSONState.IN_NUMBER:
            return "0123456789.,}"
        elif cur_state == JSONState.IN_BOOLEAN:
            return "truefals,}"
        return ""

    def _handle_object_start(self, char: str) -> bool:
        if char == "{":
            self.state.pop()
            self.state.append(JSONState.EXPECT_OBJECT_END)
            self.state.append(JSONState.EXPECT_KEY)
            return True
        return False

    def _handle_object_end(self, char: str) -> bool:
        if char == "}":
            self.state.pop()
            # TODO:
            # mettre check pydantic pour prochain EXPECT valeur
            return True
        return False

    def _handle_expect_comma(self, char: str) -> bool:
        if char == ",":
            self.state.pop()
            self.state.append(JSONState.EXPECT_KEY)
            # TODO:
            # mettre check pydantic pour prochain EXPECT valeur
            return True
        elif char == "}":
            self.state.pop()
            self.update_state(char)
            return True
        return False

    def _handle_expect_key(self, char: str) -> bool:
        if char == "\"":
            self.state.pop()
            self.state.append(JSONState.EXPECT_COLON)
            self.state.append(JSONState.IN_STRING)
            self.buffer = ""
            self.is_escaped = False
            return True
        if char == "}":
            self.state.pop()
            self.buffer = ""
            self.is_escaped = False
            return True
        return False

    def _handle_expect_colon(self, char: str) -> bool:
        if char == ":":
            self.state.pop()
            self.state.append(JSONState.EXPECT_COMMA)
            self.state.append(JSONState.EXPECT_STRING_VALUE)
            return True
            # TODO:
            # mettre check pydantic pour prochain EXPECT valeur
        return False

    def _handle_in_string(self, char: str) -> bool:
        if self.is_escaped:
            self.buffer += char
            self.is_escaped = False
            return True

        if char == "\\":
            self.is_escaped = True
            return True

        if char == "\"":
            self.state.pop()
            if self.state[-1] == JSONState.EXPECT_COLON:
                self.current_key = self.buffer

            if self.state[-1] == JSONState.EXPECT_COMMA \
               and self.current_key == "name":
                self.current_func_name = self.buffer
            return True
        if char in string.printable and char not in string.whitespace:
            self.buffer += char
            return True
        return False

    def _handle_expect_string(self, char: str) -> bool:
        if char == "\"":
            self.state.pop()
            self.state.append(JSONState.IN_STRING)
            return True
        return False

    def _handle_expect_number(self, char: str) -> bool:
        self.state.pop()
        self.state.append(JSONState.IN_NUMBER)
        self.buffer = ""
        return self._handle_in_number(char)

    def _handle_in_number(self, char: str) -> bool:
        if char == "," or char == "}":
            self.state.pop()
            return self.update_state(char)
        if char in "-0123456789":
            self.buffer += char
            return True
        return False

    def _handle_expect_boolean(self, char: str) -> bool:
        self.state.pop()
        self.state.append(JSONState.IN_BOOLEAN)
        self.buffer = ""
        return self._handle_in_boolean(char)

    def _handle_in_boolean(self, char: str) -> bool:
        if char == "," or char == "}":
            self.state.pop()
            return self.update_state(char)
        if char in "truefals":
            if self.buffer + char == "true" or self.buffer + char == "false":
                self.state.pop()
                self.state.append(JSONState.EXPECT_COMMA)
                return True
            if "true".startswith(self.buffer + char) \
               or "false".startswith(self.buffer + char):
                self.buffer += char
                return True
        return False

    def update_state(self, generated_token: str) -> bool:
        result = True
        for char in generated_token:
            if not self.state:
                return False
            cur_state = self.state[-1]
            if char in string.whitespace and cur_state != JSONState.IN_STRING:
                continue
            match cur_state:
                case JSONState.EXPECT_OBJECT_START:
                    result = self._handle_object_start(char)
                case JSONState.EXPECT_KEY:
                    result = self._handle_expect_key(char)
                case JSONState.EXPECT_COLON:
                    result = self._handle_expect_colon(char)
                case JSONState.EXPECT_COMMA:
                    result = self._handle_expect_comma(char)
                case JSONState.EXPECT_OBJECT_END:
                    result = self._handle_object_end(char)
                case JSONState.EXPECT_STRING_VALUE:
                    result = self._handle_expect_string(char)
                case JSONState.IN_STRING:
                    result = self._handle_in_string(char)
                case JSONState.EXPECT_NUMBER_VALUE:
                    result = self._handle_expect_number(char)
                case JSONState.IN_NUMBER:
                    result = self._handle_in_number(char)
                case JSONState.EXPECT_BOOLEAN_VALUE:
                    result = self._handle_expect_boolean(char)
                case JSONState.IN_BOOLEAN:
                    result = self._handle_in_boolean(char)
                case _:
                    raise RuntimeError(f"Unknown JSON state '{cur_state}'")
            if not result:
                break
        return result

    def clone(self) -> 'JSONConstrained':
        instance = JSONConstrained(self.models)
        instance.state = self.state.copy()
        instance.is_escaped = self.is_escaped
        instance.buffer = self.buffer
        instance.current_func_name = self.current_func_name
        instance.current_key = self.current_key
        return instance
