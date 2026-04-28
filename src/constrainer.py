from enum import Enum, auto


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


class JSONConstrained:
    def __init__(self) -> None:
        self.state: list[JSONState] = [JSONState.EXPECT_OBJECT_START]

    def get_allowed_chars(self) -> str:
        cur_state: JSONState = self.state[-1]
        if cur_state == JSONState.EXPECT_OBJECT_START:
            return "{"
        elif cur_state == JSONState.EXPECT_OBJECT_END:
            return "}"
        elif cur_state == JSONState.EXPECT_KEY:
            return "\""
        elif cur_state == JSONState.EXPECT_COLON:
            return ":"
        elif cur_state == JSONState.EXPECT_COMMA:
            return ","
        elif cur_state == JSONState.EXPECT_STRING_VALUE:
            return "\""
        elif cur_state == JSONState.EXPECT_NUMBER_VALUE:
            return ""
        elif cur_state == JSONState.EXPECT_BOOLEAN_VALUE:
            return "truefalse"
        elif cur_state == JSONState.IN_STRING:
            return "\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif cur_state == JSONState.IN_NUMBER:
            return "0123456789."
        return ""

    def update_state(self, generated_text: str):
        for char in generated_text:
            cur_state = self.state[-1]
            if cur_state == JSONState.EXPECT_OBJECT_START and char == "{":
                self.state.pop()
                self.state.append(JSONState.EXPECT_KEY)
            elif cur_state == JSONState.EXPECT_KEY and char == "\"":
                self.state.pop()
                self.state.append(JSONState.EXPECT_COLON)
                self.state.append(JSONState.IN_STRING)
