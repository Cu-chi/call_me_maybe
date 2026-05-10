from llm_sdk import Small_LLM_Model
from typing import Any


class Model(Small_LLM_Model):
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B",
                 *any: Any,
                 device: Any = None,
                 dtype: Any = None,
                 trust_remote_code: bool = True) -> None:
        super().__init__(model_name, device=device, dtype=dtype,
                         trust_remote_code=trust_remote_code)

    def encode(self, text: str) -> list[int] | Any:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str | list[str] | Any:
        return self._tokenizer.decode(ids, skip_special_tokens=True)
