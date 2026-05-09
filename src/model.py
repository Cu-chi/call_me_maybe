from llm_sdk import Small_LLM_Model


class Model(Small_LLM_Model):
    def __init__(self, model_name="Qwen/Qwen3-0.6B", *, device=None,
                 dtype=None, trust_remote_code=True):
        super().__init__(model_name, device=device, dtype=dtype,
                         trust_remote_code=trust_remote_code)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)
