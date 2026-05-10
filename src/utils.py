from llm_sdk import Small_LLM_Model
from pydantic import BaseModel
from typing import Type, Any
import json


def universal_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    reversed_vocab: dict[int, str] = {}
    vocab_size = len(llm._tokenizer)

    for token_id in range(vocab_size):
        real_string = llm._tokenizer.decode([token_id],
                                            skip_special_tokens=True)
        reversed_vocab[token_id] = real_string
    return reversed_vocab


def load_vocab(llm: Small_LLM_Model, name: str) -> dict[int, str]:
    if name == "Qwen/Qwen3-0.6B":
        path_vocab: str = llm.get_path_to_vocab_file()
        BPE_REPLACEMENTS: dict[str, str] = {
            "Ġ": " ",
            "Ċ": "\n",
            "ĉ": "\t",
        }

        with open(path_vocab, "r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)
    else:
        return universal_vocab(llm)

    reversed_vocab: dict[int, str] = {}
    for token_str, token_id in vocab.items():
        clean_str = token_str
        for bpe_char, real_char in BPE_REPLACEMENTS.items():
            clean_str = clean_str.replace(bpe_char, real_char)
        reversed_vocab.update({token_id: clean_str})
    return reversed_vocab


def from_model_get_dict_fields(model: Type[BaseModel]) -> dict[str, Type[Any]]:
    fields: dict[str, Type[Any]] = {}

    for key in model.__pydantic_fields__:
        field_type: Type[Any] | None = \
            model.__pydantic_fields__[key].annotation
        if field_type is not None:
            fields.update({key: field_type})
    return fields
