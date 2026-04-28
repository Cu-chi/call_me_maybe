from llm_sdk import Small_LLM_Model
import json


def load_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    path_vocab: str = llm.get_path_to_vocab_file()
    BPE_REPLACEMENTS: dict[str, str] = {
        "Ġ": " ",
        "Ċ": "\n",
        "ĉ": "\t",
    }

    with open(path_vocab, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)

    reversed_vocab: dict[int, str] = {}
    for token_str, token_id in vocab.items():
        clean_str = token_str
        for bpe_char, real_char in BPE_REPLACEMENTS.items():
            clean_str = clean_str.replace(bpe_char, real_char)
        reversed_vocab.update({token_id: clean_str})
    return reversed_vocab
