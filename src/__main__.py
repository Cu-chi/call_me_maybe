import sys
from src.definitions import create_models_from_json
from llm_sdk import Small_LLM_Model
from src.utils import load_vocab
from src.generation import generate_function

def main() -> None:
    model = Small_LLM_Model()
    create_models_from_json("./data/input/functions_definition.json")
    reversed_vocab: dict[int, str] = load_vocab(model)
    generated_text = generate_function("Generate a simple json with key number and a number",
                                       model, reversed_vocab)
    print(generated_text)
    # 
    # ids = model._tokenizer.encode("build a json data object:", add_special_tokens=False)
    # for _ in range(50):
    #     logits = model.get_logits_from_input_ids(ids)
    #     ids.append(logits.index(max(logits)))
    # print(model.decode(ids))


if __name__ == "__main__":
    main()
