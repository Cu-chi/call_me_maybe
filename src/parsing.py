import argparse


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="specify input and output files")
    parser.add_argument("--functions_definition",
                        default="./data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="./data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="./data/output/function_calling_results.json")
    parser.add_argument("--model",
                        default="Qwen/Qwen3-0.6B")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    return args
