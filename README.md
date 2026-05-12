*This project has been created as part of the 42 curriculum by equentin*

# Description
This project is an introduction to function calling in LLMs.
The goal is to create a function calling tool that translates natural language prompts into structured function calls.  
Given a question like "What is the sum of 40 and 2?", the solution should not return 42, but instead provide:
- The function name: fn_add_numbers
- The arguments: {"a": 40, "b": 2}

The implementation must use constrained decoding to guarantee 100% valid JSON
output, ensuring near-perfect reliability even with a small 0.5B parameter model.

### Algorithm explanation
The approach of the algorithm is simple, at each token, we know in which state the generation is. So it starts with the state "START" and the valid tokens will be those who respect the states.
Here is the list in order of valid tokens:
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
For example valid tokens at the "START" state will be: '{', '{"'

### Design decisions
The choice of restricting by using state is in my opinion better for understanding where the LLM is and easier to debug.

### Performance analysis
Performance analysis on CPU **Intel(R) Core(TM) Ultra 7 265**, got:
| Model | Size | Speed (Tokens/sec) |
| :--- | :--- | :--- |
| `Qwen/Qwen3-0.6B` | ~600M | **~2.2 t/s** |
| `HuggingFaceTB/SmolLM2-360M-Instruct`| 360M | **~ 3.2t/s** |
| `HuggingFaceTB/nanowhale-100m` | 100M | **~ 7.5t/s** |
| `Harley-ml/Tenete-8M` | 8M | **~98.0 t/s** |

### Challenges faced
Valid string values: I restricted values and I made a variable to check if the character is escaped so the constrainer doesn't detect an escaped '"' as the end of the string

Parameters: it was hard to restrict the model to only valid parameters. The solution was to pass at each state the parsed parameters

Number values: restriction of two '.' in float values, of '.' in integers, check if there is no useless zero in front of the value

Performance: at the beginning, I was generating the full JSON from the start to the end but for performance optimization, I passed the prompt and started the constrainer at the 'name' key

### Testing strategy
I checked every state for basic JSON, validating JSON and printing results when it was wrong.

### Example usage
You create a list of valid function definitions in the `data/input`, you enter your questions in another file in the same directory. You run the project to answer with the good function call with good parameters.

### Bonus
**Interactive mode**:
```sh
make run ARGS="--interactive"
```

**Multiple model support**:
```sh
make run ARGS="--model HuggingFaceTB/SmolLM2-360M-Instruct"
```

**Public implementation of `encode` and `decode` methods**: in `src/model.py`

Performance optimization: **cache**

**Visualization of the generation process**

**Output tester**:
```sh
make test
```

# Instructions

install uv:  
```sh
curl -LsSf https://astral.sh/uv/install.sh | less
```

sync the project dependencies
```sh
make install
```

run the project
```sh
make run
```
*optional arguments:*
pass with
```sh
make run ARGS="..."
```  
`--functions_definition [path]`: functions definition file  
`--input [path]`: function calling tests  
`--output [path]`: function calling results   

**useful**  
check flake8 and mypy:
```sh
make lint
```
or
```sh
make lint-strict
```
clean cache dirs:
```sh
make clean
```
test the output file:
```sh
make test
```
run with pdb to debug:
```sh
make debug
```


# Resources
[uv documentation](https://docs.astral.sh/uv/)  
[logit to tokens](https://medium.com/@adimodi96/from-logits-to-tokens-9a36feab9cab)  
[constrained decoding](https://www.aidancooper.co.uk/constrained-decoding/)  
[constrained decoding](https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)  
[constrained decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)  
[enum doc](https://docs.python.org/fr/3/library/enum.html)  
[tensor doc](https://medium.com/data-science/what-is-a-tensor-in-deep-learning-6dedd95d6507)  
[python argparse doc](https://docs.python.org/3/library/argparse.html)  
