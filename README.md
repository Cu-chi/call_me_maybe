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
### Design decisions
### Performance analysis
### Challenges faced
### Testing strategy
### Example usage

# Instructions

# Resources
[uv documentation](https://docs.astral.sh/uv/)
