from transformers import AutoModelForCausalLM, AutoTokenizer

# PROMPT = """
# Determine the genera class (examples: Cryptographic, C Runtime, Networking) of the function that this assembly routine belongs to.
# 
# ```assembly
# {assembly}
# ```
# 
# Function Class:
# """

PROMPT = """
Here is a function in assembly. We explain its main features.

```
"""

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype="auto", device_map="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

    content = None
    with open("simple.s", "r") as file:
        content = file.read()

    query = PROMPT + content + "\n```"

    model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    tokenized_output = model.generate(**model_inputs, max_length=500)
    output = tokenizer.batch_decode(tokenized_output)[0]
    print(output)

if __name__ == "__main__":
    main()