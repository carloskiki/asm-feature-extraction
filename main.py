from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """
Here is a function in assembly:

```assembly
{assembly}
```

Here are its main features:
"""

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype="auto", device_map="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

    content = None
    with open("simple.s", "r") as file:
        content = file.read()

    query = PROMPT.format(assembly=content)

    model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    tokenized_output = model.generate(**model_inputs, max_new_tokens=200)
    output = tokenizer.batch_decode(tokenized_output)[0]
    print(output)

if __name__ == "__main__":
    main()