from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

PROMPT = """
Here is a function in assembly:

```assembly
{assembly}
```

Here are its main features:
"""

def main():
    args = parse_arguments()
    print(args.model)

    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype="auto", device_map="auto").to("cuda")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

    # content = None
    # with open("simple.s", "r") as file:
    #     content = file.read()

    # query = PROMPT.format(assembly=content)

    # model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    # tokenized_output = model.generate(**model_inputs, max_new_tokens=200, temperature=0.5)
    # output = tokenizer.batch_decode(tokenized_output)[0]
    # print(output)

class Arguments:
    model: str # Name of the model to use. First check our model dictionary and default to using the name as a HuggingFace model
    prompt: str # File or string containing the prompt to use
    format: str # File or string containing the format to use
    assembly: str # File containing the assembly to analyze

def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Pipeline tool for assembly feature extraction"
    )
    parser.add_argument("-m", "--model", choices=["codeqwen"], type=str, default="codeqwen")
    parser.add_argument("-p", "--prompt", choices=["base"], type=str, default="base")
    parser.add_argument("-f", "--format", choices=["base"], type=str, default="base")
    parser.add_argument("assembly", metavar="ASM-FILE", type=str)

    return parser.parse_args(namespace=Arguments)

if __name__ == "__main__":
    main()
