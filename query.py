from transformers import AutoModelForCausalLM, AutoTokenizer
from arguments import Query

MODELS = {
    "codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"
}

BASE_PROMPT = """### Assembly Code ###
```assembly
{assembly}
```

### Instructions ###
{instructions}

### Output Format ###
The output must match exactly the following `JSON` schema:

```json
{format}
```"""

class Query:
    model: str # Name of the model to use. First check our model dictionary and default to using the name as a HuggingFace model
    prompt: str # File or string containing the prompt to use
    assembly: str # File containing the assembly to analyze

    def __init__(self, constructor):
        parser = constructor.add_parser(
            description="Query an LLM for the features of an assembly function"
        )
        parser.add_argument("-m", "--model", choices=["codeqwen"], type=str, default="codeqwen")
        parser.add_argument("-p", "--prompt", type=str, default="base")
        parser.add_argument("assembly", metavar="ASM-FILE", type=str)

        arguments = parser.parse_args(namespace=Query)
        self = arguments
    
    def format_prompt(self) -> str:
        with open(f"prompts/{self.prompt}/instructions.txt", "r") as instructions_file:
            instructions = instructions_file.read()
        
        with open(f"prompts/{self.prompt}/format.jsonc", "r") as format_file:
            format = format_file.read()

        with open(self.assembly, "r") as assembly_file:
            assembly = assembly_file.read()

        return BASE_PROMPT.format(instructions=instructions, format=format, assembly=assembly)
    
    def __call__(self, *args, **kwds):
        model = AutoModelForCausalLM.from_pretrained(MODELS[self.model], torch_dtype="auto", device_map="auto").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(MODELS[self.model])

        query = self.format_prompt()

        model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

        tokenized_output = model.generate(**model_inputs, max_new_tokens=2000, temperature=0.5)
        output = tokenizer.batch_decode(tokenized_output)[0]

        print(output)