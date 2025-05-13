import argparse
from query import Query
from retrieval import Retrieval
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

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

class Context:
    model: str # Name of the model to use.
    prompt: str # Directory containing the prompt and format to use
    command: Union[Query, Retrieval, None]

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", choices=["codeqwen"], type=str, default="codeqwen")
        parser.add_argument("-p", "--prompt", type=str, default="base")
        subparsers = parser.add_subparsers(description="the action to be performed", dest="subcommand", required=True)
        Query.command(subparsers)
        Retrieval.command(subparsers)

        arguments = parser.parse_args()
        match arguments.subcommand:
            case "query":
                command = Query()
            case "retrieval":
                command = Retrieval()
            case _:
                command = None
        
        for name, value in arguments.__dict__.items():
            if command != None and name != "subcommand":
                setattr(command, name, value)
        
        self.command = command
    
    """
    Returns the prompt without the assembly code included
    """
    def prompt(self, assembly: str) -> str:
        with open(f"prompts/{self.prompt}/instructions.txt", "r") as instructions_file:
            instructions = instructions_file.read()
        
        with open(f"prompts/{self.prompt}/format.jsonc", "r") as format_file:
            format = format_file.read()

        return BASE_PROMPT.format(instructions=instructions, format=format, assembly=assembly)
    
    def model(self):
        return AutoModelForCausalLM.from_pretrained(MODELS[self.model], torch_dtype="auto", device_map="auto").to("cuda")
    
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(MODELS[self.model])