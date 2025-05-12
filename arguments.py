import argparse

BASE_PROMPT = """### Assembly Code ###
```assembly
{assembly}
```

### Instructions ###
{instructions}

### Output Format ###
The output must match exactly the following `JSON` schema:

{format}

### Output ###
"""

class Arguments:
    model: str # Name of the model to use. First check our model dictionary and default to using the name as a HuggingFace model
    prompt: str # File or string containing the prompt to use
    assembly: str # File containing the assembly to analyze

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Pipeline tool for assembly feature extraction"
        )
        parser.add_argument("-m", "--model", choices=["codeqwen"], type=str, default="codeqwen")
        parser.add_argument("-p", "--prompt", type=str, default="base")
        parser.add_argument("assembly", metavar="ASM-FILE", type=str)

        arguments = parser.parse_args(namespace=Arguments)
        self = arguments
    
    def format_prompt(self) -> str:
        with open(f"prompts/{self.prompt}/instructions.txt", "r") as instructions_file:
            instructions = instructions_file.read()
        
        with open(f"prompts/{self.prompt}/format.jsonc", "r") as format_file:
            format = format_file.read()

        with open(self.assembly, "r") as assembly_file:
            assembly = assembly_file.read()

        return BASE_PROMPT.format(instructions=instructions, format=format, assembly=assembly)