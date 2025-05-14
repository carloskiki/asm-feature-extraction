import argparse
from query import Query
from retrieval import Retrieval
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}

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
    model: str  # Name of the model to use.
    prompt: str  # Directory containing the prompt and format to use
    command: Union[Query, Retrieval, None]

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
           "-m", "--model", choices=["codeqwen"], type=str, default="codeqwen"
        )
        parser.add_argument("-p", "--prompt", type=str, default="base")
        subparsers = parser.add_subparsers(
            description="the action to be performed", dest="subcommand", required=True
        )
        Query.command(subparsers)
        Retrieval.command(subparsers)

        arguments = parser.parse_args()
        sc = arguments.subcommand
        if sc == "query":
            command = Query()
        elif sc == "retrieval":
            command = Retrieval()
        else:
            command = None

        for name, value in arguments.__dict__.items():
            print(name, value)
            if command is not None and name != "subcommand":
                setattr(command, name, value)

        self.command = command

    def get_prompt(self, assembly: str) -> str:
        """
        Returns the prompt without the assembly code included
        """

        with open(
            f"prompts/{self.get_prompt}/instructions.txt", "r", encoding="utf-8"
        ) as instructions_file:
            instructions = instructions_file.read()

        with open(
            f"prompts/{self.get_prompt}/format.jsonc", "r", encoding="utf-8"
        ) as format_file:
            json_format = format_file.read()

        return BASE_PROMPT.format(
            instructions=instructions, format=json_format, assembly=assembly
        )

    def get_model(self):
        return AutoModelForCausalLM.from_pretrained(
            MODELS[self.model], torch_dtype="auto", device_map="auto"
        ).to("cuda")

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(MODELS[self.model])
