from dataclasses import dataclass
import torch
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


@dataclass
class Context:
    model: str  # Name of the model to use.
    prompt: str  # Directory containing the prompt and format to use

    def get_prompt(self, assembly: str) -> str:
        """
        Returns the prompt without the assembly code included
        """

        with open(
            f"prompts/{self.prompt}/instructions.txt", "r", encoding="utf-8"
        ) as instructions_file:
            instructions = instructions_file.read()

        with open(
            f"prompts/{self.prompt}/format.jsonc", "r", encoding="utf-8"
        ) as format_file:
            json_format = format_file.read()

        return BASE_PROMPT.format(
            instructions=instructions, format=json_format, assembly=assembly
        )

    def get_model(self):
        return AutoModelForCausalLM.from_pretrained(
            MODELS[self.model], torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(MODELS[self.model], trust_remote_code=True)
