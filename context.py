import os
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}

SYSTEM_PROMPT = """{instructions}

### Template ###
```json
{format}
```
"""

@dataclass
class Context:
    """
    Which prompt and model to use.
    """
    model: str  # Name of the model to use.
    prompt: str  # Directory containing the prompt and format to use

    def get_prompt(self, assembly: str) -> list[dict[str, str]]:
        """
        Returns the prompt to be formatted in the chat template
        """

        with open(
            f"prompts/{self.prompt}/instructions.txt", "r", encoding="utf-8"
        ) as instructions_file:
            instructions = instructions_file.read()

        with open(
            f"prompts/{self.prompt}/format.jsonc", "r", encoding="utf-8"
        ) as format_file:
            json_format = format_file.read()

        prompt = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    instructions=instructions, format=json_format
                ),
            },
        ]

        examples = Path(f"prompts/{self.prompt}/examples.txt")
        if examples.is_file():
            with open(examples, "r" , encoding="utf-8") as file:
                examples_string = file.read()
                if "\n" in examples_string:
                    examples_string = examples_string.replace("\r", "")
                else:
                    examples_string = examples_string.replace("\r", "\n")

                for example in examples_string.split(3*"\n"):
                    example_assembly, output = example.split(2*"\n")
                    prompt.append({
                        "role": "user",
                        "content": f"```assembly\n{example_assembly}\n```"
                    })
                    prompt.append({
                        "role": "assistant",
                        "content": output,
                    })

        prompt.append({
            "role": "user",
            "content": f"```assembly\n{assembly}\n```"
        })
        return prompt
        

    def get_model(self, accelerator):
        """
        Return the model
        """
        if accelerator is not None:
            return AutoModelForCausalLM.from_pretrained(
                MODELS[self.model],
                torch_dtype="auto",
                device_map={"": accelerator.process_index},
                trust_remote_code=True,
            )

        return AutoModelForCausalLM.from_pretrained(
            MODELS[self.model],
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def get_tokenizer(self):
        """
        Return the tokenizer
        """
        return AutoTokenizer.from_pretrained(MODELS[self.model], trust_remote_code=True)