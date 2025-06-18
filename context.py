from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}
MAX_NEW_TOKENS = 1024

@dataclass
class Context:
    """
    Which prompt and model to use.
    """
    model: str  # Name of the model to use.
    prompt: str  # Directory containing the prompt and format to use
    examples: int # Number of examples to give the model before query (AKA the number of "shot"s e.g., 3-shot)

    def get_prompt(self, assembly: str) -> list[dict[str, str]]:
        """
        Returns the prompt to be formatted in the chat template
        """

        with open(
            f"prompts/{self.prompt}/instructions.txt", "r", encoding="utf-8"
        ) as instructions_file:
            instructions = instructions_file.read()

        prompt = [
            {
                "role": "system",
                "content": instructions,
            },
        ]

        examples_file = Path(f"prompts/{self.prompt}/examples.txt")
        if self.examples > 0:
            if not examples_file.is_file():
                print(f"WARNING: requested {self.examples}-shot to be used, but the prompt chosen does not provide any examples.")
                print("         defaulting to 0-shot.")
            else:
                with open(examples_file, "r" , encoding="utf-8") as file:
                    example_string = file.read()
                    if "\n" in example_string:
                        example_string = example_string.replace("\r", "")
                    else:
                        example_string = example_string.replace("\r", "\n")

                    examples = example_string.split(3*"\n")
                    if len(examples) < self.examples:
                        print(f"WARNING: requested {self.examples}-shot to be used, but the prompt chosen only provides {len(examples)} examples.")
                        print(f"         defaulting to {len(examples)}-shot.")

                    
                    for example in examples[:self.examples]:
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
        

    def get_model(self, accelerator = None):
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