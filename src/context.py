from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODELS = {
    "qwen-2.5-7": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-2.5-3": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen-2.5-1.5": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen-2.5-0.5": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen-3-4": "Qwen/Qwen3-4B",
    "gemma-3-4": "google/gemma-3-4b-it",
}
MAX_NEW_TOKENS = 512


@dataclass
class Context:
    """
    Which prompt and model to use.
    """

    model: str  # Name of the model to use.
    prompt: str  # Directory containing the prompt and format to use
    examples: int  # Number of examples to give the model before query (AKA the number of "shot"s e.g., 3-shot)

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
                print(
                    f"WARNING: requested {self.examples}-shot to be used, but the prompt chosen does not provide any examples."
                )
                print("         defaulting to 0-shot.")
            else:
                with open(examples_file, "r", encoding="utf-8") as file:
                    example_string = file.read()
                    if "\n" in example_string:
                        example_string = example_string.replace("\r", "")
                    else:
                        example_string = example_string.replace("\r", "\n")

                    examples = example_string.split(3 * "\n")
                    if len(examples) < self.examples:
                        print(
                            f"WARNING: requested {self.examples}-shot to be used, but the prompt chosen only provides {len(examples)} examples."
                        )
                        print(f"         defaulting to {len(examples)}-shot.")

                    for example in examples[: self.examples]:
                        example_assembly, output = example.split(2 * "\n")
                        prompt.append(
                            {
                                "role": "user",
                                "content": example_assembly,
                            }
                        )
                        prompt.append(
                            {
                                "role": "assistant",
                                "content": output,
                            }
                        )

        prompt.append({"role": "user", "content": f"```assembly\n{assembly}\n```"})
        return prompt

    def get_model(self, accelerator=None):
        """
        Return the model
        """
        device_map = {"": accelerator.process_index} if accelerator else "auto",
        if self.model == "qwen-emb":
            return SentenceTransformer(
                "Qwen/Qwen3-Embedding-4B",
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "device_map": device_map,
                },
                tokenizer_kwargs={"padding_side": "left"},
            )

        return AutoModelForCausalLM.from_pretrained(
            MODELS[self.model],
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )

    @cached_property
    def tokenizer(self):
        """
        Return the tokenizer
        """
        return AutoTokenizer.from_pretrained(MODELS[self.model], trust_remote_code=True)

    @cached_property
    def empty_prompt_size(self) -> int:
        """
        Size of the system prompt with examples
        """
        prompt = self.get_prompt("")
        tokenizer = self.tokenizer
        tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )

        return len(tokens)
