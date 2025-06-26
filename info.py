"""
Get info about the Model and prompt used
"""

from dataclasses import dataclass
import context

@dataclass
class Info(context.Context):
    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        subparsers.add_parser(
            "info",
            description="Information about the model and prompt used"
        )

    def __call__(self):
        prompt = self.get_prompt("")
        tokenizer = self.get_tokenizer()

        chat = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(
            chat,
        )
        token_count = len(tokens['input_ids'])

        print(f"MODEL: {self.model}")
        print(f"Configuration: {self.get_model().config}")
        print(f"PROMPT: {self.prompt} - uses {token_count} for an empty query with the model selected.")