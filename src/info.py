"""
Get info about the Model and prompt used
"""

from dataclasses import dataclass
from . import context

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
        token_count = self.empty_prompt_size

        print(f"MODEL: {self.model}")
        print(f"Configuration: {self.get_model().config}")
        print(f"PROMPT: {self.prompt} - uses {token_count} for an empty query with the model selected.")