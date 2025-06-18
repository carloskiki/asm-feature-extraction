"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
import json
import context
from data_processing import LibDataset
from tqdm import tqdm


@dataclass
class Bogus(context.Context):
    threshold: int

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "bogus",
            description="",
        )
        parser.add_argument("--threshold", type=int, default=200_000)

    def __call__(self):
        # Side quest: How many tokens does the prompt use?

        prompt = self.get_prompt("")
        tokenizer = self.get_tokenizer()

        chat = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(
            chat,
        )

        print("The prompt uses ", len(tokens['input_ids']), " tokens")