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
        parser.add_argument("--threshold", type=int, default=100_000)

    def __call__(self):
        # Side quest: extract and check for size outliers

        dataset = LibDataset("lib-data", main_process=True, pool_size=None, seed=None)
        outliers = []

        for function in tqdm(dataset.functions, desc="Checking for size outliers"):
            tokenizer = self.get_tokenizer()

            tokens = tokenizer(
                str(function),
                return_tensors="pt",
            )

            if len(tokens["input_ids"]) > self.threshold:
                outliers.append((function, len(tokens["input_ids"])))

        print(len(outliers))

        with open("outliers.txt", "w", encoding="utf-8") as file:
            json.dump([{f.name: token_len} for f, token_len in outliers], file)