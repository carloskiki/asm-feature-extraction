"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
import context
from data_processing import LibDataset
import time

@dataclass
class Bogus(context.Context):
    out_file: str

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "bogus",
            description="",
        )
        parser.add_argument("out_file", type=str)

    def __call__(self):
        for _ in range(5):
            start = time.time()
            self.get_model()
            self.get_tokenizer()
            stop = time.time()
            print(stop - start)