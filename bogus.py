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

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "bogus",
            description="",
        )

    def __call__(self):
        for _ in range(5):
            start = time.time()
            self.get_model()
            self.get_tokenizer()
            stop = time.time()
            print(stop - start)