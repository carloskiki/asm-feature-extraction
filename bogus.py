"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
import context
from data_processing import LibDataset

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
        dataset = LibDataset("lib-data", True, 1000, None, "openssl", 2, "gcc")

        with open(self.out_file, "w", encoding="utf-8") as file:
            for (func, _) in dataset:
                file.write("#####\n")
                file.write(func.name + "\n")
                file.write(func)
                file.write("\n\n")