"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
from . import context
from .data_processing import LibDataset

@dataclass
class Bogus(context.Context):

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        _parser: ArgumentParser = subparsers.add_parser(
            "bogus",
            description="",
        )

    def __call__(self):
        dataset = LibDataset("lib-data", True, None, None, "putty", 0, "mips")
        fn = next(f for (f, _) in dataset if f.name == "MD5Init")
        print(fn)