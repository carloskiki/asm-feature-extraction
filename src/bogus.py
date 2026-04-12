"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
from . import context
from .data_processing import LibDataset, Function

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
        dataset = LibDataset("lib-data", True, None, None, "openssl", None, "gcc")
        
        for (fn, _) in dataset:
            if fn.name.lower() == "sha384_init":
                print("*******")
                print(fn)
