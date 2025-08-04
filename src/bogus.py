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
        dataset = LibDataset("lib-data", True, None, None, None, None, None)
        fns = {}

        for (_, file) in dataset:
            if file.binary not in fns:
                fns[file.binary] = 1
            else:
                fns[file.binary] += 1

        import code
        code.interact(local=locals())