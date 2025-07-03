"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
from argparse import ArgumentParser
import context

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
        pass