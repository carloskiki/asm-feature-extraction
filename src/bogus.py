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
        dataset = LibDataset("data", True, None, None, None, None, None)
        fn_count = {}
        fn_length = {}

        for (fn, file) in dataset:
            if file.binary not in fn_count:
                fn_count[file.binary] = 1
                fn_length[file.binary] = [instruction_count(fn)]
            else:
                fn_count[file.binary] += 1
                fn_length[file.binary].append(instruction_count(fn))
        
        import code
        code.interact(local=locals())

def instruction_count(fn: Function) -> int:
    return sum(len(block.instructions) for block in fn.blocks)
