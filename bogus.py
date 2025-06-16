"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
import context
from data_processing import LibDataset, TargetDataset

@dataclass
class Bogus(context.Context):
    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        subparsers.add_parser(
            "bogus",
            description="",
        )

    
    def __call__(self):
        dataset = LibDataset("lib-data", main_process=True, pool_size=None, seed=None)
        pool_dataset = TargetDataset(dataset, optimization_diff=1, platform_diff=None)
        
        print(set([f.name for f in dataset.functions]).difference(set([f.name for f in pool_dataset.functions])))