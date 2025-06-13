"""
Doesn't do much, just to play around & try stuff
"""

from dataclasses import dataclass
import context
from data_processing import LibDataset, TargetDataset

@dataclass
class Bogus(context.Context):
    data_path: str
    output_file: str

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "bogus",
            description="",
        )
        parser.add_argument("data_path", type=str)
        parser.add_argument("output_file", type=str)

    
    def __call__(self):
        dataset = LibDataset(self.data_path, main_process=True, pool_size=2048, binary="openssl", optimization=2, seed=40332319787)
        pool_dataset = TargetDataset(dataset, optimization_diff=1, platform_diff=None)
        
        assert([f.name for f in dataset.flattened] == [f.name for f in pool_dataset.flattened])