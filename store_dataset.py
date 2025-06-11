"""
Store the whole dataset in a pickle file.
"""

from dataclasses import dataclass
import pickle
import context
from data_processing import LibDataset

@dataclass
class StoreDataset(context.Context):
    data_path: str
    output_file: str

    @staticmethod
    def command(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "store-dataset",
            description="Store the whole dataset in a pickle file",
        )
        parser.add_argument("data_path", type=str)
        parser.add_argument("output_file", type=str)

    
    def __call__(self):
        dataset = LibDataset(self.data_path, context=self, main_process=True)
        function_count = len(dataset.data)
        with open(self.output_file, "wb") as file:
            bytes_written = pickle.dump(dataset.data, file)
        
        print(f"Wrote {bytes_written:_} bytes to {self.output_file}.")
        print(f"In total, {function_count} assembly functions were processed.")