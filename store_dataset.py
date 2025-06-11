"""
Store the whole dataset in a pickle file.
"""

from dataclasses import dataclass
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
        dataset = LibDataset(self.data_path, context=self, main_process=True, pool_size=200, binary="openssl", seed=40332319787)
        print(dataset.data[199])
        with open(self.output_file, "w", encoding="utf-8") as file:
            for function, file_id in dataset.data:
                file.write(f"{file_id.optimization}, {file_id.platform}\n\n######\n")
                file.write(str(function))
                file.write("\n\n=====\n\n")
        
        print("wrote to file")