"""
Retrieval CLI utilities
"""

import random
import sys
import gzip
import itertools
from typing import Optional, Generator
from data_processing import Function, function_count, process

BINARIES = {
    "busybox": "busybox_unstripped",
    "coreutils": "coreutils",
    "curl": "curl",
    "image-magick": "magick",
    "openssl": "openssl",
    "putty": "puttygen",
    "sqlite3": "sqlite3",
}

PLATFORMS = {
    "arm": "arm-linux-gnueabihf-gcc",
    "gcc32": "gcc32",
    "gcc": "gcc",
    "mips": "mips-linux-gnu-gcc",
    "powerpc": "powerpc-linux-gnu-gcc",
}

class FileId:
    """
    A specific file in the dataset.
    """

    data_path: str
    binary: str
    platform: str
    optimization: int

    def from_values(self, data_path, binary, platform, optimization):
        """
        Create the FileId from the provided values
        """

        self.data_path = data_path
        self.binary = binary
        self.platform = platform
        self.optimization = optimization

    def path(self):
        """
        Return the file corresponding to the Id
        """

        return f"{self.data_path}/{BINARIES[self.binary]}-{PLATFORMS[self.platform]}-g-O{self.optimization}.bin.merged.asm.json.gz"


class Retrieval:
    """
    CLI command to evaluate function retrieval
    """

    pool_size: int
    seed: int  # Seed for selection of targets, choosed randomly if not set
    src_binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    src_platform: Optional[
        str
    ]  # Run for a specific platform, run on all platforms if None
    src_optimization: Optional[
        int
    ]  # Run for a specific optimization, run on all optimizations if None
    src_function: Optional[str]
    data_path: str
    target_platform: Optional[str]
    target_optimization: Optional[str]
    same_binary: bool # Keep the same binary for the target pool
    same_platform: bool # keep the 
    same_optimization: bool
    from_pool: bool

    @staticmethod
    def command(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=100)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--src-binary", type=str, choices=BINARIES.keys())
        parser.add_argument(
            "--src-platform", type=Optional[str], choices=PLATFORMS.keys(), default=None
        )
        parser.add_argument(
            "--src-optimization", type=Optional[int], choices=range(4), default=None
        )
        parser.add_argument("--src-function")
        parser.add_argument("data_path", type=str)

    def data_file(self) -> FileId:
        """
        Get the file selected with the CLI arguments, using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)
        file = FileId()

        file.data_path = self.data_path
        if self.src_binary is None:
            file.binary = rng.choice(list(BINARIES.keys()))
        else:
            file.binary = self.src_binary

        if self.src_platform is None:
            file.platform = rng.choice(list(PLATFORMS.keys()))
        else:
            file.platform = self.src_platform

        if self.src_optimization is None:
            file.optimization = rng.randrange(4)
        else:
            file.optimization = self.src_optimization

        return file

    def data_files(self) -> Generator[FileId, None, None]:
        """
        return all files that match the selected parameters
        """

        if self.src_binary is None:
            binary = BINARIES.keys()
        else:
            binary = [self.src_binary]
        if self.src_platform is None:
            platform = PLATFORMS.keys()
        else:
            platform = [self.src_platform]
        if self.src_optimization is None:
            optimization = range(4)
        else:
            optimization = [self.src_optimization]

        for b in binary:
            for p in platform:
                for o in optimization:
                    file = FileId()
                    file.from_values(self.data_path, b, p, o)
                    yield file

    def source_function(self) -> Function:
        """
        Return the function selected with the CLI using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)

        with gzip.open(self.data_file().path()) as file:
            data = file.read()

        functions = process(data)
        if self.src_function is None:
            index = rng.randrange(function_count(data))
            return next(itertools.islice(functions, index, None))
        return next(f for f in process(data) if f.name == self.src_function)

    def generate_pool(self) -> list[tuple[Function, FileId]]:
        """
        Get the pool of targets
        """

        files = list(self.data_files())
        print(files)
        functions_per_file = self.pool_size // len(files)
        last_file_function_count = self.pool_size - (len(files) - 1) * functions_per_file

        pairs = []
        for index, file in enumerate(files):
            if index == len(files) - 1:
                sample_size = last_file_function_count
            else:
                sample_size = functions_per_file

            with gzip.open(file.path(), "rb") as file_data:
                functions = process(file_data.read())

            for sample in self.iter_sample(functions, sample_size):
                sample: Function = sample
                pairs.append((sample, file))
            
        return pairs


    def iter_sample(self, iterator, sample_size):
        """
        Sample from an iterator
        """

        rng = random.Random(self.seed)
        results = []
        # Fill in the first samplesize elements:
        try:
            for _ in range(sample_size):
                results.append(next(iterator))
        except StopIteration as exc:
            raise exc
        rng.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, sample_size):
            r = rng.randint(0, i)
            if r < sample_size:
                results[r] = v  # at a decreasing rate, replace random items
        return results
