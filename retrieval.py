"""
Retrieval CLI utilities
"""

import random
import sys
import gzip
import itertools
import glob
from typing import Optional, Iterator
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

    @staticmethod
    def command(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=10)
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

    def data_file(self) -> str:
        """
        Get the file selected with the CLI arguments, using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)

        if self.src_binary is None:
            binary = rng.choice(BINARIES.keys())
        else:
            binary = self.src_binary

        if self.src_platform is None:
            platform = rng.choice(PLATFORMS.keys())
        else:
            platform = self.src_platform

        if self.src_optimization is None:
            optimization = rng.randrange(4)
        else:
            optimization = self.src_optimization

        return f"{self.data_path}/{BINARIES[binary]}-{PLATFORMS[platform]}-g-O{optimization}.bin.merged.asm.json.gz"

    def all_files(self) -> Iterator[str]:
        """
        return all files that match the selected parameters
        """

        if self.src_binary is None:
            binary = "*"
        else:
            binary = BINARIES[self.src_binary]
        if self.src_platform is None:
            platform = "*"
        else:
            platform = BINARIES[self.src_platform]
        if self.src_optimization is None:
            optimization = "*"
        else:
            optimization = str(self.src_optimization)

        return glob.iglob(
            f"{self.data_path}/{BINARIES[binary]}-{PLATFORMS[platform]}-g-O{optimization}.bin.merged.asm.json.gz"
        )

    def source_function(self) -> Function:
        """
        Return the function selected with the CLI using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)

        with gzip.open(self.data_file()) as file:
            data = file.read()

        functions = process(data)
        if self.src_function is None:
            index = rng.randrange(function_count(data))
            return next(itertools.islice(functions, index, None))
        return next(f for f in process(data) if f.name == self.src_function)
