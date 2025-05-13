from typing import Optional, Iterator
import random
import sys
from data_processing import Function, function_count, process
import gzip
import itertools
import glob

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
        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=10)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument(
            "--src-binary", type=Optional[int], choices=BINARIES.keys(), default=None
        )
        parser.add_argument(
            "--src-platform", type=Optional[str], choices=PLATFORMS.keys(), default=None
        )
        parser.add_argument(
            "--src-optimization", type=Optional[int], choices=range(4), default=None
        )
        parser.add_argument("--src-function")
        parser.add_argument("data-path", type=str)

    def data_file(self) -> str:
        rng = random.Random(self.seed)

        if self.src_binary is None:
            self.src_binary = rng.choice(BINARIES.keys())
        if self.src_platform is None:
            self.src_platform = rng.choice(PLATFORMS.keys())
        if self.src_optimization is None:
            self.src_optimization = rng.randrange(4)

        return f"{self.data_path}/{BINARIES[self.src_binary]}-{PLATFORMS[self.src_platform]}-g-O{self.src_optimization}.bin.merged.asm.json.gz"

    def all_files(self) -> Iterator[str]:
        if self.src_binary is None:
            self.src_binary = "*"
        if self.src_platform is None:
            self.src_platform = "*"
        if self.src_optimization is None:
            self.src_optimization = "*"

        return glob.iglob(
            f"{self.data_path}/{BINARIES[self.src_binary]}-{PLATFORMS[self.src_platform]}-g-O{self.src_optimization}.bin.merged.asm.json.gz"
        )

    def source_function(self) -> Function:
        rng = random.Random(self.seed)

        with gzip.open(self.data_file()) as file:
            data = file.read()

        functions = process(data)
        if self.src_function is None:
            index = rng.randrange(function_count(data))
            return next(itertools.islice(functions, index, None))
        return next(f for f in process(data) if f.name == self.src_function)
