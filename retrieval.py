from typing import Optional
import random
import sys

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
    src_platform: Optional[
        str
    ]  # Run for a specific platform, run on all platforms if None
    src_binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    src_optimization: Optional[
        int
    ]  # Run for a specific optimization, run on all optimizations if None
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
            "--src-platform", type=Optional[str], choices=PLATFORMS.keys(), default=None
        )
        parser.add_argument(
            "--src-binary", type=Optional[int], choices=BINARIES.keys(), default=None
        )
        parser.add_argument(
            "--src-optimization", type=Optional[int], choices=range(4), default=None
        )
        parser.add_argument(
            "data-path", type=str
        )