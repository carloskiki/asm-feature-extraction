from typing import Optional
from context import Context
import random, sys

BINARY = ["busybox", "coreutils", "curl", "magick", "openssl", "puttygen", "sqlite3"]
PLATFORMS = ["arm", "gcc32", "gcc", "mips", "powerpc"]

class Retrieval:
    pool_size: int
    seed: int # Seed for selection of targets, choosed randomly if not set
    src_platform: Optional[str] # Run for a specific platform, run on all platforms if None
    src_binary: Optional[str] # Run for a specific binary, run on all binaries if None
    src_optimization: Optional[int] # Run for a specific optimization, run on all optimizations if None
    data_path: str

    @staticmethod
    def command(subparsers):
        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set"
        )
        parser.add_argument("--pool-size", type=int, default=10)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--src-platform", type=Optional[int], default=None)
        parser.add_argument("--src-binary", type=Optional[int], default=None)
        parser.add_argument("--src-optimization", type=Optional[int], default=None)
        parser.add_argument("data-path", type=str, dest="data_path", default="../lib-data")

def run(context: Context):
    pass