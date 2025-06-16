"""
Data processing
"""

from typing import Generator, Optional
from dataclasses import dataclass
import gzip
import random
import json
from itertools import islice
from torch.utils.data import Dataset
from tqdm import tqdm

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


class Instruction:
    """
    Single assembly instruction.
    """

    address: int
    mnemonic: str
    operands: list[str]

    def __init__(self, address: int, mnemonic: str, operands: list[str]):
        self.address = address
        self.mnemonic = mnemonic
        self.operands = operands

    def __str__(self):
        return f"    {self.mnemonic} " + ", ".join(self.operands)

class Block:
    """
    A labeled block of assembly instructions.
    """

    label: str
    instructions: list[Instruction]

    def __init__(self, label: str, instructions: list[Instruction]):
        self.label = label
        self.instructions = instructions

    def __str__(self):
        return f"{self.label}:\n" + "\n".join(str(i) for i in self.instructions)


class Function:
    """
    A function compiled to assembly i.e., a list of blocks.
    """

    name: str
    start: int
    end: int
    blocks: list[Block]

    def __init__(self, name: str, start: int, end: int, blocks: list[Block]):
        self.name = name
        self.start = start
        self.end = end
        self.blocks = blocks

    def __str__(self):
        return f"{self.name}:\n" + "\n".join(str(b) for b in self.blocks)


@dataclass
class FileId:
    """
    A specific file in the dataset.
    """

    data_path: str
    binary: str
    platform: str
    optimization: int

    def path(self):
        """
        Return the file corresponding to the Id
        """

        return f"{self.data_path}/{BINARIES[self.binary]}-{PLATFORMS[self.platform]}-g-O{self.optimization}.bin.merged.asm.json.gz"


def process(contents: bytes) -> Generator[Function, None, None]:
    """
    Process the contents of a `.merged.asm.json` file.
    """

    data = json.loads(contents)

    data["functions"].sort(key=lambda x: x["addr_start"])
    data["blocks"].sort(key=lambda x: x["addr_f"])

    index: int = 0
    for function in data["functions"]:
        name = function["name"]
        start = function["addr_start"]
        end = function["addr_end"]
        blocks = []

        while index < len(data["blocks"]) and data["blocks"][index]["addr_f"] == start:
            block = data["blocks"][index]
            label = block["name"]
            instructions = []

            for instruction in block["ins"]:
                address = instruction["ea"]
                mnemonic = instruction["mne"]
                operands = instruction["oprs"]
                instructions.append(Instruction(address, mnemonic, operands))

            blocks.append(Block(label, instructions))

            index += 1

        if len(blocks) == 0:
            continue

        yield Function(name, start, end, blocks)


def function_count(contents: bytes) -> int:
    """
    Count the number of functions in a `merged.asm.json` file
    """

    data = json.loads(contents)
    return len(data["functions"])


class LibDataset(Dataset):
    files: list[FileId] # There may be the possibility that a file here is not used (if using TargetDataset), but whatever...
    functions: list[Function]
    main_process: bool

    def __init__(
        self,
        path: str,
        main_process: bool,
        pool_size: Optional[int] = None,  # Take the whole dataset if not specified
        seed: Optional[int] = None,  # Don't randomize order if not specified
        binary: Optional[str] = None,
        optimization: Optional[str] = None,
        platform: Optional[str] = None,
    ):
        def data_files() -> Generator[FileId, None, None]:
            """
            return all files that match the selected parameters
            """
            for b in BINARIES.keys() if binary is None else [binary]:
                for p in PLATFORMS.keys() if platform is None else [platform]:
                    for o in range(4) if optimization is None else [optimization]:
                        yield FileId(path, b, p, o)

        def iter_sample(iterator, sample_size):
            """
            Sample from an iterator
            """

            rng = random.Random(seed)
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

        self.files = list(data_files())
        self.functions = []
        for index, file in enumerate(
            tqdm(self.files, desc="Reading dataset", disable=not main_process)
        ):
            if pool_size is None:
                sample_size = None
            elif index == len(self.files) - 1:
                sample_size = pool_size - (len(self.files) - 1) * (pool_size // len(self.files))
            else:
                sample_size = pool_size // len(self.files)

            if sample_size == 0:
                continue

            with gzip.open(file.path(), "rb") as file_data:
                functions = process(file_data.read())

            for sample in (
                islice(functions, sample_size)
                if seed is None
                else iter_sample(functions, sample_size)
            ):
                self.functions.append(sample)

        self.functions.sort(key=lambda fn: fn.name)
        self.main_process = main_process

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, idx: int) -> tuple[Function, FileId]:
        return self.functions[idx]

    def __getitems__(self, idxs: list[int]) -> list[tuple[Function, FileId]]:
        return [self.functions[i] for i in idxs]


class TargetDataset(Dataset):
    functions: list[Function]

    def __init__(self, queries: LibDataset, optimization_diff: Optional[int], platform_diff: Optional[int]):
        fn_name_set = set([f.name for f in queries.functions])
        self.functions = []

        for file in tqdm(queries.data, disable=not queries.main_process, desc="Reading target dataset"):

            if optimization_diff is not None:
                file.optimization = optimization_diff
            if platform_diff is not None:
                file.platform = platform_diff
            
            with gzip.open(file.path(), "rb") as file_data:
                target_functions = process(file_data.read())

                for fn in target_functions:
                    if fn.name in fn_name_set:
                        fn_name_set.remove(fn.name)
                        self.functions.append(fn)
        
        self.functions.sort(key=lambda fn: fn.name)
        queries.functions[:] = [x for x in queries.funtions if x not in fn_name_set]

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, idx: int) -> Function:
        return self.functions[idx]

    def __getitems__(self, idxs: list[int]) -> list[Function]:
        return [self.functions[i] for i in idxs]