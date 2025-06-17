"""
Data processing
"""

from typing import Optional
from bisect import bisect_left
from dataclasses import dataclass
import gzip
import random
import json
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

    def __eq__(self, value) -> bool:
        if not isinstance(value, FileId):
            return NotImplemented
        return (
            self.data_path == value.data_path
            and self.binary == value.binary
            and self.platform == value.platform
            and self.optimization == value.optimization
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, FileId):
            return NotImplemented
        return (self.data_path, self.binary, self.platform, self.optimization) < (
            other.data_path,
            other.binary,
            other.platform,
            other.optimization,
        )


def process(contents: bytes) -> list[Function]:
    """
    Process the contents of a `.merged.asm.json` file.
    """

    data = json.loads(contents)

    data["functions"].sort(key=lambda x: x["addr_start"])
    data["blocks"].sort(key=lambda x: x["addr_f"])

    collected = []
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

        collected.append(Function(name, start, end, blocks))

    return collected


class LibDataset(Dataset):
    files: list[
        FileId
    ]  # There may be the possibility that a file here is not used (if using TargetDataset), but whatever...
    functions: list[tuple[Function, FileId]]
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
        self.files = []

        for b in BINARIES.keys() if binary is None else [binary]:
            for p in PLATFORMS.keys() if platform is None else [platform]:
                for o in range(4) if optimization is None else [optimization]:
                    self.files.append(FileId(path, b, p, o))

        self.functions = []
        for index, file in enumerate(
            tqdm(self.files, desc="Reading dataset", disable=not main_process)
        ):
            if pool_size is None:
                sample_size = None
            elif index == len(self.files) - 1:
                sample_size = pool_size - (len(self.files) - 1) * (
                    pool_size // len(self.files)
                )
            else:
                sample_size = pool_size // len(self.files)

            if sample_size == 0:
                continue

            with gzip.open(file.path(), "rb") as file_data:
                functions = process(file_data.read())

            if seed is None:
                functions = functions[:sample_size]
            else:
                functions = random.sample(functions, sample_size)

            self.functions.extend([(f, file) for f in functions])

        self.main_process = main_process

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, idx: int) -> tuple[Function, FileId]:
        return self.functions[idx]

    def __getitems__(self, idxs: list[int]) -> list[tuple[Function, FileId]]:
        return [self.functions[i] for i in idxs]


class PairsDataset(Dataset):
    files: list[tuple[FileId, FileId]]
    functions: list[tuple[Function, Function]]

    def __init__(
        self,
        path: str,
        main_process: bool,
        pool_size: Optional[int] = None,  # Take the whole dataset if not specified
        seed: Optional[int] = None,  # Don't randomize order if not specified
        binary: Optional[str] = None,
        optimization: Optional[str] = None,
        platform: Optional[str] = None,
        optimization_diff: Optional[int] = None,
        platform_diff: Optional[str] = None,
    ):

        if (
            optimization_diff is not None
            and (optimization is None or optimization == optimization_diff)
        ) or (
            platform_diff is not None
            and (platform is None or platform == platform_diff)
        ):
            raise ValueError("Conflict between query and target sets")

        self.files: list[tuple[FileId, FileId]] = []
        for b in BINARIES.keys() if binary is None else [binary]:
            for p in PLATFORMS.keys() if platform is None else [platform]:
                p_diff = p if platform_diff is None else platform_diff
                for o in range(4) if optimization is None else [optimization]:
                    o_diff = o if optimization_diff is None else optimization_diff
                    self.files.append((FileId(path, b, p, o), FileId(path, b, p_diff, o_diff)))

        self.functions = []
        for index, (query, target) in enumerate(
            tqdm(self.files, desc="Reading dataset", disable=not main_process)
        ):
            if pool_size is None:
                sample_size = None
            elif index == len(self.files) - 1:
                sample_size = pool_size - (len(self.files) - 1) * (
                    pool_size // len(self.files)
                )
            else:
                sample_size = pool_size // len(self.files)

            if sample_size == 0:
                continue

            with gzip.open(query.path(), "rb") as file_data:
                query_functions = process(file_data.read())

            with gzip.open(target.path(), "rb") as file_data:
                target_functions = process(file_data.read())

            random.seed(seed)
            random.shuffle(query_functions)
            target_functions.sort(key=lambda x: x.name)

            function_pairs = []

            for query_function in query_functions:
                if len(function_pairs) == sample_size:
                    break

                target_index = bisect_left([f.name for f in target_functions], query_function.name)

                # No match, continue
                if target_functions[target_index].name != query_function.name:
                    continue

                function_pairs.append(query_function, target_functions[target_index])


            self.functions.extend(function_pairs)

        self.main_process = main_process

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, idx: int) -> tuple[tuple[Function, Function]]:
        return self.functions[idx]

    def __getitems__(
        self, idxs: list[int]
    ) -> list[tuple[Function, Function]]:
        return [self.functions[i] for i in idxs]