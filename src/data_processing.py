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
    "libcrypto": "libcrypto",
    "libgmp": "libgmp",
    "libmagickcore": "libMagickCore",
    "libtomcrypt": "libtomcrypt",
}

PLATFORMS = {
    "arm": "arm-linux-gnueabihf-gcc",
    "gcc32": "gcc32",
    "gcc": "gcc",
    "mips": "mips-linux-gnu-gcc",
    "powerpc": "powerpc-linux-gnu-gcc",
}

OBFUSCATED_BINARIES = {
    "libcrypto",
    "libgmp",
    "libmagickcore",
    "libtomcrypt",
}

OBFUSCATIONS = {
    "none": "none",
    "bcf": "bcf",
    "fla": "fla",
    "sub": "sub",
    "sub-fla-bcf": "sub-fla-bcf",
}

FILE_SUFFIX = ".bin.merged.asm.json.gz"


def normalize_obfuscation(obfuscation: Optional[str]) -> Optional[str]:
    if obfuscation is None:
        return None

    value = obfuscation.strip().lower()
    if value in {"", "none", "plain", "baseline"}:
        return "none"

    tokens = [token for token in value.split("-") if token]
    if not tokens:
        return "none"

    canonical = [token for token in ("sub", "fla", "bcf") if token in tokens]
    extras = sorted(token for token in tokens if token not in {"sub", "fla", "bcf"})
    return "-".join(canonical + extras)


def is_obfuscation_binary(binary: Optional[str]) -> bool:
    return binary is not None and binary in OBFUSCATED_BINARIES


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
        return "\n".join(str(b) for b in self.blocks)

@dataclass
class FileId:
    """
    A specific file in the dataset.
    """

    data_path: str
    binary: str
    platform: str
    optimization: int
    obfuscation: Optional[str] = None

    def path(self):
        """
        Return the file corresponding to the Id
        """

        if is_obfuscation_binary(self.binary):
            if self.platform != "clang":
                raise ValueError("Obfuscation binaries require platform='clang'")
            normalized_obfuscation = normalize_obfuscation(self.obfuscation) or "none"
            if normalized_obfuscation not in OBFUSCATIONS:
                raise ValueError(f"Unknown obfuscation '{normalized_obfuscation}'")

            return (
                f"{self.data_path}/{BINARIES[self.binary]}-clang-g-"
                f"{OBFUSCATIONS[normalized_obfuscation]}{FILE_SUFFIX}"
            )

        return (
            f"{self.data_path}/{BINARIES[self.binary]}-"
            f"{PLATFORMS[self.platform]}-g-O{self.optimization}{FILE_SUFFIX}"
        )

    def _sort_key(self) -> tuple:
        return (
            self.data_path,
            self.binary,
            self.platform,
            self.optimization,
            self.obfuscation or "none",
        )

    def __eq__(self, value) -> bool:
        if not isinstance(value, FileId):
            return NotImplemented
        return self._sort_key() == value._sort_key()

    def __lt__(self, other) -> bool:
        if not isinstance(other, FileId):
            return NotImplemented
        return self._sort_key() < other._sort_key()


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
        optimization: Optional[int] = None,
        platform: Optional[str] = None,
        obfuscation: Optional[str] = None,
    ):
        normalized_obfuscation = normalize_obfuscation(obfuscation)
        self.files = []

        if is_obfuscation_binary(binary):
            if optimization is not None:
                raise ValueError("--optimization is not supported for obfuscation binaries")
            if platform is not None and platform != "clang":
                raise ValueError("--platform must be 'clang' for obfuscation binaries")
            if normalized_obfuscation is not None and normalized_obfuscation not in OBFUSCATIONS:
                raise ValueError(f"Unknown obfuscation '{normalized_obfuscation}'")

            for obf in OBFUSCATIONS.keys() if normalized_obfuscation is None else [normalized_obfuscation]:
                self.files.append(FileId(path, binary, "clang", 0, obf))
        else:
            if normalized_obfuscation is not None:
                raise ValueError("--obfuscation is only supported with obfuscation binaries")

            binaries = (
                [b for b in BINARIES.keys() if b not in OBFUSCATED_BINARIES]
                if binary is None
                else [binary]
            )

            for b in binaries:
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
                functions = random.sample(functions, sample_size or len(functions))

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
        pool_size: Optional[int] = None,  # Take the whole dataset if None
        seed: Optional[int] = None,  # Don't randomize order if None
        binary: Optional[str] = None,
        optimization: Optional[int] = None,
        platform: Optional[str] = None,
        optimization_diff: Optional[int] = None,
        platform_diff: Optional[str] = None,
        obfuscation: Optional[str] = None,
        obfuscation_diff: Optional[str] = None,
    ):
        normalized_obfuscation = normalize_obfuscation(obfuscation)
        normalized_obfuscation_diff = normalize_obfuscation(obfuscation_diff)

        if is_obfuscation_binary(binary):
            if (
                platform is not None
                or platform_diff is not None
                or optimization is not None
                or optimization_diff is not None
            ):
                raise ValueError(
                    "--platform and --optimization are not supported for obfuscation binaries"
                )

            if normalized_obfuscation is not None and normalized_obfuscation not in OBFUSCATIONS:
                raise ValueError(f"Unknown obfuscation '{normalized_obfuscation}'")
            if normalized_obfuscation_diff is not None and normalized_obfuscation_diff not in OBFUSCATIONS:
                raise ValueError(f"Unknown obfuscation '{normalized_obfuscation_diff}'")

            if (
                normalized_obfuscation_diff is not None
                and (
                    normalized_obfuscation is None
                    or normalized_obfuscation == normalized_obfuscation_diff
                )
            ):
                raise ValueError("Conflict between query and target obfuscation sets")
        elif (
            optimization_diff is not None
            and (optimization is None or optimization == optimization_diff)
        ) or (
            platform_diff is not None
            and (platform is None or platform == platform_diff)
        ):
            raise ValueError("Conflict between query and target sets")

        if not is_obfuscation_binary(binary) and (
            normalized_obfuscation is not None or normalized_obfuscation_diff is not None
        ):
            raise ValueError("--obfuscation is only supported with obfuscation binaries")

        self.files: list[tuple[FileId, FileId]] = []
        if is_obfuscation_binary(binary):
            query_obfuscations = (
                [normalized_obfuscation]
                if normalized_obfuscation is not None
                else list(OBFUSCATIONS.keys())
            )

            for query_obfuscation in query_obfuscations:
                target_obfuscation = (
                    query_obfuscation
                    if normalized_obfuscation_diff is None
                    else normalized_obfuscation_diff
                )

                self.files.append(
                    (
                        FileId(path, binary, "clang", 0, query_obfuscation),
                        FileId(path, binary, "clang", 0, target_obfuscation),
                    )
                )
        else:
            binaries = (
                [b for b in BINARIES.keys() if b not in OBFUSCATED_BINARIES]
                if binary is None
                else [binary]
            )

            for b in binaries:
                for p in PLATFORMS.keys() if platform is None else [platform]:
                    p_diff = p if platform_diff is None else platform_diff
                    for o in range(4) if optimization is None else [optimization]:
                        o_diff = o if optimization_diff is None else optimization_diff
                        self.files.append(
                            (FileId(path, b, p, o), FileId(path, b, p_diff, o_diff))
                        )

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

                target_index = bisect_left(
                    [f.name for f in target_functions], query_function.name
                )

                # No match, or if the function is very small (e.g., external functions)
                if (
                    target_index == len(target_functions)
                    or target_functions[target_index].name != query_function.name
                    # Random heuristic for small functions.
                    or len(str(query_function)) < 50
                ):
                    continue

                function_pairs.append((query_function, target_functions[target_index]))

            self.functions.extend(function_pairs)

        self.main_process = main_process

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, idx: int) -> tuple[Function, Function]:
        return self.functions[idx]

    def __getitems__(self, idxs: list[int]) -> list[tuple[Function, Function]]:
        return [self.functions[i] for i in idxs]
