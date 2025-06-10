"""
Data processing
"""

from typing import Generator, Optional
from dataclasses import dataclass
import gzip
import random
import json
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
from context import Context

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
        return f"    0x{self.address:X} {self.mnemonic} " + ", ".join(self.operands)


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


class LibDataset(IterableDataset):
    data: list[tuple[Function, FileId]]
    index: int
    tokenizer: PreTrainedTokenizer
    context: Context

    def __init__(
        self,
        path: str,
        binary: Optional[str],
        optimization: Optional[str],
        platform: Optional[str],
        pool_size: int,
        seed: int,
        context: Context,
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

        files = list(data_files())
        functions_per_file = pool_size // len(files)
        last_file_function_count = pool_size - (len(files) - 1) * functions_per_file

        pairs = []
        for index, file in enumerate(tqdm(files, desc="Reading dataset")):
            if index == len(files) - 1:
                sample_size = last_file_function_count
            else:
                sample_size = functions_per_file

            if sample_size == 0:
                continue

            with gzip.open(file.path(), "rb") as file_data:
                functions = process(file_data.read())

            for sample in iter_sample(functions, sample_size):
                sample: Function = sample
                pairs.append((sample, file))
        self.data = pairs
        self.index = 0
        self.tokenizer = context.get_tokenizer()
        self.context = context

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> BatchEncoding:
        f, _ = self.data[self.index]
        prompt = self.context.get_prompt(str(f))
        chat = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        self.index += 1
        return self.tokenizer(
            chat,
            truncation=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )


    def __getitem__(self, idx: int):
        f, _ = self.data[idx]
        prompt = self.context.get_prompt(str(f))
        chat = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(
            chat,
            truncation=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

    def __getitems__(self, idxs: list[int]):
        prompts = []
        for idx in idxs:
            f, _ = self.data[idx]
            prompts.append(self.context.get_prompt(str(f)))
        
        chat = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(
                chat,
                truncation=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )