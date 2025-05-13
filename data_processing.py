from typing import Generator
import json

class Instruction:
    address: int
    mnemonic: str
    operands: list[str]

    def __init__(self, address: int, mnemonic: str, operands: list[str]):
        self.address = address
        self.mnemonic = mnemonic
        self.operands = operands

    def __str__(self):
        return f"    0x{self.address:X} {self.mnemonic} " + ', '.join(self.operands)

class Block:
    label: str
    instructions: list[Instruction]

    def __init__(self, label: str, instructions: list[Instruction]):
        self.label = label
        self.instructions = instructions
    
    def __str__(self):
        return f"{self.label}:\n" + '\n'.join(str(i) for i in self.instructions)

class Function:
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
        return f"{self.name}:\n" + '\n'.join(str(b) for b in self.blocks)

def process(contents: bytes) -> Generator[Function, None, None]:
    data = json.loads(contents)

    data["functions"].sort(key=lambda x : x['addr_start'])
    data["blocks"].sort(key=lambda x : x['addr_f'])

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
            print("Warning:", name, "has 0 blocks... skipping")
            continue
        
        yield Function(name, start, end, blocks)

def function_count(contents: bytes) -> int:
    data = json.loads(contents)
    return len(data["functions"])