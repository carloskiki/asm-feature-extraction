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

def process(file: str) -> Generator[Function, None, None]:
    with open(file, "r") as file:
        data = json.loads(file.read())

    index: int = 0
    for function in data["functions"]:
        name = function["name"]
        start = function["addr_start"]
        end = function["addr_end"]
        blocks = []

        for _ in range(function["bbs_len"]):
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
        
        yield Function(name, start, end, blocks)