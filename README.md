# Interpretable feature extraction from assembly code

All experiments and tools used for our research.

## Modules
- [`data`](./data/): The dataset used in our research.
- [`metrics`](./metrics/): Retrieved results from our experiments.
- [`graphs`](./graphs/): Graph generation code for the paper.
- [`prompts`](./prompts/): Prompts along with provided examples used for the experiments.
- [`src`](./src/): Source code for our experimentation tool and some of the reproductions of baselines.
    All can be invoked through `main.py`.

# Dataset Schema

Each `merged.asm.json` file contains the disassembly output from IDA on the specified binary. the files are "merged"
because they contain the body of the assembly functions stripped of debug symbols, while keeping the original function
names for matching and debugging purposes.

## `bin` field

This is of little utility for our purposes. It contains metadata about the binary:
```jsonc
{
  "name": "D:\\data-bk\\optimization-cross-arch\\l-openssl-openssl\\openssl-gcc32-g-O0.bin.tmp\\openssl-gcc32-g-O0.bin",
  "base": 134512640,
  "entry_points": [
    18446744073709552000,
    18446744073709552000,
    18446744073709552000
  ],
  "architecture": "metapc",
  "endian": "le",
  "bits": "b32",
  "disassembler": "ida",
  "compiler": "GNU C++",
  "data": {},
  "import_modules": [],
  "import_functions": {},
  "export_functions": {
    "134513128": ".init_proc",
    "137202804": ".term_proc",
    "134514678": "start"
  },
  "disassembled_at": "2021-04-09T22:44:15.361078",
  "functions_count": 9462,
  "strings": [], // List of strings contained in the binary
  "seg": [], // The segments contained in the binary
  "sha256": "8d51289b5a4a59a57774c7522db700d76130c02d1929eddc638f77ec14f56dd0"
}
```

## `blocks` field
The list of basic blocks:
```jsonc
{
  "addr_start": 134513128,
  "addr_end": 134513153,
  "name": "loc_80481E8", // loc_<hex> where hex is equal to addr_start
  "addr_f": 134513128, // start of the function that contains this block
  "calls": [
    134514720,
    134513153,
    134513158
  ],
  "ins": [] // Instructions for the block
}
```

### `ins` element

A single instruction:
```jsonc
{
  "ea": 134513132, // Effective Address
  "mne": "CALL", // Mnemonic
  "oprs": [ // Operands
    "sub_8048820"
  ],
  "oprs_tp": [ // Operands type
    7
  ],
  "dr": [], // Data referenced addresses
  "cr": [ // Outgoing address (matches here but not sure in general)
    134514720
  ]
}
```

## `comments` field
Added comments to the disassembly. Not useful for us.

```jsonc
{
  "author": "ida",
  "category": 3,
  "content": "status",
  "blk": 134514001,
  "address": 134514001,
  "created_at": "2021-04-09T22:44:17.810120"
}
```

## `functions` field
The list of functions:

```jsonc
{
  "name": "sub_80B3F27",
  "description": "",
  "addr_start": 134954791,
  "addr_end": 134954946,
  "calls": [],
  "tags": [],
  "bbs_len": 15 // The number of blocks that this function contains
}
```

## `functions_src` field
Array of function sources. Empty for our purposes.
