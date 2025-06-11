# Extract human readable information by querying an LLM

## Assumptions

- There is no bad actor trying to mangle or obfuscate the generated assembly.
- Assembly will be similar except for the presence of dbg symbols &| the optimization level &| the architecture used.


## Questions

- How do we handle short and long functions?
- How to prompt properly
- How to parallel stuff better

## Datasets
### Libraries

Contains 383_658 assembly functions

- Some functions have 0 blocks, so we skip them.
- Some functions are too long for the context window of most llms (e.g., 100k tokens), we could truncate them or skip them or sliding window.

### Binary Corp Issues

- The data does not contain the address of each instruction, and it is not split into blocks (so we don't know where jmp instructions lead to).

- For the lack of addresses, we could use relative addresses with 0 as the first instruction, but we would need to know size of each instruction
    (arm has a standard 8 bytes instruction size, but I know x86_64 is variable length, and I don't know about others).
- There is nothing we can do for the lack of labels, because even if the jump instruction is something like `jmp loc_8EA3`,
    We don't have the address of the instruction so we can't match the HEX value to an address.

## Pipeline

Data -> Query -> Structured JSON

Either generate sources from pool, or generate pool for each source.

### Retrieval

Generate new pool for all sources, OR generate one pool and run on all elements in the pool.


## Papers

- [A Survey of Binary Code Similarity Detection Techniques](https://www.mdpi.com/2079-9292/13/9/1715)
- [UniASM: Binary code similarity detection without fine-tuning](https://arxiv.org/abs/2211.01144)

# Data Schema

## `bin` field
```json
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
  "strings": [], // List of strings contained in the binary (presumably obtained by running `strings` on the binary)
  "seg": [], // The segments contained in the binary
  "sha256": "8d51289b5a4a59a57774c7522db700d76130c02d1929eddc638f77ec14f56dd0"
}
```

## `blocks` field
A list of blocks, where each block is:
```json
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

```json
{
  "ea": 134513132, // Effective Address
  "mne": "CALL", // Mnemonic
  "oprs": [ // Operands
    "sub_8048820"
  ],
  "oprs_tp": [ // Operands "tp"
    7
  ],
  "dr": [], // Data referenced addresses
  "cr": [ // Outgoing address (matches here but not sure in general)
    134514720
  ]
}
```

## `comments` field

```json
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

```json
{
  "name": "sub_80B3F27",
  "description": "",
  "addr_start": 134954791,
  "addr_end": 134954946,
  "calls": [],
  "tags": [],
  "bbs_len": 15 // The number of blocks that this function contain
}
```

It seems like the blocks are ordered by function e.g:
```
  ___f0___ _f1__ ___f2___
[ b0 b1 b2 b3 b4 b5 b6 b7 ]
```

## `functions_src` field

Array of function sources, empty for our case

## `.merged.asm.json` file

The structure is the same as te `.asm.json` file. The function names are
human readable and likely come from debug symbols.

## `.unstrip.asm.json` file

Same as `.asm.json`, but contains much more comments & contains function names.
I suspect that the merged version is the merge between `unstrip` and the regular file, where
the function name from the `unstrip` file is used for the dissassembled output of the regular
file.

