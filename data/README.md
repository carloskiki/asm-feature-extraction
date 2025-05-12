# File .asm.json

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

# `.merged.asm.json` file

The structure is the same as te `.asm.json` file. The function names are
human readable and likely come from debug symbols.

# `.unstrip.asm.json` file

Same as `.asm.json`, but contains much more comments & contains function names.
I suspect that the merged version is the merge between `unstrip` and the regular file, where
the function name from the `unstrip` file is used for the dissassembled output of the regular
file.