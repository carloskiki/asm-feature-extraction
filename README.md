# Extract human readable information by querying an LLM

## Binary Corp Issues

- The data does not contain the address of each instruction, and it is not split into blocks (so we don't know where jmp instructions lead to).

- For the lack of addresses, we could use relative addresses with 0 as the first instruction, but we would need to know size of each instruction
    (arm has a standard 8 bytes instruction size, but I know x86_64 is variable length, and I don't know about others).
- There is nothing we can do for the lack of labels, because even if the jump instruction is something like `jmp loc_8EA3`,
    We don't have the address of the instruction so we can't match the HEX value to an address.

## Pipeline

Data -> Query -> Structured JSON

### Retrieval

Generate new pool for all sources, OR generate one pool and run on all elements in the pool.

## Interopability

This will be generic over:
1. the llm you use.
2. the feature extraction methods (if we have multiple).

## Classification Methods

### List features

### Three (or N) ranks of features

General Class: _______
Confidence Level: _______
Reasoning: ____

... same for sub class & library & function

Examples:

- Cryptogrpahic
- Hash Function
- Sha256

- IO
- Socket connection
- TCP handshake

- IO
- Internal communication
- I2C ...


## Flow

1. Parse CLI args
2. Load the data 
3. Format & tokenize data
4. Train model & keep best
5. Inference

## Preprocessing we could do
### Jumps
If it jumps within the function we could either:
- Use a line number like CLAP
- Use a label like `label_X` or something similar
If it jumps outside of the funciton, is the assembly considered malformed?
We probably still want to handle that case, so we could use a token like `<UNK>` or something like that.

### Constants
We need to feed the constant literal, but keep it as small as possible. Prefix it as `0x` but also remove leading zeros while possible.

## Things to try:

- CLAP
- uptraining on our dataset

- Do the same as clap in terms of data processing & setup (but for firmware instead of Ubuntu binaries).

- Should we use a process similar to CLAP? (0-shot won't be very good)
    - We simply use 0-shot reasoning for the db?
