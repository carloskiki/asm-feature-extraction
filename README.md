# Extract human readable information by querying an LLM

Give the LLM information such as the target, abi, etc.

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
