# Extract human readable information by querying an LLM

## Relabeling of jumps and constants
### Jumps
If it jumps within the function we could either:
- Use a line number like CLAP
- Use a label like `label_X` or something similar
If it jumps outside of the funciton, is the assembly considered malformed?
We probably still want to handle that case, so we could use a token like `<UNK>` or something like that.

### Constants
We need to feed the constant literal, but keep it as small as possible. Prefix it as `0x` but also remove leading zeros while possible.

## Questions for Steven
Things to try:
- CLAP
- Custom training on our dataset

- Do the same as clap in terms of data processing & setup (but for firmware instead of Ubuntu binaries).
- Do we rebase the jump addresses like in CLAP?

- Should we use a process similar to CLAP? (0-shot won't be very good)
    - We simply use 0-shot reasoning for the db?



#### Unlike clap 
- No custom embeddings

## Classification

Could have multiple ranks of features:
- General Class
- Sub Class
- Close to exact match

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
