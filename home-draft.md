#### Core logic & Operations

This section specifies how to determine the kind of operation that the assembly function performs.

The information collected includes:

- Indication of loops. This is determined by the presence of jump instructions that point back to a
    previous instruction after some conditions have been checked.
- Indication of jump tables. Evaluated by patterns suggesting calculated jump addresses based on
    indices, or a series of conditional jumps.
- Extensive use of indexed addressing modes.
- Use of SIMD instructions and registers.
- Number of distinct subroutine call targets.
- Overall logical behavior. Possibilities include:
  - Arithmetic operations
  - Bitwise operations
  - Data movement and memory access.
  - Control flow and dispatching operations.
  - Memory access operations.

#### Notable constants

This section identifies notable constants. These could be common scalar values used by a specific
cryptographic algorithm, or the signature bytes used by a file format or protocol.

#### Side effects

The prompt also monitors the side effects that the assembly function has on the system. This includes:

- Modification of input arguments.
- Modification of global state.
    This is detected when writes to absolute memory addresses or addresses resolved via global data segment pointers occur.
- Memory allocation and deallocation.
    Detected by the presence of calls to memory management functions like `malloc`, `free`, or similar.
- Linear memory access patterns.
    Determined by the presence of sequential indexed memory accesses inside loops or across multiple instructions.
- System calls and software interrupts.
    This is identified by the presence of specific instructions that trigger system calls or software interrupts.

#### Final categorization

The last section tries to assign a overall category to the assembly function, by basing it on the information
collected in the analysis. The final categorization only weakly supports the similarity search because it does
not have a large impact on the similarity score. Its purpose is to provide a concise overview for reviewers
of the analysis, who might want to understand the function or verify its similarity with the target.
Categories include: cryptographic, data processing, control flow and dispatch, initialization, error handling,
wrapper/utility, file management, etc.
