# TODOs

- [ ] Add references to the binaries in dataset.


## Background

Binary code similarity detection becomes more important as the modularity and rate of production of software grows.
Modern software is almost never written from scratch, and is becoming increasingly reliant on external libraries.
For reverse engineers, reducing the amount of repetitive assembly functions to analyze is important,
    as it allows them to be more efficient and focus on the custom parts of a binary.
If a vulnerability is found in a library, it is important to be able to quickly identify if an unknown or proprietary binary
    is using the vulnerable library, so that it can be mitigated.
Other applications of BCSD include license compliance and plagiarism detection.

## Related Work

### Static Analysis

Some methods make use of static analysis to detect clone assembly routines. With these methods, a trade-off has
to be made between the robustness to obfuscation and architecture differences and the performance of the algorithm. [1]
Control flow graph analysis and comparison [ref] is known to be very robust to syntactic differences, but involves resource-intensive
computations. Simpler algorithms that use heuristics such as instruction frequency or longest-common-subsequence (LCS) [ref] are more
efficient but tend to fixate on the syntactic elements rather than the semantics.

### Dynamic Analysis

Dynamic analysis consists of analyzing the features of a binary or code fragment by monitoring its runtime behavior.
This methods is costly, but completely sidesteps the syntactic aspects of binary and solely focuses on the semantics. [2]
As such, this method works very well for cross-architecture, cross-optimization and cross-obfuscation analysis.

### Machine Learning

Machine learning approaches 

## Methodology

### Dataset

The dataset is composed of 7 binaries: busybox, coreutils, curl, image-magick, openssl, putty, and sqlite3.
All were compiled using gcc for the following platforms: x86_64, x86_32, arm, mips, powerpc.
For each binary and platform, binary objects were generated for each optimization level (O0 to O3).
Stripped all debug symbols except for function names, so as to be able to match functions across binaries.
In total, yeilds 140 different binaries to analyze.
The binaries were dissassembled using IDA Pro, yielding 383_658 assembly routines.


# Refs

1. [A Survey of Binary Code Similarity Detection Techniques](https://www.mdpi.com/2079-9292/13/9/1715)
2. [Binary Code Similiarity Detection](https://ieeexplore.ieee.org/document/9678518)