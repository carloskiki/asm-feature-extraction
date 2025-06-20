# TODOs

- [ ] Add references to the binaries in dataset.


## Background

Binary code similarity detection becomes more important as the modularity and rate of production of software grows.
Modern software is almost never written from scratch, and is becoming increasingly reliant on external libraries.
For reverse engineers, reducing the amount of repetitive assembly functions to analyze is important,
    as it allows them to be more efficient and focus on the custom parts of a binary.
If a vulnerability is found in a library, it is important to be able to quickly identify if an unknown or proprietary binary
    is using the vulnerable library, so that it can be mitigated.
Another aspect of BCSD is license compliance, which is important both for corporations and open source projects.

- Plagiarism Detection
## Related Work

## Methodology

### Dataset

The dataset is composed of 7 binaries: busybox, coreutils, curl, image-magick, openssl, putty, and sqlite3.
All were compiled using gcc for the following platforms: x86_64, x86_32, arm, mips, powerpc.
For each binary and platform, binary objects were generated for each optimization level (O0 to O3).
Stripped all debug symbols except for function names, so as to be able to match functions across binaries.
In total, yeilds 140 different binaries to analyze.
The binaries were dissassembled using IDA Pro, yielding 383_658 assembly routines.
