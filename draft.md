# TODOs

- [ ] Add references to the binaries in dataset.


### Background and Related Works

### Dataset

The dataset is composed of 7 binaries: busybox, coreutils, curl, image-magick, openssl, putty, and sqlite3.
All were compiled using gcc for the following platforms: x86_64, x86_32, arm, mips, powerpc.
For each binary and platform, binary objects were generated for each optimization level (O0 to O3).
Stripped all debug symbols except for function names, so as to be able to match functions across binaries.
In total, yeilds 140 different binaries to analyze.
The binaries were dissassembled using IDA Pro, yielding 383_658 assembly routines.