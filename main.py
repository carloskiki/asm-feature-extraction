from data_processing import process

def main():
    functions = process("C:/Users/cgagnon/ghidra-plugin/feature-extractor/data/openssl-gcc32-g-O0.bin.merged.asm.json")
    print(next(x for i,x in enumerate(functions) if i==150))


if __name__ == "__main__":
    main()