from data_processing import process
from context import Context


def main():
    context = Context()

    with open(context.command.assembly, "r") as assembly_file:
        assembly = assembly_file.read()

    query = context.prompt(assembly)
    model = context.model()
    tokenizer = context.tokenizer()

    model_inputs = query.token([query], return_tensors="pt").to("cuda")

    tokenized_output = model.generate(
        **model_inputs, max_new_tokens=2000, temperature=0.5
    )
    output = tokenizer.batch_decode(tokenized_output)[0]

    print(output)


if __name__ == "__main__":
    import gzip

    with gzip.open(
        "../lib-data/sqlite3-powerpc-linux-gnu-gcc-g-O3.bin.merged.asm.json.gz", "rb"
    ) as file:
        for index, function in enumerate(process(file.read())):
            pass
