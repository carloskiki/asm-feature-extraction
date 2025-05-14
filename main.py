from context import Context
from query import Query
from retrieval import Retrieval


def main():
    context = Context()

    if isinstance(context.command, Query):
        pass
    elif isinstance(context.command, Retrieval):
        command = context.command
        print(command.all_files())
    else:
        raise ValueError("Unreachable branch")

if __name__ == "__main__":
    main()


# with open(context.command.assembly, "r") as assembly_file:
#     assembly = assembly_file.read()

# query = context.prompt(assembly)
# model = context.model()
# tokenizer = context.tokenizer()

# model_inputs = query.token([query], return_tensors="pt").to("cuda")

# tokenized_output = model.generate(
#     **model_inputs, max_new_tokens=2000, temperature=0.5
# )
# output = tokenizer.batch_decode(tokenized_output)[0]

# print(output)