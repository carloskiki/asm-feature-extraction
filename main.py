from context import Context
from query import Query
from retrieval import Retrieval


def main():
    context = Context()

    if isinstance(context.command, Query):
        pass
    elif isinstance(context.command, Retrieval):
        command = context.command
        pool = command.generate_pool()
        print(pool)
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

def retrieval(context: Context, command: Retrieval):
    model = context.model()
    tokenizer = context.tokenizer()
    pool = command.generate_pool()

    queries = list(context.prompt(str(f)) for f, _ in pool)
    targets = list(context.prompt(str(f)) for f, _ in pool)
    query_tokens = tokenizer(queries, padding=True, return_tensors='pt').to('cuda')
    target_tokens = tokenizer(targets, padding=True, return_tensors='pt').to('cuda')

    model.generate(query_tokens, max_new_tokens=512).to('cuda')

    print("done")
