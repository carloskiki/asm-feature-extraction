from context import Context
from query import Query
from retrieval import Retrieval
from arguments import arguments



def retrieval(command: Retrieval):
    model = command.get_model()
    tokenizer = command.get_tokenizer()
    pool = command.generate_pool()

    queries = list(command.get_prompt(str(f)) for f, _ in pool)
    targets = list(command.get_prompt(str(f)) for f, _ in pool)
    query_tokens = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to('cuda')
    target_tokens = tokenizer(targets, padding=True, truncation=True, return_tensors='pt').to('cuda')
    print(query_tokens)

    model.generate(query_tokens['input_ids'], max_new_tokens=512).to('cuda')

    print("done")
    
def main():
    args = arguments()

    if isinstance(args, Query):
        pass
    elif isinstance(args, Retrieval):
        retrieval(args)
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

