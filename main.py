import torch
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

    print(torch.cuda.memory_reserved())
    query_tokens = tokenizer(queries[0], return_tensors='pt', max_length=32768, truncation=True, padding=True).to('cuda')
    # target_tokens = tokenizer(targets, padding=True, truncation=True, return_tensors='pt').to('cuda')
    print(torch.cuda.memory_reserved())
    output = model.generate(**query_tokens, max_new_tokens=512).to('cuda')
    
    print(torch.cuda.memory_reserved())

    print("done")

def query(command: Query):
    with open(command.assembly, "r") as assembly_file:
        assembly = assembly_file.read()

    query = command.get_prompt(assembly)
    model = command.get_model()
    tokenizer = command.get_tokenizer()

    model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    tokenized_output = model.generate(
        **model_inputs, max_new_tokens=2000, temperature=0.5
    )
    output = tokenizer.batch_decode(tokenized_output)[0]

    print(output)
    
def main():
    args = arguments()

    if isinstance(args, Query):
        query(args)
    elif isinstance(args, Retrieval):
        retrieval(args)
    else:
        raise ValueError("Unreachable branch")

if __name__ == "__main__":
    main()



