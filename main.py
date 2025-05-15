import numpy as np
import torch
from context import Context
from query import Query
from retrieval import Retrieval, retrieval
from arguments import arguments

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
    context = Context("codeqwen", "base")
    tokenizer = context.get_tokenizer()
    model = context.get_model()
    
    message = "this is a message"
    tokens = tokenizer(message, padding='max_length', max_length=4096, return_tensors="pt").to('cuda')
    del tokens['attention_mask']
    print(tokens['input_ids'].size())
    
    output_tokens = model.generate(**tokens, max_new_tokens=512)
    print(output_tokens)
    output = tokenizer.batch_decode(output_tokens)[0]
    print(output)

if __name__ == "__main__":
    main()

