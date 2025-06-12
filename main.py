import argparse
from query import Query
from bogus import Bogus
from retrieval import Retrieval

def query(command: Query):
    with open(command.assembly, "r", encoding="utf-8") as assembly_file:
        assembly = assembly_file.read()

    prompt = command.get_prompt(assembly)
    model = command.get_model()
    tokenizer = command.get_tokenizer()

    model_inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to("cuda")

    tokenized_output = model.generate(
        **model_inputs, max_new_tokens=2000, temperature=0.5
    )
    output = tokenizer.batch_decode(tokenized_output)[0]

    print(output)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", choices=["codeqwen"], type=str, default="codeqwen"
    )
    parser.add_argument("-p", "--prompt", type=str, default="base")
    subparsers = parser.add_subparsers(
        description="the action to be performed", dest="subcommand", required=True
    )
    Query.command(subparsers)
    Retrieval.command(subparsers)
    Bogus.command(subparsers)

    args = parser.parse_args()
    subcommand = args.subcommand
    delattr(args, "subcommand")
    if subcommand == 'query':
        return Query(**vars(args))
    if subcommand == 'retrieval':
        return Retrieval(**vars(args))
    if subcommand == 'bogus':
        return Bogus(**vars(args))

    raise ValueError("no subcommand provided")

if __name__ == "__main__":
    main()