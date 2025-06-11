from query import Query
from retrieval import Retrieval
from arguments import arguments

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
    arguments()()

if __name__ == "__main__":
    main()

