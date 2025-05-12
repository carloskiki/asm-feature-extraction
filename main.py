# from transformers import AutoModelForCausalLM, AutoTokenizer
from arguments import Arguments
from data_processing import process

MODELS = {
    "codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"
}

def main():
    # arguments = Arguments()

    functions = process("C:/Users/cgagnon/ghidra-plugin/feature-extractor/data/openssl-gcc32-g-O0.bin.merged.asm.json")
    print(next(x for i,x in enumerate(functions) if i==150))

    # model = AutoModelForCausalLM.from_pretrained(MODELS[arguments.model], torch_dtype="auto", device_map="auto").to("cuda")
    # tokenizer = AutoTokenizer.from_pretrained(MODELS[arguments.model])

    # query = arguments.format_prompt()

    # model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    # tokenized_output = model.generate(**model_inputs, max_new_tokens=2000, temperature=0.5)
    # output = tokenizer.batch_decode(tokenized_output)[0]

    # print(output)

if __name__ == "__main__":
    main()