from transformers import AutoModelForCausalLM, AutoTokenizer
from arguments import Arguments

MODELS = {
    "codeqwen": "Qwen/Qwen2.5-Coder-0.5B-Instruct"
}

def main():
    arguments = Arguments()

    model = AutoModelForCausalLM.from_pretrained(MODELS[arguments.model], torch_dtype="auto", device_map="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[arguments.model])

    query = arguments.format_prompt()

    model_inputs = tokenizer([query], return_tensors="pt").to("cuda")

    tokenized_output = model.generate(**model_inputs, max_new_tokens=200, temperature=0.5)
    output = tokenizer.batch_decode(tokenized_output)[0]

    print(output)

if __name__ == "__main__":
    main()