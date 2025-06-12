from dataclasses import dataclass
import context

@dataclass
class Query(context.Context):
    assembly: str  # File containing the assembly to analyze

    @staticmethod
    def command(subparsers):
        parser = subparsers.add_parser(
            "query", description="Query an LLM for the features of an assembly function"
        )
        parser.add_argument("assembly", metavar="ASM-FILE", type=str)
    
    def __call__(self):
        with open(self.assembly, "r", encoding="utf-8") as assembly_file:
            assembly = assembly_file.read()

        prompt = self.get_prompt(assembly)
        model = self.get_model()
        tokenizer = self.get_tokenizer()

        model_inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to("cuda")

        tokenized_output = model.generate(
            **model_inputs, max_new_tokens=2000, temperature=0.5
        )
        output = tokenizer.batch_decode(tokenized_output)[0]

        print(output)