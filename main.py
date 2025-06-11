from query import Query
from retrieval import Retrieval
from arguments import arguments
from data_processing import Function, FileId

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
    import pickle
    with open("full-dataset.pickle", "rb") as file:
        print("reading file")
        data: list[tuple[Function, FileId]] = pickle.load(file)
        print("done reading")

        import matplotlib.pyplot as plt

        # Collect data for plotting
        lengths = []
        binaries = []
        for function, file in data:
            lengths.append(len(str(function)))
            binaries.append(file.binary)

        # Assign a unique color to each binary
        unique_binaries = list(set(binaries))
        binary_to_color = {binary: idx for idx, binary in enumerate(unique_binaries)}
        colors = [binary_to_color[b] for b in binaries]

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(binaries, lengths, c=colors, cmap='tab20', alpha=0.7)
        plt.xlabel("Binary")
        plt.ylabel("Function Length")
        plt.title("Function Lengths by Binary")
        plt.xticks(rotation=90)
        # Create legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=binary,
                              markerfacecolor=plt.cm.tab20(binary_to_color[binary] % 20), markersize=8)
                   for binary in unique_binaries]
        plt.legend(handles=handles, title="Binary", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("function_lengths_by_binary.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

    # arguments()()

if __name__ == "__main__":
    main()

