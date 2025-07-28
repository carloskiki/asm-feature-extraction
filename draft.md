## Intro

Modern software is increasingly reliant on external libraries.
For security researchers, detecting whether a binary uses some vulnerable library function is crucial to assess
and mitigate vulnerability exposure [1, 2]. For reverse engineers, reducing the amount of repetitive assembly functions
to analyze means being more productive, and allows them to focus on the custom parts of a binary.  Binary code similarity
detection (BCSD) is a binary analysis task that tries to solve these problems, by determining whether two compiled
code fragments behave in similar ways. This task is becoming increasingly important as the size, modularity and rate of
production of software grows.  For instance, when libraries are statically linked to a binary, BCSD can help to
quickly identify which functions are part of common external libraries and which ones have never been seen before.
Furthermore, if a vulnerability is found in a library, BCSD can be used to efficiently determine whether a proprietary
binary or firmware is using the vulnerable library.

Early approaches to BCSD used human defined heuristics to extract a "feature vector" from a binary function. These heuristics
could be calculated by statically examining the functions and its control flow graph (CFG), or could be measured at runtime
by executing the function in an emulated environment. These methods were deterministic and had the benefit of producing
human understandable feature vectors, but were often too simplistic or sometimes used computationally intractable alogrithms in
the case of CFG analysis.

More recently, machine learning (ML) based methods have shown to be better performing.
These methods work by producing an embedding for a binary function using techniques coming from natural language processing.
The generated embeddings are usually floating point tensors and serve as the "feature vector". These vectors are compared
with each other using metrics such as cosine similarity, and use approximate nearest neighbor search when filtering through
large databases.

Our work presents a method to effectively find assembly fragment clones across binaries using any available pre-trained
large language model (LLM). The method is simpler than other ML approaches, requires no training nor fine-tuning, and
matches state-of-the-art results. It has the advantage of generating human interpretable feature vectors like earlier approaches,
instead of numerical embeddings. Additionally, it effectively scales with the performance and size of the LLM used, and thus
benefits from the ample amount of research on in this area.

### Contributions

- We provide an elementary approach to BCSD purely based on the recent advancements in large language models (LLM) that
    is effective at cross-optimization and cross-architecture retrieval. This method requires no pre-training, and
    generates human interpretable feature vectors.
- We show that our approach scales with the performance and size of the LLM used, and out performs state-of-the-art BCSD models
    both in versatility and raw metrics.
- We develop a method to combine our model we any pre-existing or future assembly function embedding model, and show that
    this combination generates the best results.
- We outline a path for future research to improve the efficiency and performance of this method, and explain how this method
    could be implemented in production environments.

## Background

### Binary Analysis

Security researchers and reverse engineers are routinely tasked with the analysis of unknown or proprietary executables.
Reverse engineers try to analyze the binary to understand its underlying algorithms, while security researchers want to assess
the risk associated with potential vulnerabilities found within the executable. This process is usually conducted using
a software reverse engineering platform such as Ghidra [24] or IDA Pro [23]. The main functions of these programs are to
disassemble and decompile a provided binary, so that its content can be analyzed by humans. Disassembly is
the process of retrieving the human-readable assembly instructions from the binary executable, whereas decompilation
is the process of generating higher-level pseudo-code from the instructions based on common patterns and heuristics.

Binary analysis is a hard task because once a program is compiled, most of the information contained in its source code
is lost [1]. Variables, data structures, functions, and comments are removed, because the compiler's task is to make
the program as efficient as possible - which often means removing as much as possible. The optimizers within the compiler
only have a single rule: They must not makes changes to the observable behavior of the program. This is ofter refered
to as the "as-if rule". As a result, compilers can remove, reorder, and inline significant parts of the code. Even worse,
adverserial programs, such as malware or digital rights management software uses obfuscation techniques to resist
having their code reverse engineered.

### Binary code similarity detection

BCSD is the task of determining whether two fragments of binary code perform similar actions.
These fragments are usually first disassembled, and are then compared for similarity. In practice,
similarity detection is performed with one known fragment (either because it was analyzed before
or because its source code is known), and one unknown fragment. If the unknown piece of code is deemed
highly similar to the known one, it greatly simplifies the analysis task, and reduces duplicate work. Known
code fragments are typically collected in a database which is queried against for clone search. For example,
if a major vulnerability in a widely used, core open-source component is found, BCSD can be used to quickly
assess if a binary contains the vulnerable code fragment. It can also be used for plagiarism detection, which
could take the form of patent infringemnt, or for malware classification. Challenges in binary analysis are
amplified in binary code similarity detection. Two code fragments that seem very different can still have
the same observable behavior.

Recent research uses deep learning to generate a vector embedding for each assembly function [11, 14, 16, 17, 19].
Generally, training a model to perform such task requires a large training dataset, and highly performant GPUs
to perform lots of computations. Once trained however, these methods can generate excellent results.
State of the art implementations are limited by their training data, and poorly generalize to out-of-domain tasks,
such as assembly code for a different architecture, or code compiled with a different compiler [1, 19]. Most methods also require
a pre-processing step after disassembly, such as extracting the control flow graph [16, 17], or slightly modifying the input
assembly in a specific manner [14, 19].

Production scale BCSD engines contain millions of assembly functions. With one vector embedding per function, nearest neighbor
search takes a significant amount of time as the algorithm has to linearly compare the query with every function in the database.
To alliviate this issue, approximate nearest neighbor search is used on large databases to reduce the time complexity of the
search [21, 22], but it may not always return the first match and thus skews similarity scores.

### Problem Definition

We deem two assembly code fragements to be semantically identical if they have the same observable behavior on a system.
This type of clone is generally referred to as "type four" clones [17, 26]. This type of clone excludes different algorithms that
might happen to have the same output, such as breadth-first search vs. depth-first search. In research it is common
to use the same code source code function compiled with different compilers or compilation options to create such
type four clones.

## Related Work

### Static Analysis

Traditional methods make use of static analysis to detect clone assembly routines. With these methods, a trade-off has
to be made between the robustness to obfuscation and architecture differences, and the performance of the algorithm. [1]
Control flow graph analysis and comparison [3, 4] is known to be robust to syntactic differences, but often involves computationally intractable
problems. Other algorithms that use heuristics such as instruction frequency, longest-common-subsequence, or locality sensitive hashes
[5, 6, 7] are less time consuming, but tend to fixate on the syntactic elements and their ordering rather than the semantics.

### Dynamic Analysis

Dynamic analysis consists of analyzing the features of a binary or code fragment by monitoring its runtime behavior. For BCSD
this method is compute intensive and requires a cross-platform emulator, but completely sidesteps the syntactic aspects of binary code
and solely analyzes its semantics. [2] As such, this method is highly resilient to obfuscations, but requires a sandboxed environment
and is hard to generalize across architectures and application binary interfaces [27].

### Machine Learning Methods

The surge of interest and applications for machine learning in recent years has also affected BCSD.
Most state-of-the-art methods use natural language processing (NLP) to achieve their results [refs].
Notably, recent machine learning approaches try to incorporate the transformer architecture into BCSD tasks [refs].

Asm2Vec [17] is one of the first methods to use a NLP based approach to tackle the BCSD problem. It interprets an
assembly function as a set of instruction sequences, where each instruction sequence is a possible execution path
of the function. It samples these sequences by randomly traversing the CFG of the assembly function, and then
uses a technique based on the PV-DM model [18] to generate an embedding for the assembly function. This solution
is not cross architecture compatible and requires pre-training.

SAFE [11] uses a process similar to Asm2Vec. It first encodes each instruction of an assembly function into a vector,
using the word2vec model [12]. Using a Self-Attentive Neural Network [13], SAFE then converts the sequence of instruction
vectors from the assembly function into a single vector embedding for the function. Much like Asm2Vec, SAFE requires
pre-training, but can perform cross architecture similarity detection. This method supports both cross optimization
and cross architecture retrieval, but was only trained on the `AMD64` and `arm` platforms.

Order Matters [16] applies a BERT language reprensentation model [15] along with control flow graph analysis
to perform BCSD. It uses BERT to learn the embeddings of instructions and basic blocks from the function,
passes the CFG through a graph neural network to obtain a graph semantic embedding, and sends the adjacency
matrix of the CFG through a convolutional neural network to compute a graph order embedding. These embeddings
are then combined using a multi-layer perceptron, obtaining the assembly function's embedding. This method
supports cross architecture and cross platform tasks, although its implementation is only trained on `x86_64`
and `arm` for cross platform retrieval.

A more recent BCSD model, PalmTree [14], also bases its work on the BERT model [15].
It considers each instruction as a sentence, and decomposes it into basic tokens. The model is trained
on three tasks. 1. As is common for BERT models, PalmTree is trained on masked language modeling. 2.
PalmTree is trained on context window prediction, that is predicting whether two instructions are found
in the same context window of an assembly function. 3. The model is also trained on Def-Use Prediction -
predicting whether there is a definition-usage relation between both instructions. This method's
reference implementation is only trained on cross compiler similarity detection, but can be trained
for other tasks.

The model CLAP [19] uses the RoBERTa [20] base model, to perform assembly function encoding.
It is adapted for assembly instruction tokenization, and directly generates an embedding for
a whole assembly function. It is also accompanied with a text encoder (CLAP-text), so that classification can
be performed using human readable classes. Categories or labels are encoded with the text encoder, and the
assembly function with the assembly encoder. The generated embeddings can be compared with cosine similarity to
calculate whether the assembly function is closely related or opposed to the category. This model requires
training and is architecture specific (`x86_64` compiled with GCC).

## Method

Our method is designed to solve some of the pain points associated with state of the art deep learning models for BCSD.
An important factor is that our method extracts human understandable features from an assembly function, instead
of a vector embedding. This unique advantage allows immediate human verification when a match is detect by the similarity
engine. A database of assembly functions and their interpretable features is more easily maintained, as defects can
be patched by humans, rather than having to regenerate the whole database when the model is modified.

By using any open-source or commercially available LLM, we entirely sidestep model training by leveraging the extenive
and diverse datasets that LLMs are pre-trained on.  Our method can be tuned by modifying the instructions provided to
the LLM, which is significantly simpler than having to retrain the model and regenerate embeddings for the whole database.
The underlying LLM can also replaced seamlessly without invalidating the database, meaning that our method will continue
to scale with the performance improvements of LLMs - an area which is showing impressive growth and development. Furthermore, if
a section of the prompt was edited to modify the output feature set, the database can still be maintained without
having to regenerate a new feature set. Default values or values derived from other fields in the feature set can be added
as is standard with database migrations.

Another key advantage of our method also stems from its textual representation of the extracted feature set. As highlighted
previously, vector embeddings are computationally expensive to match against in large databases. Textual search is much
more scalable than nearest neighbor search, as is evident in modern search engines being able to filter through billions of documents
in a fraction of a second.

### Prompt

The method consists of querying a large language model with a prompt crafted to extract the high-level behavioral features of
the provided assembly code fragment. The assembly code fragment does not require preprocessing. As output, the LLM generates a JSON
structure containing the extracted features. We outline the prompt made up of multiple parts, each designed to
extract specific semantic information from the assembly function. The full prompt is open-source and avaliable on Github.

#### Framing and conditioning

We use a prelude that contains general information about the task at hand, and the expected response.
Recent commercial LLMs have the ability to generate responses following a specified JSON schema. We do not
make use of this capability when evaluating commercial models so that the results can be compared to local LLMs,
that do not benefit from this option.

LLMs, especially smaller models will sometimes generate nonsensical output. In our early experiments, the smallest local model
evaluated (0.5B parameters) would sometimes repeat the same line of JSON until it ran out of context space, or generated invalid JSON.
To combat these artifacts, we run inference again when JSON parsing fails and increase the temperature of output token
selection. This is done in a loop until valid JSON is generated. In our experiments, the maximal amount of trials required
for any query to generate valid json was three, but 95% of the generated responses from the smallest model would constitute
valid JSON.

> You are an expert assembly code analyst, specializing in high-level semantic description and feature extraction for comparative
analysis. Your goal is to analyze an assembly routine from an **unspecified architecture and compiler** and provide its extracted
high-level features, formatted as a JSON object. For the provided assembly routine, extract the following features and infer the
algorithm. Your output **MUST** be a JSON object conforming to the structure defined by these features.

#### Type Signature

The first feature category extracted is the type signature of the provided assembly function.
We only make the distinction between two types: Integer and Pointer. Inferring more information than these two 
primitive types has shown to be too complicated for current LLMs. We extract the number of input arguments
and their types, and also extract the return value type, if any.

#### Core logic & Operations

This section specifies what to extract from the function in terms of its logical behavior, and how to determine the kind of
operation that the assembly function performs. We list some of the operations extracted here.

- Indication of loops. This is determined by the presence of jump instructions that point back to a
    previous instruction after some conditions have been checked.
- Indication of jump tables. Evaluated by patterns suggesting calculated jump addresses based on
    indices, or a series of conditional jumps.
- Extensive use of indexed addressing modes.
- Use of SIMD instructions and registers.
- Number of distinct subroutine call targets.
- Overall logical behavior. Possibilities include: Arithmetic operations, Bitwise operations, Data movement and memory access,
  Control flow and dispatching operations, Memory access operations.

#### Notable constants

This section identifies notable integer and floating point constants.  These could be common scalar values used by
a specific cryptographic algorithm, or the signature bytes used by a file format or protocol. We exclude small
values are used as struct offsets, loop counters or stack pointer adjustments.

#### Side effects

The prompt also monitors the side effects that the assembly function has on the system.
Modification of input arguments is identified when a pointer input is used to write to memory, and modification
of global states is detected similarily, when writes to absolute memory addresses or addresses resolved via global
data segment pointers occur. Memory allocation and deallocation is determined by the presence of calls to memory
management functions like `malloc` or `free`. Linear memory access patterns are detected by the presence of sequential
indexed memory accesses inside loops or across multiple instructions. Finally, system calls and software interrupts are
identified by the presence of specific instructions that trigger them.

#### Final categorization

The last section tries to assign a overall category to the assembly function, by basing it on the information
collected in the analysis. The final categorization only weakly supports the similarity search because it does
not have a large impact on the similarity score. Its purpose is to provide a concise overview for reviewers
of the analysis, who might want to understand the function or verify its similarity with the target.
Categories include: cryptographic, data processing, control flow and dispatch, initialization, error handling,
wrapper/utility, file management, etc.

#### Examples

To achieve the best results with our method, we utilize few-shot prompting by providing hand crafted examples along with our prompt.
In all of our evaluations, three examples are provided, and our ablation study confirms that adding more than three examples provides
little to no benefit. Our examples are selected from a diverse source of functions, and are varied in size and architecture to
exemplify the space of possibilities in our evaluations.

The prompt by itself is still very performant, and should be acceptable for most applications. A surpising effect of providing examples
is that the prompt is no longer needed for the analysis to be effective. Our results show that using enough examples with an empty system promt
generates the same results as a standalone system prompt without examples.

### Comparison

ML based methods that generate an embedding for each assembly function generally compare these vectors using numerical
methods such as cosine similarity.  Since our generated analysis is not numerical, we use an alternative
method to compare two assembly functions.  We flatten the JSON structure into a dictionary, where booleans, numbers, and
strings are the elements, and the concatenated path to those elements is the key.  Jaccard similarity (Intesection over union)
is used to obtain a similarity measure.

### Dataset

The dataset is composed of 7 binaries: busybox, coreutils, curl, image-magick, openssl, putty, and sqlite3.
All were compiled using gcc for the following platforms: x86_64, x86_32, arm, mips, powerpc.
For each binary and platform, binary objects were generated for all optimization levels (O0 to O3),
stripped of debug symbols. In total, this yeilds 140 different binaries to analyze.
The binaries were dissassembled using IDA Pro, yielding 383_658 assembly functions.
Functions consisting of less than 3 instructions were not included as part of the dataset.
Pairs of equivalent functions from the same platform but distinct optimization level were made for cross optimization
evaluation, and pairs from the same optimization level but different platform were formed for cross
platform evaluation.

TODO: Table with function count.

### Model

We evaluate both local models of various sizes and commercially deployed models. Qwen2.5-Coder [ref] with sizes 0.5B to 7B
is used to run most local evaluations as its small size fitted our GPU capacity.
- Qwen3 [ref] with sizes 0.6 to 4B
- Gemma-3n [ref] with sizes 0.5B to 4B

Most evaluations and tests were run using Qwen2.5-Coder, and we use this model as a baseline.
On all local models, the input context size was limited to 4096 tokens, and output tokens generation to 512.
Large assembly functions that did not fit within the input tokens were truncated.

For ablation study, the following commercial models were evaluated on our dataset.

- OpenAI's GPT-4.1-mini [ref]
- OpenAI's o4-mini [ref]
- Google's gemini-2.5-flash-lite [ref]

### Evaluation method

The mean reciprocal rank (MRR) and first position recall (Recall@1) metrics are used for evaluation and comparison to other methods.
A pool of assembly function pairs is used for evaluation, where both assembly fragments in a pair come from the same source function.
For each pair, we compare the generated features for the first element of the pair with all second elements of the pairs contained
in the pool.  For example, consider a pool of ten pairs \((a_i, b_i)\) for \(i \in [1, 10]\), where \(a_i\) is compiled for the arm
architecture with optimization level 3, and \(b_i\) is compiled for the mips architecture with optimization
level 3. The features collected for the \(a_1\) is compared for similarity with the features of \(b_i\) for \(i \in [1, 10]\).
A ranking is generated by ordering these comparisons from most to least similar. Recall@1 is successful
if \(b_1\) is ranked first, and the reciprocal rank is \(\frac{1}{\text{rank}(b_1)}\).

## Experiments

Our experiments are run on a virtual machine with 8 (missing cpu spec) cpu cores, 100 GB of RAM, and four NVIDIA Quadro RTX
6000 GPUs each having  24GB of RAM. First, we compare our method against other state-of-the-art NLP based
approaches to BCSD on our dataset. Second, we run ablation studies on our method, to determine how the size
of the model, the number of examples provided, and the different sections of the prompt contribute to our results.
Third, we show that embedding models derived from LLMs that are _not_ trained on BCSD are on par with BCSD specific
embedding models. Finally, we show that the features extracted from our novel method are not properly represented
in state-of-the-art embedding methods, and that by combining our method with an embedding model yield significantly
better results that state of the art approaches.

### Clone Search with Different Optimization Levels

This experiment benchmarks the capability of the baselines and our method for detection of similar code fragments across
different optimization levels. We use the MRR (mean recprocal rank) and Recall@1 metrics, which are used by other methods [refs].
The hardest retrieval tasks is between optimization levels 0 and 3 [T1], because there is a substantial difference
between code compiled with `-O0` and code compiled with `-O3` [FIGURE of assembly fns]. At optimization level 0,
functions perform a lot of unessecary actions such as extensively moving data between registers and perform conditional evaluation
on operations that are tautologies. The generated code mostly is mostly left untouched by the optimizer.
At optimization level 3, the compiler will inline simple functions into the body of the caller
function, meaning jumps and calls to other places in the binary are replaced by the destination's instructions.
Loops are unrolled, so that each iteration of the loop is laid out sequentially, instead of performing a conditional check and a jump
to the loop's initial instruction. Also, instructions can be heavily reordered to achieve best performance, while keeping the observable
behavior of the program untouched.

The baselines are SAFE, OrderMatters, PalmTree, Asm2Vec, and CLAP. A summary of each of their architecture is provided in the Related
Works section. We also include Qwen3-Embedding 4B, an embedding model based on Qwen3 trained for general text embedding as baseline.
We present the results of our method evaluated on both a local model and a commercially deployed model.
Qwen2.5-Coder 7B Parameters is used as the local model, and Gemini 2.5 flash is used as the commercial model.

The baselines mostly perform worse than expected on this evaluation. As our own dataset is used
and not one that was presented by the baselines, we believe this may be caused by overfitting in the training process of these models.
As evident here, one of our method's advantage is that it requires no fine-tuning to acheive good results, and thus should generalize
well to unseen settings. Qwen3-Embedding also generates impressive results, given that it was not trained for assembly clone detection.

T1 desc: Evaluation of the baselines and our method on cross optimization retrieval with a pool size of 1000.
All functions are compiled for the arm architecture using gcc with the optimization levels specified for each column.
Our method uses the gemini-2.5-flash model for the best trade-off between efficiency and performance.

### Clone Search with Different Architectures

Different CPU architectures have varying assembly code languages. It is hard for BCSD methods that analyze assembly code to support
multiple architectures. These methods need to accurately represent two functions with completely different syntaxes but with identical
semantics as being very similar in terms of their feature vector or embedding. Hence, methods that use CFG analysis have a better chance
at supporting many architectures, since the structure of the CFG itself is architecture agnostic. However, the basic blocks that constitute
this graph are still in assembly code, which does not fully resolve the issue. Furthermore, there exists many different variants of each
instruction set, because each new version of an architecture brings new instructions to understand and support. With deep learning methods,
this means training or fine-tuning the model to understand this new language. Afterwards, all embeddings in a BCSD database need to
be regenerated. Our method does not directly address this issue, but brings a significant improvement. It indirectly makes use of the vast
amount of data used to train foundational LLMs. Since a LLM has extensively seen all of the mainstream CPU architectures and their dialects
in its training data, it is able to grasp their meaning and extract features from them. If the model in use seems to poorly comprehend a
specific architecture, it can be replaced with one that better performs the specific platform without invalidating the BCSD database.

Our method surpasses the baselines, but more work in this area is clearly still needed. The recall@1 metrics show that the best method
is able to rank the the correct assembly fragment in first place only 42% of the time on average.

T2 desc: Evaluation of the baselines and our method on cross architecture retrieval with a pool size of 1000.
All functions are compiled with optimization level 2 using gcc with the architecture specified for each column.
The same baselines and models are used as in the cross architecture evaluation.

### Ablation on model size

The results on our method show that there is a clear correlation between model size and BCSD retrieval performance. LLMs with less
than 3B parameters don't seem to comprehend the task when they are not provided with any examples. When provided with examples, these
small models will mimic the example provided without basing the output on the assembly function in the query. ...

### Ablation on examples

By providing hand crafted exmaples to the large language model, we are able to increase the performance of the assembly function analysis.
This follows the general observations behind few show prompting [25]. Providing a single example significantly increases the retrieval
scores, but providing more than one provides very limited increases in scores, especially for larger models. For smaller models,
Providing more examples increases the retrieval scores.

### Comparison of using different models

- Ablation on the number of examples provided.
- Ablation on the model size.
- Ablation on the prompt used, examples of output.
- Show results on commercial models
- Token usage on commercial models
- Combination with embedding models stuff

### Human interpretability

Our method offers a distinct advantage over the current state-of-the-art, because the feature vectors generated are human interpretable.
Other machine learning based methods generate numerical vector embeddings and compare the vectors using numerical methods such as cosine similarity.

## Future Research

These results open avenues for further investigation in LLM based BCSD, and more broadly in LLM assisted reverse engineering.
A few of these oportunities are outlined here.

### Distillation & Fine Tuning

It was shown that commercial LLMs perform better than smaller local models. Using the process of distillation [ref] on smaller models
will likely cause large gains for these models, approaching the performance of foundational LLMs. Fine Tuning is another opportunity
that should be explored to determine whether it provides meaningful gains in performance. One method would be using open source software
as fine tuning data. One could use source code and generate a feature vector (either manually or by prompting a large language model),
and then use this analysis as baseline to fine tune the analysis of the compiled assembly code.

### Comparison as opposed to function embedding generation

TODO

### LLM Reasoning

Some state of the art LLMs provide reasoning capabilities by having the model generate a chain of thought [8] it its response.
It has been shown that this type of output produces more accurate and higher quality responses [8, 9, 10]. We believe that similar
gains could be seen in assembly function analysis, at the cost of more output token generation and thus more compute.

## Conclusion

# Figures, images & graphs

- Diff example for generated json output vs. diff for assembly function.
- 


# Refs

1. [A Survey of Binary Code Similarity Detection Techniques](https://www.mdpi.com/2079-9292/13/9/1715)
2. [Binary Code Similiarity Detection](https://ieeexplore.ieee.org/document/9678518)
3. [Scalable Graph-based Bug Search for Firmware Images](https://dl.acm.org/doi/10.1145/2976749.2978370)
4. [Graph-based comparison of Executable Objects](https://www.semanticscholar.org/paper/Graph-based-comparison-of-Executable-Objects-Dullien/7661d4110ef24dea74190f4af69bd206d6253db9)

5. [Detecting Clones Across Microsoft .NET Programming Languages](https://ieeexplore.ieee.org/document/6385136)
6. [Idea: Opcode-Sequence-Based Malware Detection](https://link.springer.com/chapter/10.1007/978-3-642-11747-3_3)
7. [Binary Function Clustering Using Semantic Hashes](https://ieeexplore.ieee.org/document/6406693)

8. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
9. [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)
10. [THINKING LLMS: GENERAL INSTRUCTION FOLLOWING WITH THOUGHT GENERATION](https://arxiv.org/pdf/2410.10630)

11. [SAFE: Self-Attentive Function Embeddings for Binary Similarity](https://arxiv.org/pdf/1811.05296)
12. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)
13. [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130)

14. [PalmTree: Learning an Assembly Language Model for Instruction Embedding](https://arxiv.org/pdf/2103.03809)
15. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

16. [Order Matters: Semantic-Aware Neural Networks for Binary Code Similarity Detection](https://cdn.aaai.org/ojs/5466/5466-13-8691-1-10-20200511.pdf)

17. [Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://ieeexplore.ieee.org/document/8835340)
18. [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053)

19. [CLAP: Learning Transferable Binary Code Representations with Natural Language Supervision](https://dl.acm.org/doi/pdf/10.1145/3650212.3652145)
20. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

21. [Approximate Nearest Neighbor Search in High Dimensions](https://arxiv.org/pdf/1806.09823)
22. [Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations](https://arxiv.org/abs/2310.19126)

23. [IDA Pro A powerful disassembler, decompiler and a versatile debugger. In one tool.](https://hex-rays.com/ida-pro)
24. [Ghidra Software Reverse Engineering Framework](https://github.com/NationalSecurityAgency/ghidra)

25. [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)

26. [BinClone: Detecting Code Clones in Malware](https://ieeexplore.ieee.org/document/6895418)

27. [Blanket execution: Dynamic similarity testing for program binaries and components](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/egele)
28. [Kam1n0: MapReduce-based Assembly Clone Search for Reverse Engineering](https://dl.acm.org/doi/pdf/10.1145/2939672.2939719)

- [BinSim: Trace-based Semantic Binary Diffing via System Call Sliced Segment Equivalence Checking](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-ming.pdf)