"""
Retrieval CLI utilities
"""

import random
import sys
import gzip
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Generator
from data_processing import Function, function_count, process
import context

BINARIES = {
    "busybox": "busybox_unstripped",
    "coreutils": "coreutils",
    "curl": "curl",
    "image-magick": "magick",
    "openssl": "openssl",
    "putty": "puttygen",
    "sqlite3": "sqlite3",
}

PLATFORMS = {
    "arm": "arm-linux-gnueabihf-gcc",
    "gcc32": "gcc32",
    "gcc": "gcc",
    "mips": "mips-linux-gnu-gcc",
    "powerpc": "powerpc-linux-gnu-gcc",
}


class FileId:
    """
    A specific file in the dataset.
    """

    data_path: str
    binary: str
    platform: str
    optimization: int

    def from_values(self, data_path, binary, platform, optimization):
        """
        Create the FileId from the provided values
        """

        self.data_path = data_path
        self.binary = binary
        self.platform = platform
        self.optimization = optimization

    def path(self):
        """
        Return the file corresponding to the Id
        """

        return f"{self.data_path}/{BINARIES[self.binary]}-{PLATFORMS[self.platform]}-g-O{self.optimization}.bin.merged.asm.json.gz"


@dataclass
class Retrieval(context.Context):
    """
    CLI command to evaluate function retrieval
    """

    pool_size: int
    seed: int  # Seed for selection of targets, choosed randomly if not set
    pool_binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    pool_platform: Optional[
        str
    ]  # Run for a specific platform, run on all platforms if None
    pool_optimization: Optional[
        int
    ]  # Run for a specific optimization, run on all optimizations if None
    batch_size: int  # Number of batches processed at once
    context_size: int  # Context window for the LLM
    data_path: str

    # target_platform: Optional[str]
    # target_optimization: Optional[str]
    # same_binary: bool # Keep the same binary for the target pool
    # same_platform: bool # keep the
    # same_optimization: bool
    # from_pool: bool

    @staticmethod
    def command(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=1000)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--pool-binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--pool-platform", type=str, choices=PLATFORMS.keys())
        parser.add_argument("--pool-optimization", type=int, choices=range(4))
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--context-size", type=int, default=2048)
        parser.add_argument("data_path", type=str)

    def data_file(self) -> FileId:
        """
        Get the file selected with the CLI arguments, using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)
        file = FileId()

        file.data_path = self.data_path
        if self.pool_binary is None:
            file.binary = rng.choice(list(BINARIES.keys()))
        else:
            file.binary = self.pool_binary

        if self.pool_platform is None:
            file.platform = rng.choice(list(PLATFORMS.keys()))
        else:
            file.platform = self.pool_platform

        if self.pool_optimization is None:
            file.optimization = rng.randrange(4)
        else:
            file.optimization = self.pool_optimization

        return file

    def data_files(self) -> Generator[FileId, None, None]:
        """
        return all files that match the selected parameters
        """

        if self.pool_binary is None:
            binary = BINARIES.keys()
        else:
            binary = [self.pool_binary]
        if self.pool_platform is None:
            platform = PLATFORMS.keys()
        else:
            platform = [self.pool_platform]
        if self.pool_optimization is None:
            optimization = range(4)
        else:
            optimization = [self.pool_optimization]

        for b in binary:
            for p in platform:
                for o in optimization:
                    file = FileId()
                    file.from_values(self.data_path, b, p, o)
                    yield file

    def source_function(self) -> Function:
        """
        Return the function selected with the CLI using random values for the unspecified parameters
        """

        rng = random.Random(self.seed)

        with gzip.open(self.data_file().path()) as file:
            data = file.read()

        functions = process(data)
        if self.src_function is None:
            index = rng.randrange(function_count(data))
            return next(itertools.islice(functions, index, None))
        return next(f for f in process(data) if f.name == self.src_function)

    def generate_pool(self) -> list[tuple[Function, FileId]]:
        """
        Get the pool of targets
        """

        files = list(self.data_files())
        functions_per_file = self.pool_size // len(files)
        last_file_function_count = (
            self.pool_size - (len(files) - 1) * functions_per_file
        )

        pairs = []
        for index, file in enumerate(tqdm(files, desc="Reading dataset")):
            if index == len(files) - 1:
                sample_size = last_file_function_count
            else:
                sample_size = functions_per_file

            with gzip.open(file.path(), "rb") as file_data:
                functions = process(file_data.read())

            for sample in self.iter_sample(functions, sample_size):
                sample: Function = sample
                pairs.append((sample, file))

        return pairs

    def iter_sample(self, iterator, sample_size):
        """
        Sample from an iterator
        """

        rng = random.Random(self.seed)
        results = []
        # Fill in the first samplesize elements:
        try:
            for _ in range(sample_size):
                results.append(next(iterator))
        except StopIteration as exc:
            raise exc
        rng.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, sample_size):
            r = rng.randint(0, i)
            if r < sample_size:
                results[r] = v  # at a decreasing rate, replace random items
        return results


# def calculate_mrr(scores:np.ndarray, relevance:np.ndarray):
#     """
#     Calculate the Mean Reciprocal Rank (MRR) for a batch of data where each row contains relevance scores.
#
#     Args:
#     scores (np.ndarray): An array of shape (query_batch, values_batch) where the relevant item for each query is at the diagonal position (i, i).
#     relevance (np.ndarray): skip ith element if the ith masking is zero.
#
#     Returns:
#     float: The Mean Reciprocal Rank (MRR).
#     """
#     query_batch = scores.shape[0]
#
#     # Initialize a list to store ranks
#     ranks = []
#
#     for i in range(query_batch):
#
#         # Get the relevance scores for the i-th row
#         relevance_scores = scores[i]
#         relevant_score = relevance_scores[relevance[i]]
#
#         # Calculate the rank of the relevant score within the row
#         rank = (relevance_scores > relevant_score).sum().item() + 1
#
#         # Append the rank to the list
#         ranks.append(rank)
#
#     # Convert ranks to an array
#     ranks = np.array(ranks)
#
#     # Calculate the reciprocal ranks
#     reciprocal_ranks = 1 / ranks
#
#     # Compute the Mean Reciprocal Rank (MRR)
#     mrr = reciprocal_ranks.mean()
#
#     return float(mrr)
#
# def find_common_elements(*lists):
#     lists = lists[0]
#     # Find intersection of all lists
#     common_elements = set(lists[0])
#     for lst in lists[1:]:
#         common_elements &= set(lst)
#     return list(common_elements)
#
#
# def recall_at_k(scores: np.ndarray, relevance:np.ndarray, k: int) -> float:
#     """
#     Compute the recall@k for a scores matrix.
#
#     Parameters:
#     - scores: np.ndarray of shape (query_batch_size, values_batch_size)
#     - relevance: np.ndarray of shape (query_batch_size)
#     - k: int, the number of top items to consider
#
#     Returns:
#     - float, the average recall@k
#     """
#     # Get the batch size
#     query_batch = scores.shape[0]
#
#     # Get the indices of the top-k scores for each query
#     top_k_indices = np.argsort(-scores, axis=1)[:, :k]
#
#     # Initialize the recall@k counter
#     recall_at_k_count = 0
#
#     # Check if the relevant item (diagonal element) is among the top-k items
#     for i in range(query_batch):
#         if relevance[i] in top_k_indices[i]:
#             recall_at_k_count += 1
#
#     # Compute the average recall@k
#     recall_at_k = recall_at_k_count / query_batch
#
#     return recall_at_k
#
#
# def test_retrieval(query_embs, value_embs):
#     scores = []
#     for i in query_embs:
#         scores.append(F.cosine_similarity(i.unsqueeze(0), value_embs, dim=1).numpy())
#     scores = np.array(scores)
#
#     ## this takes too much memory for large pool size like 10k
#     # scores = F.cosine_similarity(query_embs.unsqueeze(1), value_embs.unsqueeze(0), dim=2).numpy()
#     relevance = np.arange(query_embs.size(0))
#     return compute_retrieval_metrics(scores, relevance)
#
# def compute_retrieval_metrics(scores, relevance):
#     pool_size = scores.shape[1]
#     if relevance is None:
#         relevance = np.arange(scores.shape[0])
#     mrr = calculate_mrr(scores, relevance)
#     recall_at_1 = recall_at_k(scores, relevance, 1)
#     recall_at_10 = recall_at_k(scores, relevance, 10)
#     return {
#         'mrr':mrr,
#         'recall_at_1':recall_at_1,
#         'recall_at_10':recall_at_10,
#         'pool_size': pool_size
#     }


def retrieval(command: Retrieval):
    # model = command.get_model()
    # tokenizer = command.get_tokenizer()
    pool = command.generate_pool()

    queries = list(command.get_prompt(str(f)) for f, _ in pool)
    # targets = list(command.get_prompt(str(f)) for f, _ in pool)
    print("Max query len:", max([len(q) for q in queries]))

    # query_tokens = tokenizer(queries[0], return_tensors='pt', truncation=True, padding=True).to('cuda')
    # # target_tokens = tokenizer(targets, padding=True, truncation=True, return_tensors='pt').to('cuda')
    # output = model.generate(**query_tokens, max_new_tokens=512).to('cuda')

    # print(torch.cuda.memory_reserved())

    print("done")
