"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from typing import Optional
import random
import sys
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from data_processing import Function, BINARIES, PLATFORMS, LibDataset, FileId
import context
import jaccard

MAX_NEW_TOKENS = 1024
MAX_LENGTH = 16384


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
    save_output: Optional[str]
    save_pool: Optional[str]

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
        parser.add_argument("--save-output", type=str)
        parser.add_argument("--save-pool", type=str)
        parser.add_argument("data_path", type=str)

    def cache(self, data: list[tuple[Function, FileId]]):
        if self.save_pool is None:
            return

        with open(
            "{self.cache_pool}/{self.binary}-{self.platform}-{self.opt}-{self.seed}.pkl",
            "wb",
        ) as file:
            file.write(pickle.dumps(data))

    def __call__(self):
        accelerator = Accelerator()

        # No need to prepare the model, because we only do inference
        model = self.get_model(accelerator)
        tokenizer = self.get_tokenizer()

        dataset = LibDataset(
            self.data_path,
            self,
            accelerator.is_local_main_process,
            self.pool_size,
            self.seed,
            self.pool_binary,
            self.pool_optimization,
            self.pool_platform,
        )
        if accelerator.is_local_main_process:
            self.cache(dataset.data)

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        loader = accelerator.prepare_data_loader(loader, device_placement=False)

        tokenizer = self.get_tokenizer()

        query_vectors = []
        batches = []

        for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
            prompts = []
            batches.append(batch)
            for f, _ in batch:
                prompts.append(self.get_prompt(str(f)))

            chat = tokenizer.apply_chat_template(
                prompts, tokenize=False, add_generation_prompt=True
            )
            token_batch = tokenizer(
                chat,
                truncation=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                max_length=MAX_LENGTH,
            ).to(accelerator.device)

            query_outputs = model.generate(
                **token_batch,
                max_new_tokens=MAX_NEW_TOKENS,
            )[:, batch["input_ids"].shape[1] :]
            query_vectors.append(query_outputs)

        if self.save_output is not None:
            # Clear out and or create the file for all processes to write to it later
            if accelerator.is_local_main_process:
                with open(self.save_output, "w", encoding="utf-8") as file:
                    pass

            accelerator.wait_for_everyone()

            with open(self.save_output, "a", encoding="utf-8") as file:
                for outputs, batch in zip(query_vectors, batches):
                    outputs = tokenizer.batch_decode(outputs)
                    for output, function, file_id in zip(outputs, batch):
                        file.write("############\n")
                        file.write(f"Binary: {file_id.binary}\n")
                        file.write(f"Platform: {file_id.platform}\n")
                        file.write(f"Optimization: {file_id.optimization}\n")
                        file.write("```assembly\n")
                        file.write(str(function))
                        file.write("\n```\n")
                        file.write("Output:")
                        file.write(output)
                        file.write("\n")

        # query_vectors = torch.cat(query_vectors, dim=0).view(-1, query_vectors[0].size(-1)).cpu().float()
        # target_vectors = torch.cat(target_vectors, dim=0).view(-1, target_vectors[0].size(-1)).cpu().float()
        # metrics = test_retrieval(query_vectors, target_vectors)
        # print(metrics)

        # print("done")


def calculate_mrr(scores: np.ndarray, relevance: np.ndarray) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a batch of data where each row contains relevance scores.

    Args:
    scores (np.ndarray): An array of shape (query_batch, values_batch) where the relevant item for each query is at the diagonal position (i, i).
    relevance (np.ndarray): skip ith element if the ith masking is zero.

    Returns:
    float: The Mean Reciprocal Rank (MRR).
    """
    query_batch = scores.shape[0]

    # Initialize a list to store ranks
    ranks = []

    for i in range(query_batch):
        # Get the relevance scores for the i-th row
        relevance_scores = scores[i]
        relevant_score = relevance_scores[relevance[i]]

        # Calculate the rank of the relevant score within the row
        rank = (relevance_scores > relevant_score).sum().item() + 1

        # Append the rank to the list
        ranks.append(rank)

    # Convert ranks to an array
    ranks = np.array(ranks)

    # Calculate the reciprocal ranks
    reciprocal_ranks = 1 / ranks

    # Compute the Mean Reciprocal Rank (MRR)
    mrr = reciprocal_ranks.mean()

    return float(mrr)


def recall_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """
    Compute the recall@k for a scores matrix.

    Parameters:
    - scores: np.ndarray of shape (query_batch_size, values_batch_size)
    - relevance: np.ndarray of shape (query_batch_size)
    - k: int, the number of top items to consider

    Returns:
    - float, the average recall@k
    """
    # Get the batch size
    query_batch = scores.shape[0]

    # Get the indices of the top-k scores for each query
    top_k_indices = np.argsort(-scores, axis=1)[:, :k]

    # Initialize the recall@k counter
    recall_at_k_count = 0

    # Check if the relevant item (diagonal element) is among the top-k items
    for i in range(query_batch):
        if relevance[i] in top_k_indices[i]:
            recall_at_k_count += 1

    # Compute the average recall@k
    return recall_at_k_count / query_batch


def test_retrieval(query_tokens, target_tokens, vocab_size):
    """
    Tests the retrieval of each query against the pool of candidates (values).

    # Arguments
    query_tokens: 2D Tensor containing an embedding for each candidate.
    target_tokens: 2D Tensor containing an embedding for each candidate.
    """

    scores: list[list[float]] = []
    for index, query in enumerate(query_tokens):
        scores.append([])
        for target in target_tokens:
            scores[index].append(jaccard.similarity(query, target, vocab_size))
    scores = np.array(scores)

    ## this takes too much memory for large pool size like 10k
    # scores = F.cosine_similarity(query_embs.unsqueeze(1), value_embs.unsqueeze(0), dim=2).numpy()
    relevance = np.arange(query_tokens.size(0))
    return compute_retrieval_metrics(scores, relevance)


def compute_retrieval_metrics(scores, relevance):
    if relevance is None:
        relevance = np.arange(scores.shape[0])
    mrr = calculate_mrr(scores, relevance)
    recall_at_1 = recall_at_k(scores, relevance, 1)
    recall_at_10 = recall_at_k(scores, relevance, 10)
    return {
        "mrr": mrr,
        "recall_at_1": recall_at_1,
        "recall_at_10": recall_at_10,
    }
