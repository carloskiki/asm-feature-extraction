"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from typing import Optional
import random
import sys
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_jaccard_index
from accelerate import Accelerator
from data_processing import (
    BINARIES,
    PLATFORMS,
    LibDataset,
    TargetDataset,
)
from context import Context, MAX_LENGTH, MAX_NEW_TOKENS


@dataclass
class Retrieval(Context):
    """
    CLI command to evaluate function retrieval
    """

    pool_size: Optional[int]
    seed: int  # Seed for selection of targets, choosed randomly if not set
    binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    platform: Optional[str]  # Run for a specific platform, run on all platforms if None
    optimization: Optional[
        int
    ]  # Run for a specific optimization, run on all optimizations if None
    batch_size: int  # Number of batches processed at once
    context_size: int  # Context window for the LLM
    data_path: str
    save_output: Optional[str]
    save_pool: Optional[str]

    target_platform: Optional[str]
    target_optimization: Optional[int]

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=str, choices=PLATFORMS.keys())
        parser.add_argument("--optimization", type=int, choices=range(4))
        parser.add_argument("--target-platform", type=str, choices=PLATFORMS.keys())
        parser.add_argument("--target-optimization", type=int, choices=range(4))
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--context-size", type=int, default=2048)
        parser.add_argument("--save-output", type=str)
        parser.add_argument("--save-pool", type=str)
        parser.add_argument("data_path", type=str)

    def __call__(self):
        accelerator = Accelerator()

        # No need to prepare the model, because we only do inference
        model = self.get_model(accelerator)
        tokenizer = self.get_tokenizer()

        dataset = LibDataset(
            self.data_path,
            accelerator.is_local_main_process,
            self.pool_size,
            self.seed,
            self.binary,
            self.optimization,
            self.platform,
        )
        pool_dataset = TargetDataset(
            dataset, self.target_optimization, self.target_platform
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        pool_loader = DataLoader(
            pool_dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )
        loader = accelerator.prepare_data_loader(loader, device_placement=False)
        pool_loader = accelerator.prepare_data_loader(loader, device_placement=False)

        tokenizer = self.get_tokenizer()

        query_decoded = []
        targets_decoded = []
        functions = []

        for batch in tqdm(
            loader,
            desc="Query batches",
            disable=not accelerator.is_local_main_process,
        ):
            # Tokenize the prompts for the batch
            prompts = [self.get_prompt(str(f)) for f in batch]
            functions.extend(batch)

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

            # Pass the tokens to LLM
            query_outputs = model.generate(
                **token_batch,
                max_new_tokens=MAX_NEW_TOKENS,
            )[:, token_batch["input_ids"].shape[1] :].cpu()
            decoded = tokenizer.batch_decode(query_outputs)

        for pool_batch in tqdm(
            pool_loader,
            desc="Target Batches",
            disable=not accelerator.is_local_main_process,
        ):
            # Tokenize the prompts for the batch
            prompts = [self.get_prompt(str(f)) for f in pool_batch]
            functions.extend(pool_batch)

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

            # Pass the tokens to LLM
            target_outputs = model.generate(
                **token_batch,
                max_new_tokens=MAX_NEW_TOKENS,
            )[:, token_batch["input_ids"].shape[1] :].cpu()
            decoded = tokenizer.batch_decode(target_outputs)

            # Add all outputs to targets_decoded
            targets_decoded.extend(decoded)

        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            return

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
            scores[index].append(jaccard_similarity(query, target, vocab_size))
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


def jaccard_similarity(query: Tensor, potential_target: Tensor, vocab_size: int):
    """
    Run Jaccard Similarity over the generated tokens of an LLM query
    """

    multiclass_jaccard_index(query, potential_target, vocab_size)
