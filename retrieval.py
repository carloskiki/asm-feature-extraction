"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union
from statistics import median, mean
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import random
import json
import sys
import gc
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from data_processing import (
    BINARIES,
    PairsDataset,
)
from context import Context, MAX_NEW_TOKENS

CLEAR_CACHE_PERIOD = 32


def platform_parser(s):
    # If input looks like key:value,key:value,...
    if ":" in s and "," in s:
        try:
            return [tuple(p.split(":", 1)) for p in s.split(",")]
        except ValueError as e:
            raise ArgumentTypeError("Malformed key:value pair.") from e
    else:
        # Just treat it as a plain string
        return s


def optimization_parser(s):
    # Try to parse as a single int
    try:
        return int(s)
    except ValueError:
        pass  # Not a single int

    # Try to parse as list of int:int pairs
    try:
        pairs = []
        for p in s.split(","):
            k, v = p.split(":", 1)
            pairs.append((int(k), int(v)))
        return pairs
    except Exception as e:
        raise ArgumentTypeError(
            "Expected an int or comma-separated int:int pairs"
        ) from e


@dataclass
class Retrieval(Context):
    """
    CLI command to evaluate function retrieval
    """

    pool_size: Optional[int]
    seed: int  # Seed for selection of targets, choosed randomly if not set
    binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    platform: Union[
        str, list[tuple[str, str]], None
    ]  # Run for a specific platform, or run on all pairs, or run on all platforms if None
    optimization: Union[
        int, list[tuple[int, int]], None
    ]  # Run for a specific optimization, or run on all pairs, or run on all optimizations if None.
    batch_size: int  # Number of batches processed at once
    context_size: int  # Context window for the LLM
    data_path: str  # Path containing the dataset

    save_metrics: bool  # Save results to a file

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=platform_parser)
        parser.add_argument("--optimization", type=optimization_parser)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--context-size", type=int, default=8192)
        parser.add_argument("--save-metrics", action="store_true")
        parser.add_argument("data_path", type=str)

    def __call__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        accelerator = Accelerator()

        metrics = []

        if isinstance(self.platform, list):
            optimization = (
                None if isinstance(self.optimization, list) else self.optimization
            )

            for query_platform, target_platform in self.platform:
                dataset = PairsDataset(
                    self.data_path,
                    accelerator.is_local_main_process,
                    self.pool_size,
                    self.seed,
                    self.binary,
                    optimization,
                    query_platform,
                    None,
                    target_platform,
                )
                scores = self.generate_scores(accelerator, dataset)

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(scores)
                    parameters = {
                        "binary": self.binary or "all",
                        "platform": query_platform,
                        "target-platform": target_platform,
                        "optimization": "all"
                        if self.optimization is None
                        or isinstance(self.optimization, list)
                        else self.optimization,
                        "pool-size": self.pool_size,
                        "examples": self.examples,
                        "prompt": self.prompt,
                        "model": self.model,
                    }
                    data = {
                        "parameters": parameters,
                        "results": raw_metrics,
                    }

                    metrics.append(data)
                    print(metrics[-1])

        if isinstance(self.optimization, list):
            platform = None if isinstance(self.platform, list) else self.platform

            for query_optimization, target_optimization in self.optimization:
                dataset = PairsDataset(
                    self.data_path,
                    accelerator.is_local_main_process,
                    self.pool_size,
                    self.seed,
                    self.binary,
                    query_optimization,
                    platform,
                    target_optimization,
                    None,
                )
                scores = self.generate_scores(accelerator, dataset)

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(scores)
                    parameters = {
                        "binary": self.binary or "all",
                        "optimization": query_optimization,
                        "target-optimization": target_optimization,
                        "platform": "all"
                        if self.platform is None or isinstance(self.platform, list)
                        else self.platform,
                        "pool-size": self.pool_size,
                        "examples": self.examples,
                        "prompt": self.prompt,
                        "model": self.model,
                    }
                    data = {
                        "parameters": parameters,
                        "results": raw_metrics,
                    }

                    metrics.append(data)

                    if self.save_metrics:
                        save_metrics(metrics, timestamp)

                    print(metrics[-1])

        print("done")

    def generate_scores(
        self, accelerator: Accelerator, dataset: PairsDataset
    ) -> tuple[list[str], list[str]]:
        # No need to prepare the model, because we only do inference
        model = self.get_model(accelerator)

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        loader = accelerator.prepare_data_loader(loader, device_placement=False)

        query_outputs = []
        target_outputs = []

        clear_cache_counter = 0
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Generating",
                disable=not accelerator.is_local_main_process,
            ):
                # Tokenize the prompts for the batch
                (queries, targets) = zip(*batch)

                query_outputs.extend(self.generate(queries, accelerator, model))
                target_outputs.extend(self.generate(targets, accelerator, model))

                if clear_cache_counter == CLEAR_CACHE_PERIOD:
                    torch.cuda.empty_cache()
                    gc.collect()
                    clear_cache_counter = 0

        all_queries = accelerator.gather_for_metrics(query_outputs)
        all_targets = accelerator.gather_for_metrics(target_outputs)

        with open("query-outputs.txt", "w", encoding = "utf-8") as file:
            for idx, query in enumerate(all_queries):
                file.write(f"##### {idx}\n")
                file.write(query)
                file.write("\n")

        with open("target-outputs.txt", "w", encoding = "utf-8") as file:
            for idx, query in enumerate(all_targets):
                file.write(f"##### {idx}\n")
                file.write(query)
                file.write("\n")

        scores: list[list[float]] = []
        for index, query in tqdm(
            enumerate(query_outputs),
            desc="Scoring results",
            disable=not accelerator.is_main_process,
        ):
            scores.append([])
            query = flatten_to_strings(query)
            for target in all_targets:
                scores[index].append(jaccard_index(query, flatten_to_strings(target)))

        # Assemble all scores together for main process
        all_scores = accelerator.gather_for_metrics(scores)

        if accelerator.is_main_process:
            for index, score in all_scores:
                maximum_score = max(score)
                if maximum_score > 0.7 and score.index(maximum_score) == index:
                    print(f"found example {index} with similarity {maximum_score}")

        torch.cuda.empty_cache()
        return all_scores

    def generate(self, batch, accelerator: Accelerator, model) -> list[object]:
        tokens = self.tokenize_prompts([str(f) for f in batch]).to(
            accelerator.device
        )
        # Pass the tokens to LLM
        outputs = model.generate(
            **tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature = 0.5
        )[:, tokens["input_ids"].shape[1] :].cpu()

        generated = [parse_json(d) for d in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        wrong_json_indices = [i for i, x in enumerate(generated) if x is None]

        for index in wrong_json_indices:
            indexed_tokens = { key: val[index].unsqueeze(0) for key, val in tokens.items() }

            for x in range(3):
                print(f"got invalid JSON, retrying {x + 1}/3")
                output = model.generate(
                    **indexed_tokens,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature = 1. + 0.5 * x
                )[:, indexed_tokens["input_ids"].shape[1] :].cpu()

                gen = parse_json(self.tokenizer.decode(output[0], skip_special_tokens=True))
                if gen is not None:
                    generated[index] = gen
                    break
        
        return generated

    def tokenize_prompts(self, fns: list[str]):
        size_for_query = self.context_size - self.empty_prompt_size
        batch = self.tokenizer(
            fns,
            truncation=True,
            padding=False,
            max_length=size_for_query,
        )
        for idx, ids in enumerate(batch["input_ids"]):
            if len(ids) < size_for_query:
                continue

            # -- decode -> cut at last '\n' -> re-encode --
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            trimmed_text = decoded.rsplit("\n", 1)[0]  # everything up to LAST newline
            fns[idx] = trimmed_text

        chat = self.tokenizer.apply_chat_template(
            [self.get_prompt(f) for f in fns],
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.tokenizer(
            chat,
            truncation=False,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )


def save_metrics(metrics, timestamp):
    with open(Path("metrics") / f"{timestamp}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file)


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


def test_retrieval(scores: list[list[float]]):
    """
    Tests the retrieval of each query against the pool of candidates (values).

    # Arguments
    query_tokens: 2D Tensor containing an embedding for each candidate.
    target_tokens: 2D Tensor containing an embedding for each candidate.
    """

    return compute_retrieval_metrics(np.array(scores), None)


def compute_retrieval_metrics(scores, relevance):
    if relevance is None:
        relevance = np.arange(scores.shape[0])
    mrr = calculate_mrr(scores, relevance)
    recall_at_1 = recall_at_k(scores, relevance, 1)
    recall_at_10 = recall_at_k(scores, relevance, 10)

    max_similarity_mean = sum(max(col) for col in scores) / len(scores)
    min_similarity_mean = sum(min(col) for col in scores) / len(scores)
    mean_similarity_mean = sum(mean(col) for col in scores) / len(scores)
    median_similarity_mean = sum(median(col) for col in scores) / len(scores)

    return {
        "mrr": mrr,
        "recall_at_1": recall_at_1,
        "recall_at_10": recall_at_10,
        "similarity_mean": {
            "max": max_similarity_mean,
            "median": median_similarity_mean,
            "mean": mean_similarity_mean,
            "min": min_similarity_mean,
        },
    }


def jaccard_index(query: list[str], potential_target: list[str]) -> float:
    set1 = set(query)
    set2 = set(potential_target)
    intersection = set1 & set2
    union = set1 | set2
    return (
        len(intersection) / len(union) if union else 1.0
    )  # Return 1.0 if both sets are empty


def flatten_to_strings(obj, parent_key="", sep="."):
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}"
            result.extend(flatten_to_strings(v, new_key, sep))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(flatten_to_strings(item, parent_key, sep))
    else:
        result.append(f"{parent_key}{sep}{obj}")
    return result


def parse_json(s: str) -> Optional[object]:
    """
    Parse the generated json, and 
    """

    s = s.strip()  # Remove surrounding whitespace
    _, _, s = s.partition("```json") # Remove up to and including the opening "```json"
    s = s.removesuffix("```")
    s = s.strip()  # Remove any remaining whitespace or newlines

    try:
        parsed = json.loads(s)
        return parsed
    except json.JSONDecodeError:
        with open("invalid-json.txt", "a", encoding="utf-8") as file:
            file.write("#####\n")
            file.write(s)
            file.write("\n")

        return None