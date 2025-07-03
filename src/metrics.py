from typing import Optional
from pathlib import Path
import json
from statistics import median, mean
import numpy as np

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