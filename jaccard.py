from torch import Tensor
from torchmetrics.functional.classification import multiclass_jaccard_index

def similarity(query: Tensor, potential_target: Tensor, vocab_size: int):
    """
    Run Jaccard Similarity over the generated tokens of an LLM query
    """

    multiclass_jaccard_index(query, potential_target, vocab_size)