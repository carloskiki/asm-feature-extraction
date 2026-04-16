"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union
from argparse import ArgumentParser
from itertools import islice
import random
import sys
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from .parsing import platform_parser, optimization_parser, obfuscation_parser
from .metrics import (
    save_metrics,
    test_retrieval,
)
from .data_processing import BINARIES, PairsDataset, Function
from .context import Context

CLEAR_CACHE_PERIOD = 32


@dataclass
class Clap(Context):
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
    obfuscation: Union[
        str, list[tuple[str, str]], None
    ]  # Run for a specific obfuscation, or run on obfuscation pairs.
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
            "clap",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=platform_parser)
        parser.add_argument("--optimization", type=optimization_parser)
        parser.add_argument("--obfuscation", type=obfuscation_parser)
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
            obfuscation = (
                None if isinstance(self.obfuscation, list) else self.obfuscation
            )

            for query_platform, target_platform in self.platform:
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=accelerator.is_local_main_process,
                    pool_size=self.pool_size,
                    seed=self.seed,
                    binary=self.binary,
                    optimization=optimization,
                    platform=query_platform,
                    optimization_diff=None,
                    platform_diff=target_platform,
                    obfuscation=obfuscation,
                    obfuscation_diff=None,
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
                        "obfuscation": "all"
                        if self.obfuscation is None
                        or isinstance(self.obfuscation, list)
                        else self.obfuscation,
                        "pool-size": self.pool_size,
                        "model": "clap",
                    }
                    data = {
                        "parameters": parameters,
                        "results": raw_metrics,
                    }

                    metrics.append(data)
                    print(metrics[-1])

        if isinstance(self.optimization, list):
            platform = None if isinstance(self.platform, list) else self.platform
            obfuscation = (
                None if isinstance(self.obfuscation, list) else self.obfuscation
            )

            for query_optimization, target_optimization in self.optimization:
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=accelerator.is_local_main_process,
                    pool_size=self.pool_size,
                    seed=self.seed,
                    binary=self.binary,
                    optimization=query_optimization,
                    platform=platform,
                    optimization_diff=target_optimization,
                    platform_diff=None,
                    obfuscation=obfuscation,
                    obfuscation_diff=None,
                )
                scores = self.generate_scores(accelerator, dataset)

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(scores)
                    parameters = {
                        "binary": self.binary or "all",
                        "optimization": query_optimization,
                        "target-optimization": target_optimization,
                        "platform": (
                            "all"
                            if self.platform is None or isinstance(self.platform, list)
                            else self.platform
                        ),
                        "obfuscation": "all"
                        if self.obfuscation is None
                        or isinstance(self.obfuscation, list)
                        else self.obfuscation,
                        "pool-size": self.pool_size,
                        "model": "clap",
                    }
                    data = {
                        "parameters": parameters,
                        "results": raw_metrics,
                    }

                    metrics.append(data)

                    if self.save_metrics:
                        save_metrics(metrics, timestamp)

                    print(metrics[-1])

        if isinstance(self.obfuscation, list):
            platform = None if isinstance(self.platform, list) else self.platform
            optimization = (
                None if isinstance(self.optimization, list) else self.optimization
            )

            for query_obfuscation, target_obfuscation in self.obfuscation:
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=accelerator.is_local_main_process,
                    pool_size=self.pool_size,
                    seed=self.seed,
                    binary=self.binary,
                    optimization=optimization,
                    platform=platform,
                    optimization_diff=None,
                    platform_diff=None,
                    obfuscation=query_obfuscation,
                    obfuscation_diff=target_obfuscation,
                )
                scores = self.generate_scores(accelerator, dataset)

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(scores)
                    parameters = {
                        "binary": self.binary or "all",
                        "obfuscation": query_obfuscation,
                        "target-obfuscation": target_obfuscation,
                        "platform": "all"
                        if self.platform is None or isinstance(self.platform, list)
                        else self.platform,
                        "optimization": "all"
                        if self.optimization is None
                        or isinstance(self.optimization, list)
                        else self.optimization,
                        "pool-size": self.pool_size,
                        "model": "clap",
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
        model = AutoModel.from_pretrained(
            "hustcw/clap-asm",
            trust_remote_code=True,
        ).to(accelerator.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hustcw/clap-asm", trust_remote_code=True
        )

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

                query_outputs.extend(self.generate(queries, model, accelerator))
                target_outputs.extend(self.generate(targets, model, accelerator))

                if clear_cache_counter == CLEAR_CACHE_PERIOD:
                    torch.cuda.empty_cache()
                    gc.collect()
                    clear_cache_counter = 0

        all_targets = accelerator.gather_for_metrics(target_outputs, use_gather_object=True)

        scores: list[list[float]] = []
        for index, query in tqdm(
            enumerate(query_outputs),
            desc="Scoring results",
            disable=not accelerator.is_main_process,
        ):
            scores.append([])
            for target in all_targets:
                cosine = torch.nn.CosineSimilarity(dim=0)
                similarity = cosine(query, target).item()
                scores[index].append(similarity)

        # Assemble all scores together for main process
        all_scores = accelerator.gather_for_metrics(scores)

        torch.cuda.empty_cache()
        return all_scores

    def generate(self, batch: list[Function], model, accelerator: Accelerator) -> list[object]:
        instructions = (
            (str(ins) for block in f.blocks for ins in block.instructions)
            for f in batch
        )

        tokens = self.tokenizer(
            [
                {str(i): instruction for i, instruction in enumerate(islice(f, 512))}
                for f in instructions
            ],
            padding=True,
            return_tensors="pt",
        ).to(accelerator.device)
        # Pass the tokens to LLM
        embeddings = model(**tokens).cpu()

        return embeddings
