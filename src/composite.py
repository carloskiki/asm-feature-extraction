"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Union
from argparse import ArgumentParser
import random
import sys
from google import genai
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from .parsing import platform_parser, optimization_parser
from .metrics import (
    save_metrics,
    test_retrieval,
)
from .gemini import GeminiRetrieval
from .data_processing import (
    BINARIES,
    PairsDataset,
)
from .context import Context

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")


@dataclass
class Composite(Context):
    """
    CLI command to evaluate function retrieval
    """

    pool_size: Optional[int]
    top_k: int
    seed: int  # Seed for selection of targets, uses random seed if not set
    binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    platform: Union[
        str, list[tuple[str, str]], None
    ]  # Run for a specific platform, or run on all pairs, or run on all platforms if None
    optimization: Union[
        int, list[tuple[int, int]], None
    ]  # Run for a specific optimization, or run on all pairs, or run on all optimizations if None.
    batch_size: int  # Number of batches processed at once
    data_path: str  # Path containing the dataset

    save_metrics: bool  # Save results to a file

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "composite",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--top-k", type=int, default=10)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=platform_parser)
        parser.add_argument("--optimization", type=optimization_parser)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--save-metrics", action="store_true")
        parser.add_argument("data_path", type=str)

    def __call__(self):
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))]
        )

        metrics = []
        gemini = GeminiRetrieval(
            model="",
            prompt="extensive-3",
            examples=3,
            pool_size=self.pool_size,
            seed=self.seed,
            binary=self.binary,
            platform=self.platform,
            optimization=self.optimization,
            batch_size=self.batch_size,
            data_path=self.data_path,
            request_per_minute=60,
            save_metrics=self.save_metrics,
            action="normal",
        )
        client = genai.Client()

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

                first_part = self.generate_emb_scores(accelerator, dataset)
                second_part = gemini.generate_scores(dataset, client)
                score = (np.array(first_part) + np.array(second_part)) / 2

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(score)
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

                    if self.save_metrics:
                        save_metrics(metrics, timestamp)

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

                first_part = self.generate_emb_scores(accelerator, dataset)
                second_part = gemini.generate_scores(dataset, client)
                score = (np.array(first_part) + np.array(second_part)) / 2

                if accelerator.is_main_process:
                    raw_metrics = test_retrieval(score)
                    parameters = {
                        "binary": self.binary or "all",
                        "optimization": query_optimization,
                        "target-optimization": target_optimization,
                        "platform": (
                            "all"
                            if self.platform is None or isinstance(self.platform, list)
                            else self.platform
                        ),
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

    def generate_emb_scores(self, accelerator: Accelerator, dataset: PairsDataset):
        model = self.get_model(accelerator)

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        loader = accelerator.prepare_data_loader(loader, device_placement=False)

        query_embs = []
        target_embs = []

        for batch in tqdm(
            loader,
            desc="Generating",
            disable=not accelerator.is_main_process,
        ):
            (queries, targets) = zip(*batch)
            queries = ["\n".join(str(q).splitlines()[:128]) for q in queries]
            targets = ["\n".join(str(t).splitlines()[:128]) for t in targets]

            query_embs.extend(model.encode(queries))
            target_embs.extend(model.encode(targets))

        query_embs = np.stack(query_embs, axis=0)
        target_embs = np.stack(target_embs, axis=0)

        target_embs = accelerator.gather_for_metrics(
            target_embs, use_gather_object=True
        )
        scores = model.similarity(query_embs, target_embs).tolist()

        all_scores = accelerator.gather_for_metrics(scores)

        return all_scores
