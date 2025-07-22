from argparse import ArgumentParser
from typing import Optional, Union
from dataclasses import dataclass
import sys
import random
import time
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from openai import OpenAI
from .context import Context 
from .parsing import platform_parser, optimization_parser
from .data_processing import BINARIES, PairsDataset
from .metrics import save_metrics, flatten_to_strings, jaccard_index, test_retrieval, parse_json


@dataclass
class OpenAIRetrieval(Context):
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
    data_path: str  # Path containing the dataset
    request_per_minute: int # Maximum number of requests per minute to run

    save_metrics: bool  # Save results to a file

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "openai",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--request-per-minute", type=int)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=platform_parser)
        parser.add_argument("--optimization", type=optimization_parser)
        parser.add_argument("--save-metrics", action="store_true")
        parser.add_argument("data_path", type=str)

    def __call__(self, *args, **kwds):
        client = OpenAI()
        metrics = []

        if isinstance(self.optimization, list):
            platform = None if isinstance(self.platform, list) else self.platform

            for query_optimization, target_optimization in self.optimization:
                dataset = PairsDataset(
                    self.data_path,
                    True,
                    self.pool_size,
                    self.seed,
                    self.binary,
                    query_optimization,
                    platform,
                    target_optimization,
                    None,
                )
                scores = self.generate_scores(dataset, client)

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
                    "model": "gpt-4.1-mini",
                }
                data = {
                    "parameters": parameters,
                    "results": raw_metrics,
                }

                metrics.append(data)
                if self.save_metrics:
                    save_metrics(metrics, datetime.now().strftime("%Y-%m-%d_%H-%M"))

                print(metrics[-1])

    def generate_scores(
        self, dataset: PairsDataset, client: OpenAI
    ) -> list[list[float]]:
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )

        query_outputs = []
        target_outputs = []

        interval = 60 * 2 * self.batch_size / self.request_per_minute

        for batch in tqdm(loader):
            start_time = time.time()

            # Tokenize the prompts for the batch
            (queries, targets) = zip(*batch)

            query_outputs.extend(self.generate(queries, client))
            target_outputs.extend(self.generate(targets, client))
            
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))

        scores: list[list[float]] = []
        for index, query in tqdm(
            enumerate(query_outputs),
            desc="Scoring results",
        ):
            scores.append([])
            query = flatten_to_strings(query)
            for target in target_outputs:
                scores[index].append(jaccard_index(query, flatten_to_strings(target)))
        
        return scores

    def generate(self, batch, client: OpenAI):

        responses = []
        for fn in batch:
            # We could instead provide a schema for the model to follow and not parse. But we want to imitate
            # our local setup as much as possible.
            response = client.chat.completions.create(
                messages=self.get_prompt('\n'.join(str(fn).splitlines()[:256])),
                model="gpt-4.1-mini",
                user="l1nna-lab",
            )

            responses.append(response.choices[0].message.content)

        return responses