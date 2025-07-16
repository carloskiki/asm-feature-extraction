from argparse import ArgumentParser
from typing import Optional, Union
from dataclasses import dataclass
import sys
import random
import time
from datetime import datetime
from google import genai
from google.genai import types
from tqdm import tqdm
from torch.utils.data import DataLoader
from .context import Context
from .parsing import platform_parser, optimization_parser
from .data_processing import BINARIES, PairsDataset
from .metrics import (
    save_metrics,
    flatten_to_strings,
    jaccard_index,
    test_retrieval,
    parse_json,
)

date = datetime.now().strftime("%Y-%m-%d_%H-%M")


@dataclass
class GeminiRetrieval(Context):
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
    request_per_minute: int  # Maximum number of requests per minute to run

    save_metrics: bool  # Save results to a file
    action: str

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser: ArgumentParser = subparsers.add_parser(
            "gemini",
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

        action = parser.add_subparsers(dest="action")
        action.add_parser("batch", description="Send batch file")
        action.add_parser("normal", description="Normal mode")

        parser.add_argument("data_path", type=str)

    def __call__(self, *args, **kwds):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.

        client = genai.Client()
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
                if self.action == "batch":
                    self.batch_send(dataset, client)
                    continue

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
                    "model": "gemini-2.5-flash-lite-preview-06-17",
                }
                data = {
                    "parameters": parameters,
                    "results": raw_metrics,
                }

                metrics.append(data)
                if self.save_metrics:
                    save_metrics(metrics, date)

                print(metrics[-1])

        if isinstance(self.platform, list):
            optimization = (
                None if isinstance(self.optimization, list) else self.optimization
            )

            for query_platform, target_platform in self.platform:
                dataset = PairsDataset(
                    self.data_path,
                    True,
                    self.pool_size,
                    self.seed,
                    self.binary,
                    optimization,
                    query_platform,
                    None,
                    target_platform,
                )
                if self.action == "batch":
                    self.batch_send(dataset, client)
                    continue

                scores = self.generate_scores(dataset, client)

                raw_metrics = test_retrieval(scores)
                parameters = {
                    "binary": self.binary or "all",
                    "optimization": self.platform,
                    "platform": query_platform,
                    "target-platform": target_platform,
                    "pool-size": self.pool_size,
                    "examples": self.examples,
                    "prompt": self.prompt,
                    "model": "gemini-2.5-flash-lite-preview-06-17",
                }
                data = {
                    "parameters": parameters,
                    "results": raw_metrics,
                }

                metrics.append(data)
                if self.save_metrics:
                    save_metrics(metrics, date)

                print(metrics[-1])

    def generate_scores(
        self, dataset: PairsDataset, client: genai.Client
    ) -> list[list[float]]:
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )

        query_outputs = []
        target_outputs = []

        interval = 60 * 2 * self.batch_size / self.request_per_minute

        for _index, batch in enumerate(tqdm(loader)):
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

    def generate(self, batch, client: genai.Client):
        # TODO: Use cache instead
        prompt = self.get_prompt("")
        system_prompt = prompt[0]["content"]

        chat = client.chats.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
            history=[
                types.Content(
                    role=obj["role"] if obj["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=obj["content"])],
                )
                for obj in prompt[1:-1]
            ],
        )

        responses = []
        for fn in batch:
            print(fn.name)
            # We could instead provide a schema for the model to follow and not parse. But we want to
            # imitate our local setup as much as possible.
            responses.append(
                parse_json(
                    chat.send_message(f"```assembly\n{str(fn)[:10_000]}\n```").text
                )
            )

        return responses

    def batch_send(self, dataset: PairsDataset, client: genai.Client):
        cache = self.cache_system_prompt(client, "gemini-2.5-flash")
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )

        # for batch in tqdm(loader):
        #     pass

        query, _ = next(loader)[0]

        generated = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            config=types.GenerateContentConfig(
                cached_content=cache.name,
            ),
            contents=types.Content(role="user", parts=[types.Part(text=f"```assembly\n{str(query)[:10_000]}\n```")])
        )

        import code
        code.interact(local=locals())


    def cache_system_prompt(self, client: genai.Client, model: str) -> types.CachedContent:
        prompt = self.get_prompt("")
        system_prompt = prompt[0]["content"]

        return client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=f"prompt-{date}",  # used to identify the cache
                system_instruction=system_prompt,
                contents=[
                    types.Content(
                        role=obj["role"] if obj["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=obj["content"])],
                    )
                    for obj in prompt[1:-1]
                ],
                ttl="6000s"
            )
        )