from argparse import ArgumentParser
from typing import Optional, Union, List
from dataclasses import dataclass
from pydantic import BaseModel
import sys
import random
import time
from datetime import datetime
from google import genai
from google.genai import types, errors
from tqdm import tqdm
from torch.utils.data import DataLoader
from .context import Context
from .parsing import platform_parser, optimization_parser, obfuscation_parser
from .data_processing import BINARIES, PairsDataset
from .metrics import (
    save_metrics,
    flatten_to_strings,
    jaccard_index,
    test_retrieval,
    parse_json,
)

date = datetime.now().strftime("%Y-%m-%d_%H-%M")


class FunctionAnalysis(BaseModel):
    input_parameter_count: int
    input_parameter_types: List[str]
    return_value_type: str
    dominant_operation_categories: List[str]
    loop_indicators: bool
    number_of_distinct_subroutine_call_targets: int
    use_of_indexed_addressing_modes: bool
    jump_table_indicators: bool
    presence_of_simd_instructions: bool
    presence_of_notable_integer_constants: List[str]
    presence_of_notable_floating_point_constants: List[float]
    count_of_distinct_immediate_values: int
    string_literal_presence: bool
    likely_modifies_input_parameters: bool
    likely_modifies_global_state: bool
    likely_performs_memory_allocation_deallocation: bool
    likely_performs_io_operations: bool
    likely_performs_block_memory_operations: bool
    likely_performs_linear_memory_accesses: bool
    likely_performs_error_handling: bool
    number_of_software_interrupts_system_calls: int
    inferred_algorithm: str
    inferred_category: str


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
    obfuscation: Union[
        str, list[tuple[str, str]], None
    ]  # Run for a specific obfuscation, or run on obfuscation pairs.
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
        parser.add_argument("--obfuscation", type=obfuscation_parser)
        parser.add_argument("--save-metrics", action="store_true")

        action = parser.add_subparsers(dest="action")
        action.add_parser("batch", description="Send batch file")
        action.add_parser("normal", description="Normal mode")
        action.add_parser("no-cache", description="Don't use cache")

        parser.add_argument("data_path", type=str)

    def __call__(self, *args, **kwds):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.

        client = genai.Client()
        metrics = []

        if isinstance(self.optimization, list):
            platform = None if isinstance(self.platform, list) else self.platform

            for query_optimization, target_optimization in self.optimization:
                obfuscation = (
                    None if isinstance(self.obfuscation, list) else self.obfuscation
                )
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=True,
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
                    "obfuscation": "all"
                    if self.obfuscation is None
                    or isinstance(self.obfuscation, list)
                    else self.obfuscation,
                    "pool-size": self.pool_size,
                    "examples": self.examples,
                    "prompt": self.prompt,
                    "model": "gemini-2.5-flash",
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
            obfuscation = (
                None if isinstance(self.obfuscation, list) else self.obfuscation
            )

            for query_platform, target_platform in self.platform:
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=True,
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
                    "obfuscation": "all"
                    if self.obfuscation is None
                    or isinstance(self.obfuscation, list)
                    else self.obfuscation,
                    "pool-size": self.pool_size,
                    "examples": self.examples,
                    "prompt": self.prompt,
                    "model": "gemini-2.5-flash",
                }
                data = {
                    "parameters": parameters,
                    "results": raw_metrics,
                }

                metrics.append(data)
                if self.save_metrics:
                    save_metrics(metrics, date)

                print(metrics[-1])

        if isinstance(self.obfuscation, list):
            platform = None if isinstance(self.platform, list) else self.platform
            optimization = (
                None if isinstance(self.optimization, list) else self.optimization
            )

            for query_obfuscation, target_obfuscation in self.obfuscation:
                dataset = PairsDataset(
                    path=self.data_path,
                    main_process=True,
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

                if self.action == "batch":
                    self.batch_send(dataset, client)
                    continue

                scores = self.generate_scores(dataset, client)
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
                    "examples": self.examples,
                    "prompt": self.prompt,
                    "model": "gemini-2.5-flash",
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
        model = "gemini-2.5-flash"
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )
        if self.action != "no-cache":
            cache = self.cache_system_prompt(client, model, 5 * self.pool_size)
        else:
            cache = None

        function_names = []
        stored_queries = []
        query_outputs = []
        stored_targets = []
        target_outputs = []

        interval = 60 * 2 * self.batch_size / self.request_per_minute

        for _index, batch in enumerate(tqdm(loader)):
            start_time = time.time()

            # Tokenize the prompts for the batch
            (queries, targets) = zip(*batch)

            function_names.extend(q.name for q in queries)
            stored_queries.extend(str(q) for q in queries)
            query_outputs.extend(self.generate(queries, client, cache))
            stored_targets.extend(str(t) for t in targets)
            target_outputs.extend(self.generate(targets, client, cache))

            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))

        archive = {
            "function_names": function_names,
            "queries": stored_queries,
            "query_outputs": query_outputs,
            "targets": stored_targets,
            "target_outputs": target_outputs,
        }

        with open("archive.pkl", "wb") as file:
            import pickle
            pickle.dump(archive, file)

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

    def generate(
        self, batch, client: genai.Client, cache: Optional[types.CachedContent]
    ):
        responses = []
        for fn in batch:
            # We could instead provide a schema for the model to follow and not parse. But we want to
            # imitate our local setup as much as possible.

            fn_str = f"```assembly\n{'\n'.join(str(fn).splitlines()[:256])}\n```"
            print(fn_str)

            if cache is not None:
                config = types.GenerateContentConfig(
                    cached_content=cache.name,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=800,
                )
                contents = fn_str
            else:
                prompt = self.get_prompt("")
                system_prompt = prompt[0]["content"]

                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=800,
                    response_mime_type="application/json",
                    response_schema=FunctionAnalysis,
                )
                contents = [
                    types.Content(
                        role=obj["role"] if obj["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=obj["content"])],
                    )
                    for obj in prompt[1:-1]
                ]
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=fn_str)]
                    )
                )

            for _ in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", config=config, contents=contents
                    )
                except errors.APIError:
                    time.sleep(60)
                    continue
                break

            parsed = parse_json(response.text)
            if parsed is None:
                print(parsed)
            responses.append(parsed)

        return responses

    def batch_send(self, dataset: PairsDataset, client: genai.Client):
        model = "gemini-2.5-flash"
        cache = self.cache_system_prompt(client, model, 120)
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )

        queries, _ = zip(*next(iter(loader)))

        batch_response = client.batches.create(
            model=model,
            src=[
                types.InlinedRequest(
                    model=model,
                    config=types.GenerateContentConfig(
                        cached_content=cache.name,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                    contents=types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"```assembly\n{str(query)[:10_000]}\n```")
                        ],
                    ),
                )
                for query in queries
            ],
            config=types.CreateBatchJobConfig(
                display_name=date,
            ),
        )

        import code

        code.interact(local=locals())

    def cache_system_prompt(
        self, client: genai.Client, model: str, duration: int
    ) -> types.CachedContent:
        prompt = self.get_prompt("")
        system_prompt = prompt[0]["content"]
        contents = [
            types.Content(
                role=obj["role"] if obj["role"] == "user" else "model",
                parts=[types.Part.from_text(text=obj["content"])],
            )
            for obj in prompt[1:-1]
        ]

        return client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=f"prompt-{date}",  # used to identify the cache
                system_instruction=system_prompt,
                contents=contents,
                ttl=f"{duration}s",
            ),
        )
