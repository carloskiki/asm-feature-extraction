from argparse import ArgumentParser
from typing import Optional, Union
from dataclasses import dataclass
import sys
import pickle
import random
from datetime import datetime
from pydantic import BaseModel
from google import genai
from tqdm import tqdm
from torch.utils.data import DataLoader
from context import Context, MAX_NEW_TOKENS
from parsing import platform_parser, optimization_parser
from data_processing import BINARIES, PairsDataset, LibDataset
from metrics import save_metrics, flatten_to_strings, jaccard_index, test_retrieval


class Schema(BaseModel):
    input_parameter_count: int
    input_parameter_types: list[str]
    return_value_type: str
    dominant_operation_categories: list[str]
    loop_indicators: bool
    number_of_distinct_subroutine_call_targets: int
    use_indexed_addressing_modes: bool
    presence_of_notable_integer_constants: list[str]
    presence_of_notable_floating_point_constants: list[float]
    count_of_distinct_immediate_values: int
    likely_modifies_input_parameters: bool
    likely_modifies_global_state: bool
    likely_performs_memory_allocation_deallocation: bool
    likely_performs_io_operations: bool
    likely_performs_block_memory_operations: bool
    inferred_algorithm: str


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

    save_metrics: bool  # Save results to a file

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
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=platform_parser)
        parser.add_argument("--optimization", type=optimization_parser)
        parser.add_argument("--save-metrics", action="store_true")
        parser.add_argument("data_path", type=str)

    def __call__(self, *args, **kwds):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.

        client = genai.Client()
        metrics = []

        dataset = LibDataset(self.data_path, True, pool_size=self.pool_size, binary=self.binary, optimization=self.optimization, platform=self.platform)

        self.generate(dataset, client)
        print("done generating")

        # if isinstance(self.optimization, list):
        #     platform = None if isinstance(self.platform, list) else self.platform

        #     for query_optimization, target_optimization in self.optimization:
        #         dataset = PairsDataset(
        #             self.data_path,
        #             True,
        #             self.pool_size,
        #             self.seed,
        #             self.binary,
        #             query_optimization,
        #             platform,
        #             target_optimization,
        #             None,
        #         )
        #         scores = self.generate_scores(dataset)

        #         raw_metrics = test_retrieval(scores)
        #         parameters = {
        #             "binary": self.binary or "all",
        #             "optimization": query_optimization,
        #             "target-optimization": target_optimization,
        #             "platform": "all"
        #             if self.platform is None or isinstance(self.platform, list)
        #             else self.platform,
        #             "pool-size": self.pool_size,
        #             "examples": self.examples,
        #             "prompt": self.prompt,
        #             "model": self.model,
        #         }
        #         data = {
        #             "parameters": parameters,
        #             "results": raw_metrics,
        #         }

        #         metrics.append(data)
        #         if self.save_metrics:
        #             save_metrics(metrics, datetime.now().strftime("%Y-%m-%d_%H-%M"))

        #         print(metrics[-1])

    def generate_scores(
        self, dataset: PairsDataset, client: genai.Client
    ) -> list[list[float]]:
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x: x
        )

        query_outputs = []
        target_outputs = []

        for batch in tqdm(loader):
            # Tokenize the prompts for the batch
            (queries, targets) = zip(*batch)

            query_outputs.extend(self.generate(queries, client))
            target_outputs.extend(self.generate(targets, client))

        scores: list[list[float]] = []
        for index, query in tqdm(
            enumerate(query_outputs),
            desc="Scoring results",
        ):
            scores.append([])
            query = flatten_to_strings(query)
            for target in target_outputs:
                scores[index].append(jaccard_index(query, flatten_to_strings(target)))

    def generate(self, batch, client: genai.Client):
        prompt = self.get_prompt("")
        system_prompt = prompt[0]["content"]

        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_otkens=MAX_NEW_TOKENS,
                response_mime_type="aplicatoin/json",
                response_schema=Schema,
            ),
            history=[
                genai.types.Content(
                    role=obj["role"] if obj["role"] == "user" else "model",
                    parts=[genai.types.Part.from_text(text=obj["content"])],
                )
                for obj in prompt[1:]
            ],
        )

        responses = []

        for fn in batch:
            responses.append(chat.send_message(f"```assembly\n{str(fn)}\n```"))
        

        with open("saved-chats.pkl", "w") as file:
            pickle.dump(responses, file)