"""
Retrieval CLI utilities
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Union
from argparse import ArgumentParser
import random
import sys
import gc
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, InitProcessGroupKwargs
from .parsing import platform_parser, optimization_parser
from .metrics import (
    save_metrics,
    test_retrieval,
    parse_json,
    flatten_to_strings,
    jaccard_index,
)
from .data_processing import (
    BINARIES,
    PairsDataset,
)
from .context import Context, MAX_NEW_TOKENS

CLEAR_CACHE_PERIOD = 32
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")


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

    save_answers: bool  # Save all answers
    save_top_k: Optional[int]  # Save examples that are top-k but not top-1

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
        parser.add_argument("--save-answers", action="store_true")
        parser.add_argument("--save-top-k", type=int)
        parser.add_argument("data_path", type=str)

    def __call__(self):
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))]
        )

        # Nasty nasty patch
        self.invalid_json_count = 0

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
                if self.model.startswith("qwen-emb"):
                    scores = self.generate_emb_scores(accelerator, dataset)
                else:
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
                        "invalid-json": self.invalid_json_count,
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
                if self.model.startswith("qwen-emb"):
                    scores = self.generate_emb_scores(accelerator, dataset)
                else:
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
                        "pool-size": self.pool_size,
                        "examples": self.examples,
                        "prompt": self.prompt,
                        "model": self.model,
                    }
                    data = {
                        "parameters": parameters,
                        "results": raw_metrics,
                        "invalid-json": self.invalid_json_count,
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

        if accelerator.is_main_process and self.save_answers:
            with open(f"examples/all-{timestamp}.txt", "w", encoding="utf-8") as file:
                for index, (query_out, target_out) in enumerate(
                    zip(query_outputs, target_outputs)
                ):
                    (query_fn, target_fn) = loader.dataset[index]
                    file.write(
                        f"##### Q {index} - {query_fn.name}\n```assembly\n{str(query_fn)}\n```\n{query_out}\n\n"
                        f"##### T {index} - {target_fn.name}\n```assembly\n{str(target_fn)}\n```\n{target_out}\n\n"
                    )

        all_targets = accelerator.gather_for_metrics(target_outputs)

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

        if accelerator.is_main_process and self.save_top_k is not None:
            # Get the indices of the top-k scores for each query
            top_k_indices = np.argsort(-np.array(scores), axis=1)[:, : self.save_top_k]

            with open(
                f"examples/top-{self.save_top_k}-{timestamp}", "w", encoding="utf-8"
            ) as f:
                for i, indices in enumerate(top_k_indices):
                    if i in indices and i != indices[0]:
                        f.write(f"{i} - {str(indices)}]\n")

        # Assemble all scores together for main process
        all_scores = accelerator.gather_for_metrics(scores)

        torch.cuda.empty_cache()
        return all_scores

    def generate(self, batch, accelerator: Accelerator, model) -> list[object]:
        tokens = self.tokenize_prompts([str(f) for f in batch]).to(accelerator.device)
        # Pass the tokens to LLM
        outputs = model.generate(
            **tokens, max_new_tokens=MAX_NEW_TOKENS, temperature=0.5
        )[:, tokens["input_ids"].shape[1] :].cpu()

        generated = [
            parse_json(d)
            for d in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        wrong_json_indices = [i for i, x in enumerate(generated) if x is None]

        for index in wrong_json_indices:
            indexed_tokens = {
                key: val[index].unsqueeze(0) for key, val in tokens.items()
            }

            output = model.generate(
                **indexed_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.5,
            )[:, indexed_tokens["input_ids"].shape[1] :].cpu()

            gen = parse_json(self.tokenizer.decode(output[0], skip_special_tokens=True))
            if gen is not None:
                generated[index] = gen
            else:
                self.invalid_json_count += 1

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
            queries = ["\n".join(str(q).splitlines()[:64]) for q in queries]
            targets = ["\n".join(str(t).splitlines()[:64]) for t in targets]

            query_embs.extend(model.encode(queries))
            target_embs.extend(model.encode(targets))

        query_embs = np.stack(query_embs, axis=0)
        target_embs = np.stack(target_embs, axis=0)

        target_embs = accelerator.gather_for_metrics(target_embs, use_gather_object=True)
        scores = model.similarity(query_embs, target_embs).tolist()

        all_scores = accelerator.gather_for_metrics(scores)

        return all_scores
