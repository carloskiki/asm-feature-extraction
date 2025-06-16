import random
from typing import Optional
from dataclasses import dataclass
import sys
import gc

from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from data_processing import LibDataset, BINARIES, PLATFORMS
from context import Context, MAX_NEW_TOKENS

MAX_LENGTH = 8192
CLEAR_CACHE_PERIOD = 32

@dataclass
class BatchQuery(Context):
    """
    Run queries from the dataset in batch for output analysis
    """
    size: Optional[int]
    batch_size: int
    seed: int  # Seed for selection of targets, choosed randomly if not set
    binary: Optional[str]  # Run for a specific binary, run on all binaries if None
    platform: Optional[str]  # Run for a specific platform, run on all platforms if None
    optimization: Optional[
        int
    ]  # Run for a specific optimization, run on all optimizations if None
    context_size: int  # Context window for the LLM
    data_path: str
    out_file: Optional[str]

    @staticmethod
    def arguments(subparsers):
        """
        Configure the CLI
        """

        parser = subparsers.add_parser(
            "batch-query",
            description="Find the most similar assembly function from a set",
        )
        parser.add_argument("--size", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=random.randrange(sys.maxsize))
        parser.add_argument("--binary", type=str, choices=BINARIES.keys())
        parser.add_argument("--platform", type=str, choices=PLATFORMS.keys())
        parser.add_argument("--optimization", type=int, choices=range(4))
        parser.add_argument("--context-size", type=int, default=2048)
        parser.add_argument("data_path", type=str)
        parser.add_argument("out_file", type=str)

    def __call__(self):
        accelerator = Accelerator()

        # No need to prepare the model, because we only do inference
        model = self.get_model(accelerator)
        tokenizer = self.get_tokenizer()

        dataset = LibDataset(
            self.data_path,
            accelerator.is_local_main_process,
            self.size,
            self.seed,
            self.binary,
            self.optimization,
            self.platform,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        loader = accelerator.prepare_data_loader(loader, device_placement=False)

        tokenizer = self.get_tokenizer()

        query_decoded = []
        functions = []

        clear_cache_counter = 0
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Query batches",
                disable=not accelerator.is_local_main_process,
            ):
                # Tokenize the prompts for the batch
                prompts = [self.get_prompt(str(f)) for f in batch]
                functions.extend(batch)

                chat = tokenizer.apply_chat_template(
                    prompts, tokenize=False, add_generation_prompt=True
                )
                token_batch = tokenizer(
                    chat,
                    truncation=True,
                    padding=self.batch_size > 1,
                    padding_side="left",
                    return_tensors="pt",
                    max_length=MAX_LENGTH,
                ).to(accelerator.device)

                # Pass the tokens to LLM
                query_outputs = model.generate(
                    **token_batch,
                    max_new_tokens=MAX_NEW_TOKENS,
                )[:, token_batch["input_ids"].shape[1] :].cpu()
                query_decoded.extend(tokenizer.batch_decode(query_outputs))

                if clear_cache_counter == CLEAR_CACHE_PERIOD:
                    torch.cuda.empty_cache()
                    gc.collect()
                    clear_cache_counter = 0

        # Clear out and or create the file for all processes to write to it later
        if accelerator.is_local_main_process:
            with open(self.out_file, "w", encoding="utf-8") as file:
                pass

        accelerator.wait_for_everyone()

        # TODO: share for metrics instead of writing to file

        print(f"Writing results to {self.out_file}")
        with open(self.out_file, "a", encoding="utf-8") as file:
            for output, fn in zip(query_decoded, functions):
                file.write("############\n")
                file.write("```assembly\n")
                file.write(str(fn))
                file.write("\n```\n")
                file.write("Output:")
                file.write(output)
                file.write("\n")
