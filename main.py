"""
Main.
"""

import argparse
import os
from query import Query
from bogus import Bogus
from retrieval import Retrieval
from batch_query import BatchQuery
from gemini import GeminiRetrieval
from info import Info
from context import MODELS

COMMANDS = {
    'query': Query,
    'retrieval': Retrieval,
    'bogus': Bogus,
    'batch-query': BatchQuery,
    'info': Info,
    'gemini': GeminiRetrieval
}

def main():
    """
    Main again.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", choices=MODELS.keys(), type=str, default="qwen-2.5-0.5"
    )
    parser.add_argument("-p", "--prompt", choices=os.listdir("prompts"), type=str, default="tokens-2")
    parser.add_argument("-e", "--examples", type=int, default=0)
    subparsers = parser.add_subparsers(
        description="the action to be performed", dest="subcommand", required=True
    )

    for command in COMMANDS.values():
        command.arguments(subparsers)

    args = parser.parse_args()
    subcommand = args.subcommand
    delattr(args, "subcommand")
    COMMANDS[subcommand](**vars(args))()

if __name__ == "__main__":
    main()
