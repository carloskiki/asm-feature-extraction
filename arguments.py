from typing import Union
from retrieval import Retrieval
from query import Query
from store_dataset import StoreDataset
import argparse

def arguments() -> Union[Query, Retrieval]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", choices=["codeqwen"], type=str, default="codeqwen"
    )
    parser.add_argument("-p", "--prompt", type=str, default="base")
    subparsers = parser.add_subparsers(
        description="the action to be performed", dest="subcommand", required=True
    )
    Query.command(subparsers)
    Retrieval.command(subparsers)
    StoreDataset.command(subparsers)

    args = parser.parse_args()
    subcommand = args.subcommand
    delattr(args, "subcommand")
    if subcommand == 'query':
        return Query(**vars(args))
    if subcommand == 'retrieval':
        return Retrieval(**vars(args))
    if subcommand == 'store-dataset':
        return StoreDataset(**vars(args))

    raise ValueError("no subcommand provided")