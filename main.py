import argparse
from query import Query
from bogus import Bogus
from retrieval import Retrieval
from batch_query import BatchQuery

COMMANDS = {
    'query': Query,
    'retrieval': Retrieval,
    'bogus': Bogus,
    'batch-query': BatchQuery
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", choices=["codeqwen"], type=str, default="codeqwen"
    )
    parser.add_argument("-p", "--prompt", type=str, default="v2")
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