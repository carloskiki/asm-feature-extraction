from dataclasses import dataclass
import context

@dataclass
class Query(context.Context):
    assembly: str  # File containing the assembly to analyze

    @staticmethod
    def command(subparsers):
        parser = subparsers.add_parser(
            "query", description="Query an LLM for the features of an assembly function"
        )
        parser.add_argument("assembly", metavar="ASM-FILE", type=str)
