class Retrieval:
    pool_size: int

    @staticmethod
    def command(subparsers):
        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set"
        )
        parser.add_argument("assembly", metavar="ASM-FILE", type=str)