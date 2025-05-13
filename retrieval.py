class Retrieval:
    pool_size: int
    data: str # Data file

    @staticmethod
    def command(subparsers):
        parser = subparsers.add_parser(
            "retrieval",
            description="Find the most similar assembly function from a set"
        )
        parser.add_argument("--pool-size", type=int, default=10)
        parser.add_argument("data", metavar="DATA-FILE", type=str)