from argparse import ArgumentTypeError

def platform_parser(s):
    # If input looks like key:value,key:value,...
    if ":" in s:
        try:
            return [tuple(p.split(":", 1)) for p in s.split(",")]
        except ValueError as e:
            raise ArgumentTypeError("Malformed key:value pair.") from e
    else:
        # Just treat it as a plain string
        return s


def optimization_parser(s):
    # Try to parse as a single int
    try:
        return int(s)
    except ValueError:
        pass  # Not a single int

    # Try to parse as list of int:int pairs
    try:
        pairs = []
        for p in s.split(","):
            k, v = p.split(":", 1)
            pairs.append((int(k), int(v)))
        return pairs
    except Exception as e:
        raise ArgumentTypeError(
            "Expected an int or comma-separated int:int pairs"
        ) from e

