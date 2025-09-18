from collections.abc import Callable

import jsonargparse
from typing import Any


def parse_args(func: Callable[..., Any]) -> jsonargparse.Namespace:
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(func)
    return parser.parse_args()
