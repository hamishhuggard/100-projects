from typing import Callable
import functools


def logger(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__}(*{args}, **{kwargs}) -> {result})")
        return result
    return wrapper
