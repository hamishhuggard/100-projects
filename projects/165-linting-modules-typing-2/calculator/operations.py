from .decorators import logger


@logger
def add(a: float, b: float) -> float:
    return a + b


@logger
def subtract(a: float, b: float) -> float:
    return a - b


def untyped(a):
    return a
