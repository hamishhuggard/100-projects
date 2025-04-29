from calculator.operations import add, subtract

def add_test():
    assert add(1, 2) == 3
    assert add(-1, -2) == -3

def subtract_test():
    assert subtract(1, 2) == -1
    assert subtract(-1, -2) == 1
