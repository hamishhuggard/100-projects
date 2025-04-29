from calculator.operations import add, subtract

def main() -> None:
    x, y = 10, 5
    print("Add", add(x, y))
    print("Sub", subtract(x, y))

if __name__ == "__main__":
    main()
