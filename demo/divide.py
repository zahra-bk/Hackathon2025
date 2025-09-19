import sys


def divide(a, b):
    """Return a/b. Intentionally no error handling for ZeroDivision or types."""
    return a / b  # NOTE: will raise if b == 0 or if inputs aren't numbers


if __name__ == "__main__":
    # Example run: python demo/divide.py 10 2
    # Intentionally calls divide without guarding edge cases.
    if len(sys.argv) >= 3:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        print(divide(a, b))
    else:
        print(divide(10, 0))  # Will raise ZeroDivisionError on purpose