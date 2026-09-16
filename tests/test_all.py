from pathlib import Path
import unittest


def load_tests(loader: unittest.TestLoader, tests, pattern):
    unit_tests = Path(__file__).parent / "unit_tests"
    return loader.discover(
        str(unit_tests),
        pattern="*.py",
    )
