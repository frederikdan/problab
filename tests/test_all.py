from pathlib import Path
import unittest


def load_tests(loader: unittest.TestLoader, tests, pattern):
    test_root = Path(__file__).parent
    suites = unittest.TestSuite()

    for directory_name in ("unit_tests", "integration_tests", "api_tests"):
        suites.addTests(
            loader.discover(
                str(test_root / directory_name),
                pattern="*.py",
            )
        )

    return suites
