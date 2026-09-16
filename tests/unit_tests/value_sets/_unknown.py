import unittest

from problab.value_sets._unknown import _UnknownValueSet


class UnknownValueSetTests(unittest.TestCase):

    def test_representation_is_stable(self):
        self.assertEqual(repr(_UnknownValueSet()), "UNKNOWN_VALUE_SET")


if __name__ == "__main__":
    unittest.main()
