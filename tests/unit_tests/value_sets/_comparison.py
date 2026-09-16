import unittest

import numpy as np

from problab.value_sets._comparison import _objects_equal


class ObjectComparisonTests(unittest.TestCase):

    def test_compares_scalars_and_array_like_values(self):
        self.assertTrue(_objects_equal("red", "red"))
        self.assertFalse(_objects_equal("red", "blue"))
        self.assertTrue(_objects_equal(np.array([1, 2]), np.array([1, 2])))
        self.assertFalse(_objects_equal(np.array([1, 2]), np.array([1, 3])))

    def test_returns_false_when_equality_raises(self):
        class Uncomparable:
            def __eq__(self, other):
                raise RuntimeError("cannot compare")

        self.assertFalse(_objects_equal(Uncomparable(), Uncomparable()))


if __name__ == "__main__":
    unittest.main()
