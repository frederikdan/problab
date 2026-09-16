import unittest

from problab.random_variables._config import DEF_MAX_GRAPH_SIZE


class RandomVariableConfigTests(unittest.TestCase):

    def test_default_graph_size_is_positive(self):
        self.assertEqual(DEF_MAX_GRAPH_SIZE, 100)
        self.assertGreater(DEF_MAX_GRAPH_SIZE, 0)


if __name__ == "__main__":
    unittest.main()
