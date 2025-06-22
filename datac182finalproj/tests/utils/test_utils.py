import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from utils.utils import add_entry_to_auto_grader_data

import unittest


class TestAddEntryToAutoGraderData(unittest.TestCase):
    """add_entry_to_auto_grader_data()"""
    def test_simple(self):
        auto_grader_data = {}
        add_entry_to_auto_grader_data(auto_grader_data, ["key1", "key2"], 42)
        self.assertEqual(auto_grader_data, {"key1": {"key2": 42}})

        add_entry_to_auto_grader_data(auto_grader_data, ["key1", "key3", "key4"], 43)
        self.assertEqual(auto_grader_data, {"key1": {"key2": 42, "key3": {"key4": 43}}})

        add_entry_to_auto_grader_data(auto_grader_data, ["key1", "key2"], 44)
        self.assertEqual(auto_grader_data, {"key1": {"key2": 44, "key3": {"key4": 43}}})

    def test_simple2(self):
        auto_grader_data = {}
        add_entry_to_auto_grader_data(auto_grader_data, ["output", "part", "random", "metrics"], 42)
        self.assertEqual(auto_grader_data, {"output": {"part": {"random": {"metrics": 42}}}})

        add_entry_to_auto_grader_data(auto_grader_data, ["output", "part", "always_pos", "metrics"], 43)
        self.assertEqual(auto_grader_data, {"output": {"part": {"random": {"metrics": 42}, "always_pos": {"metrics": 43}}}})

        add_entry_to_auto_grader_data(auto_grader_data, ["output", "part", "always_neg", "metrics"], 44)
        self.assertEqual(auto_grader_data, {"output": {"part": {"random": {"metrics": 42}, "always_pos": {"metrics": 43}, "always_neg": {"metrics": 44}}}})


if __name__ == '__main__':
    unittest.main()
