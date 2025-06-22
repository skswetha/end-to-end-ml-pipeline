import torch

import unittest

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


def assert_equal(tester: unittest.TestCase, a, b):
    if isinstance(a, torch.Tensor):
        tester.assertTrue(
            torch.all(a == b),
            msg=f"Expected equal: {a} vs {b}"
        )
    else:
        tester.assertEqual(a, b)
