import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from utils.utils import count_model_parameters
from modeling.single_layer_nn import SingleLayerNN

import torch

import unittest


class TestSingleLayerNN(unittest.TestCase):
    """SingleLayerNN"""
    def test_simple(self):
        dim_input_feats = 5
        model = SingleLayerNN(dim_input_feats=dim_input_feats)
        num_params = count_model_parameters(model)
        self.assertEqual(num_params, dim_input_feats + 1)  # [dx1] matrix, [1] bias

        bs = 8
        x = torch.zeros([bs, dim_input_feats], dtype=torch.float32)
        out = model(x)
        self.assertEqual(out.shape, (bs, 1))


if __name__ == '__main__':
    unittest.main()
