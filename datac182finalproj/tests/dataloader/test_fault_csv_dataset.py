import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import unittest

import pandas as pd
import torch
import numpy as np
from numpy.testing import assert_allclose

from dataloader.fault_csv_dataset import compute_data_preprocessor, FaultCSVDataset
from utils.test_utils import assert_equal


class TestComputeDataPreprocessor(unittest.TestCase):
    """compute_data_preprocessor()"""
    def test_simple(self):
        df_train = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charles", "Daria"],
            "Age": [28, 20, 32, 19],
            "IgnoreFeat": ["meow", "bark", "meow", "meow"],
        })
        numerical_features = ['Age']
        categorical_features = ['Name']

        preprocessor = compute_data_preprocessor(
            dataframe=df_train,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        preprocessor.fit(df_train)

        df_train_preproc = preprocessor.transform(df_train)
        self.assertEqual(df_train_preproc.shape, (4, 5))

        age_np = df_train["Age"].to_numpy()
        age_mn = np.mean(age_np)
        age_std = np.std(age_np, ddof=0)

        df_test = pd.DataFrame({
            "Name": ["Alice", "Eve"],
            "Age": [28, 35],
            "IgnoreFeat": ["bark", "meow"],
        })
        df_test_preproc = preprocessor.transform(df_test)
        self.assertEqual(df_test_preproc.shape, (2, 5))
        assert_allclose(df_test_preproc[0, :], df_train_preproc[0, :])
        self.assertAlmostEqual(df_test_preproc[1, 0], (35 - age_mn) / age_std)
        # "Eve" is missing, so should be assigned to all-0 one-hot encoding vector
        self.assertTrue(np.all(df_test_preproc[1, 1:] == 0))
        print("Passed tests!")


class TestFaultCsvDataset(unittest.TestCase):
    def test_simple(self):
        try:
            csvpath_dev_train = "data/split/faulty_commit_dev_train.csv"
            df_train_dev = pd.read_csv(csvpath_dev_train)
        except:
            # Gradescope autograder needs a different path
            csvpath_dev_train = "/autograder/source/solution/data/split/faulty_commit_dev_train.csv"
            df_train_dev = pd.read_csv(csvpath_dev_train)

        numerical_features = [
            'modifications_count',
        ]

        categorical_features = [
            'author_name',
        ]

        preprocessor = compute_data_preprocessor(
            dataframe=df_train_dev,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        dataset = FaultCSVDataset(
            dataframe=df_train_dev,
            preprocessor=preprocessor,
        )

        def construct_one_hot(num_categories: int, index: int) -> torch.Tensor:
            out = torch.zeros([num_categories], dtype=torch.float32)
            out[index] = 1.0
            return out

        row_0 = dataset[0]
        row_0_categorical_expected = construct_one_hot(189, 127)
        row_0_expected = {
            "label": 0,
            "features": torch.tensor([-0.32034456729888916] + row_0_categorical_expected.tolist(), dtype=torch.float32)
        }
        assert_equal(self, row_0["label"], row_0_expected["label"])
        assert_allclose(row_0["features"].numpy(), row_0_expected["features"].numpy())


if __name__ == '__main__':
    unittest.main()
