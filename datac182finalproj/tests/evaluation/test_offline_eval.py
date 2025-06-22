import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from typing import List
import unittest

import torch.utils.data
import numpy as np

from modeling.trivial_models import AlwaysPositiveBinaryClassifier
from evaluation.offline_eval import predict_samples, compute_eval_metrics
from evaluation.eval_structs import PredictionMetadata, OperatingPointMetrics, EvalMetrics


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[float], labels: List[int]):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __getitem__(self, idx: int):
        return {
            "features": self.data[idx],
            "label": self.labels[idx],
        }

    def __len__(self):
        return len(self.data)


class TestPredictSamples(unittest.TestCase):
    """predict_samples()"""
    def test_batchsize_1(self):
        model_trigger_happy = AlwaysPositiveBinaryClassifier()
        data = [1.0, 2.0, 3.0, 4.0]
        labels = [0, 1, 1, 0]
        dataset = TestDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

        prediction_meta = predict_samples(model_trigger_happy, dataloader, device=torch.device("cpu"))
        self.assertEqual(prediction_meta.labels_gt.tolist(), labels)
        self.assertEqual(list(prediction_meta.pred_probs.shape), [len(data)])

    def test_batchsize_3(self):
        model_trigger_happy = AlwaysPositiveBinaryClassifier()
        data = [1.0, 2.0, 3.0, 4.0]
        labels = [0, 1, 1, 0]
        dataset = TestDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3, num_workers=0, shuffle=False)

        prediction_meta = predict_samples(model_trigger_happy, dataloader, device=torch.device("cpu"))
        self.assertEqual(prediction_meta.labels_gt.tolist(), labels)
        self.assertEqual(list(prediction_meta.pred_probs.shape), [len(data)])

    def test_batchsize_4(self):
        model_trigger_happy = AlwaysPositiveBinaryClassifier()
        data = [1.0, 2.0, 3.0, 4.0]
        labels = [0, 1, 1, 0]
        dataset = TestDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)

        prediction_meta = predict_samples(model_trigger_happy, dataloader, device=torch.device("cpu"))
        self.assertEqual(prediction_meta.labels_gt.tolist(), labels)
        self.assertEqual(list(prediction_meta.pred_probs.shape), [len(data)])


class TestComputeEvalMetrics(unittest.TestCase):
    """compute_eval_metrics()"""
    def test_simple(self):
        prediction_meta = PredictionMetadata(
            pred_probs=torch.tensor([0.95, 0.9, 0.6, 0.5, 0.3], dtype=torch.float64),
            labels_gt=torch.tensor([1, 1, 0, 1, 0], dtype=torch.float64)
        )
        eval_metrics = compute_eval_metrics(prediction_meta)
        eval_metrics_expected = EvalMetrics(
            precisions=torch.tensor([0.6, 0.75, 0.6666666666, 1.0, 1.0, 1.0], dtype=torch.float64),
            recalls=torch.tensor([1.0, 1.0, 0.6666666666, 0.6666666666, 0.333333333, 0.0], dtype=torch.float64),
            # shape=[num_rows]
            thresholds=torch.tensor([0.3, 0.5, 0.6, 0.9, 0.95], dtype=torch.float64),

            # threshold-agnostic score. similar (but not exactly) to area-under-the-curve of the precision-recall curve.
            # Tip: see: sklearn.metrics.average_precision_score
            average_precision=0.916666666666,

            # Metrics at a specific operating point (threshold that maximizes f1)
            metrics_op=OperatingPointMetrics(
                # precision/recall/f1 at the given classification threshold
                precision_op=0.75,
                recall_op=1.0,
                f1_score_op=0.8571428571428571,
                # Classification threshold
                threshold_op=0.5,
            ),
        )

        self.assertTrue(np.allclose(eval_metrics.precisions.numpy(), eval_metrics_expected.precisions.numpy(), equal_nan=True))
        self.assertTrue(np.allclose(eval_metrics.recalls.numpy(), eval_metrics_expected.recalls.numpy(), equal_nan=True))
        self.assertTrue(np.allclose(eval_metrics.thresholds.numpy(), eval_metrics_expected.thresholds.numpy(), equal_nan=True))
        self.assertAlmostEqual(eval_metrics.average_precision, eval_metrics_expected.average_precision)
        self.assertAlmostEqual(eval_metrics.metrics_op.precision_op, eval_metrics_expected.metrics_op.precision_op)
        self.assertAlmostEqual(eval_metrics.metrics_op.recall_op, eval_metrics_expected.metrics_op.recall_op)
        self.assertAlmostEqual(eval_metrics.metrics_op.f1_score_op, eval_metrics_expected.metrics_op.f1_score_op)
        self.assertAlmostEqual(eval_metrics.metrics_op.threshold_op, eval_metrics_expected.metrics_op.threshold_op)


if __name__ == '__main__':
    unittest.main()
