import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.utils import create_dataloaders
from evaluation.offline_eval import eval_model
from modeling.model_random import RandomBinaryClassifier
from modeling.trivial_models import AlwaysPositiveBinaryClassifier, AlwaysNegativeBinaryClassifier

import torch

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


def main_evaluate_model_random():
    numerical_features = [
        'modifications_count',
        'additions_count',
        'deletions_count',
        'hour',
        'day',
        'repo_id',
    ]
    categorical_features = [
        'author_name',
        'author_email',
        'committer_name',
        'committer_email',
        'ext'
    ]
    _, _, test_dataloader, _ = create_dataloaders(numerical_features, categorical_features)

    model = RandomBinaryClassifier(prob_predict_positive=0.5)

    # Test
    print(f"======== Test metrics (RandomBinaryClassifier) ========")
    test_eval_metrics = eval_model(model, test_dataloader, device=torch.device("cpu"))
    print(f"(Test) {test_eval_metrics.summarize_to_str()}")
    return test_eval_metrics


def main_evaluate_always_pos_neg():
    numerical_features = [
        'modifications_count',
        'additions_count',
        'deletions_count',
        'hour',
        'day',
        'repo_id',
    ]
    categorical_features = [
        'author_name',
        'author_email',
        'committer_name',
        'committer_email',
        'ext'
    ]
    _, _, test_dataloader, _ = create_dataloaders(numerical_features, categorical_features)

    model_pos = AlwaysPositiveBinaryClassifier()

    # Test
    print(f"======== Test metrics (AlwaysPositiveBinaryClassifier) ========")
    test_eval_metrics_always_pos = eval_model(model_pos, test_dataloader, device=torch.device("cpu"))
    print(f"(Test) {test_eval_metrics_always_pos.summarize_to_str()}")

    model_neg = AlwaysNegativeBinaryClassifier()

    print(f"======== Test metrics (AlwaysNegativeBinaryClassifier) ========")
    test_eval_metrics_always_neg = eval_model(model_neg, test_dataloader, device=torch.device("cpu"))
    print(f"(Test) {test_eval_metrics_always_neg.summarize_to_str()}")
    return test_eval_metrics_always_pos, test_eval_metrics_always_neg


if __name__ == '__main__':
    main_evaluate_model_random()
    main_evaluate_always_pos_neg()
