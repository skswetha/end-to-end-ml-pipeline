import sklearn.metrics
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F

from evaluation.eval_structs import PredictionMetadata, OperatingPointMetrics, EvalMetrics


def predict_samples(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> PredictionMetadata:
    """Given a model and a dataloader, generates model predictions on each sample in dataloader.

    Tip: as the model.forward() returns logits, you will want to call `torch.nn.functional.sigmoid(logits)` to transform
        the logits to probability scores.

    Args:
        model:
        dataloader: Dataloader to run inference on.
            Take care that this produces batched inputs.
        device: Device to run inference on. Ex: CPU or cuda:0

    Returns:
        prediction_meta: Struct containing all inference results, along with ground truth labels.
            See `PredictionMetadata` for more details.

    """
    # Tip: remember to call `model.eval()`!
    model = model.eval()
    # BEGIN YOUR CODE
    labels_gt = []
    logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            features, labels = batch['features'].to(device), batch['label'].to(device)
            outputs = model(features)
            logits.extend(outputs.squeeze(1).cpu().numpy()) 
            labels_gt.extend(labels.cpu().numpy())

    return PredictionMetadata(
        labels_gt=np.array(labels_gt),
        pred_probs=np.array(logits),  # return logits
    )

def compute_eval_metrics(prediction_meta: PredictionMetadata) -> EvalMetrics:
    """Computes evaluation metrics.

    Args:
        prediction_meta: Contains model predictions and ground-truth labels.

    Returns:
        eval_metrics: evaluation metrics that we care about:
            precision-recall curve
            average precision
            operating-point metrics

    """
    # BEGIN YOUR CODE
    pred_probs = prediction_meta.pred_probs
    labels_gt = prediction_meta.labels_gt
    precisions,recalls,thresholds = precision_recall_curve(labels_gt, pred_probs)
    thresholds = thresholds.copy() # got error for tensors with negative strides and can be worked around by making copy of array

    average_precision = float(sklearn.metrics.average_precision_score(labels_gt, pred_probs))
    metrics_op = compute_operating_point_metrics_max_f1(precisions=torch.tensor(precisions), recalls=torch.tensor(recalls), thresholds=torch.tensor(thresholds))
   
    # END YOUR CODE
    return EvalMetrics(
        precisions=torch.tensor(precisions),
        recalls=torch.tensor(recalls),
        thresholds=torch.tensor(thresholds),
        average_precision=average_precision,
        metrics_op=metrics_op,
    )


def eval_model(
        model: torch.nn.Module,
        dataloader_test: torch.utils.data.DataLoader,
        device: torch.device
) -> EvalMetrics:
    """Given a model and test dataset, evaluates the model on the test dataset.

    Args:
        model:
        dataloader_test:
        device:

    Returns:
        eval_metrics:
    """
    prediction_meta = predict_samples(model, dataloader_test, device)
    return compute_eval_metrics(prediction_meta)


def compute_operating_point_metrics_max_f1(
        precisions: torch.Tensor,
        recalls: torch.Tensor,
        thresholds: torch.Tensor
) -> OperatingPointMetrics:
    """Calculate eval metrics at the operating point (aka threshold) that maximizes F1 score.
    Precisions/recalls/thresholds are computed from: sklearn.metrics.precision_recall_curve()

    Args:
        precisions:
        recalls:
        thresholds:

    Returns:
        operating_point_metrics: eval metrics at the threshold that maximizes the F1 score.

    """
    # BEGIN YOUR CODE
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    max_f1_idx = np.argmax(f1_scores)

    # END YOUR CODE
    return OperatingPointMetrics(
        precision_op=precisions[max_f1_idx].item(),
        recall_op=recalls[max_f1_idx].item(),
        f1_score_op=f1_scores[max_f1_idx].item(),
        threshold_op=thresholds[max_f1_idx].item(),
    )


def compute_operating_point_metrics_at_threshold(
        precisions: torch.Tensor,
        recalls: torch.Tensor,
        thresholds: torch.Tensor,
        threshold_op: float,
) -> OperatingPointMetrics:
    """Compute eval metrics at a specific input threshold.
    Precisions/recalls/thresholds are computed from: sklearn.metrics.precision_recall_curve()

    Args:
        precisions:
        recalls:
        thresholds:
        threshold_op: Threshold to calculate precision/recall/f1 for.
            Note that `threshold_op` will in general not be exactly in `thresholds`.
            In this case, use the precision/recall values corresponding to the first threshold
            in `thresholds` where `threshold >= threshold_op`.

    Returns:
        operating_point_metrics: Eval metrics at the given threshold (`threshold_op`).
    """
    # BEGIN YOUR CODE
    idx = (thresholds >= threshold_op).nonzero(as_tuple=True)[0]
    if idx.numel() > 0: 
        idx = idx[0].item()  

    precision_op = precisions[idx].item()
    recall_op = recalls[idx].item()
    f1_score_op = (2 * (precisions[idx] * recalls[idx]) / (precisions[idx] + recalls[idx] + 1e-5)).item()
    threshold_op = thresholds[idx].item()
    # END YOUR CODE
    return OperatingPointMetrics(
        precision_op=precision_op,
        recall_op=recall_op,
        f1_score_op=f1_score_op,
        threshold_op=threshold_op,
    )
