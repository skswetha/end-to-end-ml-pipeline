import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any

from evaluation.eval_structs import EvalMetrics

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


@dataclass
class TrainMetadata:
    # Training loss values after each train iteration
    losses: List[float]
    # Batch accuracies after each train iteration
    train_accs: List[float]
    # Batch accuracies after each log step (just for positive class)
    train_accs_pos_class: List[float]
    # Validation eval metrics after each training epoch.
    all_val_eval_metrics: List[EvalMetrics]

    # Number of model parameters
    num_model_parameters: int

    def to_pydict(self) -> Dict[str, Any]:
        """Convert to a python-friendly dict format.
        Similar in spirit to dataclasses.asdict(self), but also convert non-built-in types like
        torch.Tensor to similar built-in types.

        Returns:

        """
        return dataclasses.asdict(self)
