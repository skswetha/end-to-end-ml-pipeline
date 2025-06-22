import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class AlwaysPositiveBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """Model that always outputs the positive class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given an input sample, return the predicted logits for the positive class ("is faulty?").
        Note that, as we are approaching this as a binary classification problem, we don't output logits for the
        negative class ("is not faulty").

        Args:
            x: Input features.
                shape=[batchsize, dim_feats].
                For details on format, see: `FaultCSVDataset`.

        Returns:
            is_faulty_commit_logit:
                shape=[batchsize, 1]

        """
        # BEGIN YOUR CODE
        batch_size = x.shape[0]
        return torch.ones(batch_size, 1, device=x.device)

        # END YOUR CODE


class AlwaysNegativeBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """Model that always outputs the negative class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given an input sample, return the predicted logits for the positive class ("is faulty?").
        Note that, as we are approaching this as a binary classification problem, we don't output logits for the
        negative class ("is not faulty").

        Args:
            x: Input features.
                shape=[batchsize, dim_feats].
                For details on format, see: `FaultCSVDataset`.

        Returns:
            is_faulty_commit_logit:
                shape=[batchsize, 1]

        """
        # BEGIN YOUR CODE
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 1, device=x.device)
        # END YOUR CODE
