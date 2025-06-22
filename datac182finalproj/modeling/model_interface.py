import torch

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


class FaultyCommitBinaryClassifierModel(torch.nn.Module):
    """Binary classification model interface for the "is faulty commit?" problem.
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
        # Models should implement this
        raise NotImplementedError()
