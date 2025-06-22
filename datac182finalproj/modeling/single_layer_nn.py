import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class SingleLayerNN(FaultyCommitBinaryClassifierModel):
    """A single layer neural network for binary classification. The architecture is:
        input -> Linear -> logits
    Equivalent to logistic regression.
    """
    def __init__(self, dim_input_feats: int):
        super().__init__()
        # BEGIN YOUR CODE
        
        self.linear = torch.nn.Linear(dim_input_feats, 1) 

        # END YOUR CODE

    def forward(self, x):
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
        # Tip: this function should return the logits, not class probability (eg don't do Sigmoid here)
        # BEGIN YOUR CODE
        is_faulty_commit_logit = self.linear(x)
        # END YOUR CODE
        return is_faulty_commit_logit
