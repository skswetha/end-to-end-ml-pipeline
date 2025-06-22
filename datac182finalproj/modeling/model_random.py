import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class RandomBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """Classifier that returns positive/negative class based on flipping a (biased) coin.
    """
    def __init__(self, prob_predict_positive: float = 0.5):
        """

        Args:
            prob_predict_positive:
                Value between [0.0, 1.0].
                0.0: never predict positive
                1.0: always predict positive
        """
        super().__init__()
        self.prob_predict_positive = prob_predict_positive

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
        logit = torch.log(torch.tensor(self.prob_predict_positive) / (1 - self.prob_predict_positive))
        logits = torch.empty(batch_size, 1, device=x.device).uniform_(-logit, logit)
        
        return logits

        # END YOUR CODE
