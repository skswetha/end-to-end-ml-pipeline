from typing import Tuple, Optional
import os

from matplotlib import pyplot as plt

from consts import STUDENT_SUBMISSION_OUTDIR
from evaluation.eval_structs import EvalMetrics, OperatingPointMetrics
from train import train_and_eval
from trainer.train_structs import TrainMetadata
from utils.utils import create_dataloaders, plot_train_eval_metrics

import torch
import torch.optim as optim
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, dim_input_feats, hidden_layer_sizes, dropout_rate=0.3):
        super(ModelNew, self).__init__()
        layers = []
        input_size = dim_input_feats
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_and_eval_pt5() -> Tuple[TrainMetadata, Optional[Tuple[EvalMetrics, OperatingPointMetrics]], plt.Figure]:
    """Train and evaluate your "Improved model" for (Part 5).
    Your goal is to achieve a test AP >= 0.025.

    Returns:
        (same outputs as: train_and_eval())
        plt.Figure:

    """
    # Tip: while we've provided you with some skeleton code here, you're welcome to change anything here (eg modifying
    #   dataloaders/optimizers/trainer/etc) as long as you return the same output types.
    # Feel free to modify batchsizes, train_total_num_epochs
    train_batchsize = 1024
    val_batchsize = 1024
    test_batchsize = 1024
    train_total_num_epochs = 10
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

    # Create train/val/test dataloaders
    train_dataloader, val_dataloader, test_dataloader, _ = create_dataloaders(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        batchsize_train=train_batchsize,
        batchsize_val=val_batchsize,
        batchsize_test=test_batchsize,
    )
    dim_input_feats = train_dataloader.dataset[0]["features"].shape[0]
    # Feel free to customize the optimizer (and criterion if you wish, but may not be as useful to modify this)
    model = ModelNew(dim_input_feats=dim_input_feats, hidden_layer_sizes=[256, 128, 64], dropout_rate=0.3)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    # Instantiate your improved model (Part 5)
    # BEGIN YOUR CODE
    # END YOUR CODE
    model = model
    train_metadata, (test_eval_metrics, test_metrics_op) = train_and_eval(
        train_batchsize=train_batchsize,
        val_batchsize=val_batchsize,
        test_batchsize=test_batchsize,
        train_total_num_epochs=train_total_num_epochs,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )
    fig = plot_train_eval_metrics(train_metadata, test_eval_metrics=test_eval_metrics,
                                  outpath_fig=os.path.join(STUDENT_SUBMISSION_OUTDIR, "main_train_pt5.png"))
    return train_metadata, (test_eval_metrics, test_metrics_op), fig


def main():
    train_metadata, (test_eval_metrics, test_metrics_op), fig = train_and_eval_pt5()
    plt.show()


if __name__ == '__main__':
    main()
