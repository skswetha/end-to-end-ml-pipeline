import argparse
import time
import os
from typing import Tuple, Optional

from matplotlib import pyplot as plt
import torch

from consts import STUDENT_SUBMISSION_OUTDIR
from evaluation.offline_eval import eval_model, compute_operating_point_metrics_at_threshold
from evaluation.eval_structs import OperatingPointMetrics, EvalMetrics
from modeling.single_layer_nn import SingleLayerNN
from trainer.trainer import ClassificationTrainer
from trainer.train_structs import TrainMetadata
from utils.utils import count_model_parameters, create_dataloaders, plot_train_eval_metrics

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


def train_and_eval(
        train_batchsize: int,
        val_batchsize: int,
        test_batchsize: int,
        train_total_num_epochs: int,
        skip_val: bool = False,
        skip_test: bool = False,
        model: Optional[torch.nn.Module] = None,
        criterion: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[TrainMetadata, Optional[Tuple[EvalMetrics, OperatingPointMetrics]]]:
    """Trains and (optionally) evaluates model on a test set.

    Args:
        train_batchsize:
        val_batchsize:
        test_batchsize:
        train_total_num_epochs:
        skip_val:
        skip_test:
        model: If given, use the provided model. Otherwise, default to SingleLayerNN.
        criterion: If given, use the provided criterion. Otherwise, default to BCEWithLogitsLoss
        optimizer: If given, use the provided optimizer. Otherwise, default to AdamW

    Returns:
        train_metadata:
        (test_eval_metrics, test_metrics_op):
    """
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

    # Create model
    if model is None:
        model = SingleLayerNN(
            dim_input_feats=dim_input_feats,
        )
    print(f"Model: {count_model_parameters(model)} params")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device={device}")
    print(f"(pre model.to) GPU max_memory_allocated: {torch.cuda.max_memory_allocated() / 1e6} MB")
    model = model.to(device=device)
    print(f"(post model.to) GPU max_memory_allocated: {torch.cuda.max_memory_allocated() / 1e6} MB")

    # Create loss and optimizer
    # Binary cross entropy
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=1e-9)

    # Instantiate trainer, enter train loop
    trainer = ClassificationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_data_loader=train_dataloader,
        val_data_loader=val_dataloader,
        device=device,
        log_every_n_batches=20,
        skip_val=skip_val,
    )
    tic_train = time.time()
    train_metadata = trainer.train(total_num_epochs=train_total_num_epochs)
    dur_train = time.time() - tic_train
    print(f"Finished training! {dur_train} secs (total_num_epochs={train_total_num_epochs})")

    if skip_test:
        print(f"Skipping test (skip_test={skip_test})")
        return train_metadata, None

    # Print validation metrics
    print(f"======== Validation metrics ========")
    for ind_epoch, val_eval_metrics in enumerate(train_metadata.all_val_eval_metrics):
        print(f"[epoch={ind_epoch + 1}/{train_total_num_epochs}] (val) {val_eval_metrics.summarize_to_str()}")

    # Evaluate on test set
    print(f"======== Test metrics ========")
    test_eval_metrics = eval_model(model, test_dataloader, device=device)
    print(f"(Test) {test_eval_metrics.summarize_to_str()}")

    # Use operating point threshold from validation set on test set
    print(f"Using validation set operating point threshold T={train_metadata.all_val_eval_metrics[-1].metrics_op.threshold_op}:")
    test_metrics_op = compute_operating_point_metrics_at_threshold(
        precisions=test_eval_metrics.precisions,
        recalls=test_eval_metrics.recalls,
        thresholds=test_eval_metrics.thresholds,
        threshold_op=train_metadata.all_val_eval_metrics[-1].metrics_op.threshold_op,
    )
    print(f"(Test) {test_metrics_op.summarize_to_str()}")

    return train_metadata, (test_eval_metrics, test_metrics_op)


def main_train_pt2c():
    # by default, train_and_eval() uses SingleLayerNN
    train_meta_pt2c, _ = train_and_eval(
        train_batchsize=256,
        val_batchsize=256,
        test_batchsize=256,
        train_total_num_epochs=5,
        skip_val=True,
        skip_test=True,
    )
    fig = plot_train_eval_metrics(train_meta_pt2c, test_eval_metrics=None, outpath_fig=os.path.join(STUDENT_SUBMISSION_OUTDIR, "main_train_pt2c.png"))
    plt.show()
    return train_meta_pt2c, fig


def main_train_and_eval_pt4():
    # by default, train_and_eval() uses SingleLayerNN
    train_meta, (test_eval_metrics, test_metrics_op) = train_and_eval(
        train_batchsize=1024,
        val_batchsize=1024,
        test_batchsize=1024,
        train_total_num_epochs=10,
        skip_val=False,
        skip_test=False,
    )
    fig = plot_train_eval_metrics(train_meta, test_eval_metrics=test_eval_metrics,
                                  outpath_fig=os.path.join(STUDENT_SUBMISSION_OUTDIR, "main_train_pt4.png"))
    return train_meta, (test_eval_metrics, test_metrics_op), fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_total_num_epochs", type=int,
        default=10,
    )
    ap.add_argument(
        "--train_batchsize", type=int,
        default=1024,
    )
    ap.add_argument(
        "--val_batchsize", type=int,
        default=1024,
    )
    ap.add_argument(
        "--test_batchsize", type=int,
        default=1024,
    )
    ap.add_argument(
        "--skip_val", action="store_true",
        help="Skip validation."
    )
    ap.add_argument(
        "--skip_test", action="store_true",
        help="Skip test."
    )
    args = ap.parse_args()
    train_metadata, _test_things = train_and_eval(
        train_batchsize=args.train_batchsize,
        val_batchsize=args.val_batchsize,
        test_batchsize=args.test_batchsize,
        train_total_num_epochs=args.train_total_num_epochs,
        skip_val=args.skip_val,
        skip_test=args.skip_test,
    )
    if not args.skip_test:
        test_eval_metrics, test_metrics_op = _test_things
    else:
        test_eval_metrics, test_metrics_op = None, None
    plot_train_eval_metrics(train_metadata, test_eval_metrics=test_eval_metrics, outpath_fig=os.path.join(STUDENT_SUBMISSION_OUTDIR, "plot.png"))
    plt.show()


if __name__ == '__main__':
    main()
    # main_train_pt2c()
