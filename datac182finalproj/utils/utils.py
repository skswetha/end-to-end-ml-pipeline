import os
from typing import Tuple, List, Optional, Dict, Any

from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
import pandas as pd
import torch

from dataloader.fault_csv_dataset import compute_data_preprocessor, FaultCSVDataset
from evaluation.eval_structs import EvalMetrics
from trainer.train_structs import TrainMetadata


################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


def save_model(model: torch.nn.Module, outpath: str):
    out_dict = {
        "model_state_dict": model.state_dict(),
    }
    os.makedirs(os.path.split(outpath)[0], exist_ok=True)
    torch.save(out_dict, outpath)
    print(f"Saved model weights to: {outpath}")


def save_auto_grader_data(auto_grader_data: Dict[str, Any], outpath: str):
    os.makedirs(os.path.split(outpath)[0], exist_ok=True)
    torch.save(
        {'auto_grader_data': auto_grader_data},
        outpath,
    )


def add_entry_to_auto_grader_data(auto_grader_data: Dict[str, Any], keys: List[str], value: Any):
    """Adds a desired entry to the auto_grader_data. Allows arbitrary number of nested keys in a
    `defaultdict(dict)`-style.
    Mutates input auto_grader_data.

    >>> auto_grader_data = {}
    >>> add_entry_to_auto_grader_data(auto_grader_data, ["key1", "key2"], 42)
    >>> auto_grader_data
    {"key1": {"key2": 42}}

    Args:
        auto_grader_data:
        keys:
        value:

    Returns:

    """
    cur_dict = auto_grader_data
    for key in keys[:-1]:
        if key not in cur_dict:
            cur_dict[key] = {}
        cur_dict = cur_dict[key]
    cur_dict[keys[-1]] = value


def count_model_parameters(model: torch.nn.Module) -> int:
    """Counts number of parameters of the input model.

    Args:
        model:

    Returns:
        num_params:
    """
    return sum(p.numel() for p in model.parameters())


def create_dataloaders(
        numerical_features: List[str],
        categorical_features: List[str],
        batchsize_train: int = 256,
        batchsize_val: int = 256,
        batchsize_test: int = 256,
        csvpath_dev_train: str = "data/split/faulty_commit_dev_train.csv",
        csvpath_dev_val: str = "data/split/faulty_commit_dev_val.csv",
        csvpath_test: str = "data/split/faulty_commit_test.csv",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, ColumnTransformer]:
    """Creates the train/val/test dataloaders.
    Convenience function.

    Args:
        numerical_features:
        categorical_features:
        batchsize_train:
        batchsize_val:
        batchsize_test:
        csvpath_dev_train:
        csvpath_dev_val:
        csvpath_test:

    Returns:
        train_dataloader:
        val_dataloader:
        test_dataloader:
        preprocessor:
    """
    df_commits_dev_train = pd.read_csv(csvpath_dev_train)
    df_commits_dev_val = pd.read_csv(csvpath_dev_val)
    df_commits_test = pd.read_csv(csvpath_test)

    preprocessor = compute_data_preprocessor(
        dataframe=df_commits_dev_train,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    dataset_train = FaultCSVDataset(
        dataframe=df_commits_dev_train,
        preprocessor=preprocessor,
    )
    dataset_val = FaultCSVDataset(
        dataframe=df_commits_dev_val,
        preprocessor=preprocessor,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batchsize_train,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batchsize_val,
        shuffle=False,
        num_workers=0,
    )

    dataset_test = FaultCSVDataset(
        dataframe=df_commits_test,
        preprocessor=preprocessor,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batchsize_test,
        shuffle=False,
        num_workers=0,
    )

    dim_input_feats = train_dataloader.dataset[0]["features"].shape[0]
    print(f"Loaded train dataset from {csvpath_dev_train}, num_train_rows={len(dataset_train)} dim_input_feats: {dim_input_feats}")

    return train_dataloader, val_dataloader, test_dataloader, preprocessor


def plot_train_eval_metrics(train_meta: TrainMetadata, test_eval_metrics: Optional[EvalMetrics] = None, outpath_fig: Optional[str] = None) -> plt.Figure:
    """Plots train/val/test metrics.
    Note that val/test metrics are allowed to be missing: these plots will be empty.

    Args:
        train_meta:
        test_eval_metrics:
        outpath_fig: If given, save the figure to disk at this outpath.
    Returns:

    """
    # Plotting
    fig, axd = plt.subplot_mosaic([['train_loss', 'train_acc', 'train_acc_pos_class', 'val_ap'], ["pr_val", "pr_test", ".", "."]], figsize=(15, 5))

    axd['train_loss'].plot(train_meta.losses)
    axd['train_loss'].set_xlabel("Step")
    axd['train_loss'].set_ylabel("Train Loss")
    axd['train_loss'].set_title(f'Train Loss (num_params={train_meta.num_model_parameters})')

    axd['train_acc'].plot(train_meta.train_accs)
    axd['train_acc'].set_xlabel("Step")
    axd['train_acc'].set_ylabel("Train Accuracy")
    axd['train_acc'].set_title('Train Accuracy')

    axd['train_acc_pos_class'].plot(train_meta.train_accs_pos_class)
    axd['train_acc_pos_class'].set_xlabel("Step")
    axd['train_acc_pos_class'].set_ylabel("Train Accuracy (Positive class)")
    axd['train_acc_pos_class'].set_title('Train Accuracy (Positive class)')

    axd['val_ap'].plot([val_eval_meta.average_precision for val_eval_meta in train_meta.all_val_eval_metrics])
    axd['val_ap'].set_xlabel("Epoch")
    axd['val_ap'].set_ylabel("Val AP")
    axd['val_ap'].set_title('Val AP')

    # Precision/recall on validation set for final epoch
    axd['pr_val'].plot(train_meta.all_val_eval_metrics[-1].recalls, train_meta.all_val_eval_metrics[-1].precisions)
    axd['pr_val'].set_xlabel("Recall")
    axd['pr_val'].set_ylabel("Precision")
    axd['pr_val'].set_title(f'PR curve (val) AP={train_meta.all_val_eval_metrics[-1].average_precision}')

    if test_eval_metrics is not None:
        axd['pr_test'].plot(test_eval_metrics.recalls, test_eval_metrics.precisions)
        axd['pr_test'].set_xlabel("Recall")
        axd['pr_test'].set_ylabel("Precision")
        axd['pr_test'].set_title(f'PR curve (test) AP={test_eval_metrics.average_precision}')

    fig.tight_layout()

    if outpath_fig:
        print(f"Saving figure to: {outpath_fig}")
        os.makedirs(os.path.split(outpath_fig)[0], exist_ok=True)
        fig.savefig(outpath_fig)
    return fig
