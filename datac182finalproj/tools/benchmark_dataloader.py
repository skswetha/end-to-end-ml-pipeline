from typing import Dict

import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dataloader.fault_csv_dataset import FaultCSVDataset, compute_data_preprocessor

################################################################################################################
# IMPORTANT: Do not modify the contents of this file! The autograder assumes that this file is unchanged.
################################################################################################################


def benchmark_dataloader(dataloader: torch.utils.data.DataLoader, num_steps: int = 100, num_warmup_steps: int = 50) -> Dict[str, float]:
    """Benchmarks an input dataloader.

    Args:
        dataloader:
        num_steps:

    Returns:
        benchmark_stats:
    """
    # initial warmup
    iter_dataloader = iter(dataloader)
    for _ in range(num_warmup_steps):
        next(iter_dataloader)
    # benchmark
    tic_all = time.time()
    durs_step = []
    for _ in range(num_steps):
        tic_step = time.time()
        next(iter_dataloader)
        durs_step.append(time.time() - tic_step)
    dur_all = time.time() - tic_all
    tput_mean = (num_steps * dataloader.batch_size) / dur_all
    latency_mean = np.mean(durs_step)
    latency_std = np.std(durs_step)
    return {
        "throughput_rows_per_sec": tput_mean,
        "total_time_secs": dur_all,
        "latency_secs_avg": latency_mean,
        "latency_secs_std": latency_std,
    }


def main_benchmark_dataloader():
    csvpath_dev_train = "data/split/faulty_commit_dev_train.csv"
    df_train_dev = pd.read_csv(csvpath_dev_train)
    tic_create_dataset = time.time()

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

    preprocessor = compute_data_preprocessor(
        dataframe=df_train_dev,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    dataset = FaultCSVDataset(
        dataframe=df_train_dev,
        preprocessor=preprocessor,
    )
    dur_create_dataset = time.time() - tic_create_dataset
    print(f"Created FaultCSVDataset ({dur_create_dataset} secs)")
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=False, num_workers=0)
    benchmark_stats = benchmark_dataloader(
        dataloader=dataloader,
        num_steps=1000,
    )
    print(f"benchmark_stats: {benchmark_stats}")

    # print a few batches
    for ind_row, row_dict in enumerate(dataloader):
        print(f"[{ind_row + 1}/{len(dataloader)}] row_dict={row_dict}")
        if ind_row >= 3:
            break

    return benchmark_stats


if __name__ == '__main__':
    main_benchmark_dataloader()
