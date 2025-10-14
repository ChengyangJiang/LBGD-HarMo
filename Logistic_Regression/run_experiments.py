import argparse
import os
import numpy as np
from parameters import Parameters
from experiment import run_experiment


def build_params(method, d, n, n_cores, num_epoch, split_name, random_seed):
    """Helper to construct Parameters for different methods."""
    if method == "DSGD":
        return Parameters(
            name="DSGD",
            num_epoch=num_epoch,
            lr_type='decay',
            initial_lr=0.1,
            tau=d,
            regularizer=1 / n,
            quantization='full',
            n_cores=n_cores,
            method='plain',
            split_data_random_seed=random_seed,
            distribute_data=True,
            split_data_strategy=split_name,
            topology='ring',
            estimate='final'
        )

    elif method == "CHOCO":
        return Parameters(
            name="CHOCO-top-alpha",
            num_epoch=num_epoch,
            lr_type='decay',
            initial_lr=0.1,
            tau=d,
            regularizer=1 / n,
            consensus_lr=0.01,
            quantization='top',
            alpha=0.1, 
            n_cores=n_cores,
            method='choco',
            split_data_random_seed=random_seed,
            distribute_data=True,
            split_data_strategy=split_name,
            topology='ring',
            estimate='final'
        )

    elif method == "LBGD":
        return Parameters(
            name="LBGD",
            num_epoch=num_epoch,
            lr_type='decay',
            initial_lr=0.1,
            tau=d,
            regularizer=1 / n,
            quantization='Sign_Quantizer', 
            m1=6, m2=8,  
            n_cores=n_cores,
            method='LBGD',
            split_data_random_seed=random_seed,
            distribute_data=True,
            split_data_strategy=split_name,
            topology='ring',
            estimate='final'
        )

    elif method == "MoTEF":
        return Parameters(
            name="MoTEF",
            num_epoch=num_epoch,
            lr_type='decay',
            initial_lr=0.1,
            tau=d,
            regularizer=1 / n,
            quantization='top',
            alpha=0.1,
            n_cores=n_cores,
            method='MoTEF',
            gamma=0.1,
            eta=0.0005,
            lam=0.005,
            split_data_random_seed=random_seed,
            distribute_data=True,
            split_data_strategy=split_name,
            topology='ring',
            estimate='final'
        )

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed optimization experiments on epsilon dataset")
    parser.add_argument('experiment', nargs='?', default='DSGD', type=str,
                        help="Choose from: DSGD, CHOCO, LBGD, MoTEF")
    args = parser.parse_args()
    print(f"[INFO] Running experiment: {args.experiment}")

    # Dataset (epsilon)
    dataset_path = os.path.expanduser('../data/epsilon.pickle')
    n, d = 400000, 2000

    # Experiment setup
    n_cores = 9
    num_epoch = 10
    n_repeat = 3
    split_way = 'random'
    split_name = split_way

    # Build parameters for multiple random seeds
    params = [
        build_params(args.experiment, d, n, n_cores, num_epoch, split_name, seed)
        for seed in np.arange(1, n_repeat + 1)
    ]

    # Run experiment
    save_dir = f"dump/epsilon-{args.experiment}-{split_way}-{n_cores}/"
    print(f"[INFO] Saving results to: {save_dir}")
    run_experiment(save_dir, dataset_path, params, nproc=10)
