import argparse
import d3rlpy

from d3rlpy.algos import DiscreteBC
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split

import torch
import os
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(args):
    dataset, env = get_atari(args.dataset)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    bc = DiscreteBC(
        n_frames=4,  # frame stacking
        scaler='pixel',
        use_gpu=args.gpu)

    bc.fit(train_episodes,
           eval_episodes=test_episodes,
           n_epochs=20,
           scorers={'environment': evaluate_on_environment(env, epsilon=0.05)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    dataset_list = ['breakout-mixed-v0', 'breakout-expert-v0', 'breakout-random-v0']
    seed_list = [0, 1, 2]

    for dataset in dataset_list:
        args.dataset = dataset

        for seed in seed_list:
            args.seed = seed

            main(args)

