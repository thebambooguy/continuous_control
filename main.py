import argparse
import logging

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from unityagents import UnityEnvironment

from ddpg import ddpg
from ddpg_agent import DDPGAgent
from util import get_information_about_env, configure_logging, close_logging
from exploration_noise import OUNoise

log = logging.getLogger(__name__)

ENV_PATH = './Reacher_Linux/Reacher.x86_64'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e-7)    # replay buffer size
BATCH_SIZE = 128
ACTOR_LR = 5e-4
CRITIC_LR = 5e-4
TAU = 1e-3


def create_and_parse_args(args=None):
    parser = argparse.ArgumentParser(description='Continuous Control')
    parser.add_argument("--n_iterations", type=int, default=3000, help='Maximum number of training iteraions')
    parser.add_argument("--max_t", type=int, default=1500, help='Maximum number of timesteps per episode')
    parser.add_argument("--hidden_layer_size", type=int, default=128, help='Number of neurons in hidden layer')
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount rate')
    parser.add_argument("--results_dir", type=Path, default='results/temp', help='Results dir where results will be stored')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = create_and_parse_args()
    configure_logging(args.results_dir)

    env = UnityEnvironment(file_name=ENV_PATH)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents, action_size, state_size = get_information_about_env(env_info, brain)

    agent = DDPGAgent(state_size, action_size, hidden_dims=args.hidden_layer_size, batch_size=BATCH_SIZE,
                      buffer_size=BUFFER_SIZE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=args.gamma, tau=TAU,
                      noise=OUNoise(action_size, args.seed), device=DEVICE, seed=args.seed)

    scores = ddpg(env, brain_name, args.results_dir, agent, args.n_iterations, args.max_t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    args.results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.results_dir / 'agent_scores.png')

    with open(args.results_dir / "args.yaml", 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    close_logging()
