import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from unityagents import UnityEnvironment

from ddpg import ddpg
from ddpg_agent import Agent
from util import get_information_about_env


ENV_PATH = './Reacher_Linux/Reacher.x86_64'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_and_parse_args(args=None):
    parser = argparse.ArgumentParser(description='Continuous Control')
    parser.add_argument("--n_iterations", type=int, default=3000, help='Maximum number of training iteraions')
    parser.add_argument("--max_t", type=int, default=1500, help='Maximum number of timesteps per episode')
    parser.add_argument("--actor_lr", type=int, default=2e-4, help='Actor learning rate')
    parser.add_argument("--critic_lr", type=float, default=2e-4, help='Critic learning rate')
    parser.add_argument("--weight_decay", type=int, default=0, help='Weight decay for critic optimizer')
    parser.add_argument("--buffer_size", type=int, default=int(1e7), help='Replay Buffer size')
    parser.add_argument("--batch_size", type=int, default=256, help='Batch Size')
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount factor')
    parser.add_argument("--tau", type=float, default=1e-3, help='Soft Update interpolation parameter')
    parser.add_argument("--results_dir", type=Path, default='results/1_agent_env', help='Results dir')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = create_and_parse_args()

    env = UnityEnvironment(file_name=ENV_PATH, seed=2)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents, action_size, state_size = get_information_about_env(env_info, brain)

    agent = Agent(state_size, action_size, args.actor_lr, args.critic_lr, args.weight_decay, args.buffer_size,
                  args.batch_size, args.gamma, args.tau, DEVICE, args.seed)

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

