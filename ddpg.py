import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch


log = logging.getLogger(__name__)


def ddpg(env, brain_name, result_dir, agent, n_episodes=200, max_t=500):
    """
    DDPG
    :param env: RL environment
    :param brain_name: Chosen brain
    :param Path result_dir: Path to result dir
    :param agent: Agent object
    :param int n_episodes: Maximum number of training episodes
    :param int max_t: Maximum number of timesteps per episode
    :return list scores: Scores from each episode
    """

    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.get_action(state)  # select action
            env_info = env.step(action)[brain_name]     # send the action to the environment
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score

        print(f'\rEpisode: {i_episode}\tAverage_score: {np.mean(scores_window)}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode: {i_episode}\tAverage_score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 30.0:
            print(f'\nEnvironment solved in: {i_episode - 100} episodes!\tAverage_score: {np.mean(scores_window)}')
            torch.save(agent.q_network_local.state_dict(), result_dir / 'navigation_model_solution.pth')
            break
    return scores
