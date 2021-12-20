import logging

from logging import StreamHandler, FileHandler


def get_action_space_size(brain):
    return brain.vector_action_space_size


def get_state_space_size(env):
    states = env.vector_observations
    return states.shape[1]


def get_information_about_env(env_info, brain):
    # number of agents
    num_agents = len(env_info.agents)
    # size of each action
    action_size = get_action_space_size(brain)
    # examine the state space
    state_size = get_state_space_size(env_info)
    return num_agents, action_size, state_size


def configure_logging(root_path):
    root_path.mkdir(exist_ok=True, parents=True)

    log = logging.getLogger()
    # remove existing handlers
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in [StreamHandler(), FileHandler(str(root_path / 'cross_entropy.log'))]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)


def close_logging():
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
