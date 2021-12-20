import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class ReinforcePolicy(nn.Module):
    """
    Current version of Reinforce algorithm will not work because it is not adjusted to continuous action space!
    For an environment with a continuous action space, a policy network could have an
    output layer that parametrizes a continuous probability distribution.

    For instance, assume the output later returns the mean and variance od a normal distribution. Then in order
    to select an action, the agent needs only to pass the most recent state st as input to the network and then
    use the output mean and variance to sample from the distribution.

    This should work in theory, but it's unlikely to perform well in practice!
    """
    def __init__(self, space_size, hidden_size, action_size, device):
        super(ReinforcePolicy, self).__init__()
        self.state_size = space_size
        self.action_size = action_size
        self.device = device

        self.fc1 = nn.Linear(self.state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        # we use log prob to construct equivalent loss function
        return action.item(), m.log_prob(action)


class CrossEntropyPolicy(nn.Module):
    def __init__(self, env, brain, state_size, hidden_size, action_size, device, seed):
        super(CrossEntropyPolicy, self).__init__()
        self.env = env
        self.brain = brain
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.device = device
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)
    
    def set_weights(self, weights):
        # separate the weights for each layer
        fc1_end = self.state_size * self.hidden_size + self.hidden_size
        fc1_W = torch.from_numpy(weights[:self.state_size * self.hidden_size].reshape(self.state_size, self.hidden_size))
        fc1_b = torch.from_numpy(weights[self.state_size * self.hidden_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (self.hidden_size*self.action_size)].reshape(self.hidden_size, self.action_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (self.hidden_size*self.action_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        
    def get_weights_dim(self):
        return (self.state_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.action_size
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data
    
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        env_info = self.env.reset(train_mode=True)[self.brain]
        state = env_info.vector_observations[0]
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.forward(state).numpy()
            env_info = self.env.step(action)[self.brain]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            episode_return += reward * math.pow(gamma, t)
            state = next_state
            if done:
                break
        return episode_return
