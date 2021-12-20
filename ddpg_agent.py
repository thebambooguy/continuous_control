import torch
import torch.optim as optim
import torch.nn.functional as F

from models import Actor, Critic
from replay_buffer import ReplayBuffer


N_LEARN_UPDATES = 10    # number of learning updates
N_TIME_STEPS = 20       # every n_time do update


class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_dims, batch_size, buffer_size, actor_lr, critic_lr, gamma, tau,
                 noise, device, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.device = device

        self.actor = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.critic = Critic(self.state_size + self.action_size, self.action_size, seed).to(self.device)
        self.critic_target = Critic(self.state_size + self.action_size, self.action_size, seed).to(self.device)

        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device, seed=0)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=0)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def reset(self):
        self.noise.reset()

    def get_action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Agent makes a step in environment.
        :param int state:  Value of current state
        :param int action: Action chosen by the agent
        :param int reward: Reward after choosing action
        :param int next_state: Value of next state
        :param int done: Whether episode is finished
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            for i in range(N_LEARN_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.critic_optim.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # if self.t_step == 0:
        #     self.soft_update(self.critic, self.critic_target, self.tau)
        #     self.soft_update(self.actor, self.actor_target, self.tau)

        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters - 0_target = tau * 0_local + (1 - tau) * Q_target
        Polyak averaging method to mix the target network with a tiny bit of the online network more frequently (?)
        Equation suggests that.
        :param PyTorch model local_model: weights will be copied from
        :param PyTorch model target_model: weights will be copied to
        :param float tau: interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
