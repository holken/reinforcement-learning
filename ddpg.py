import torch
import torch.nn as nn
import gym
import numpy as np
from copy import deepcopy

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def simple_deterministc_nn(obs_dim, act_dim):
    return nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, act_dim)
    )


class ReplayBuffer():

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rew = np.zeros(buffer_size, dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)
        self.pointer, self.size, self.max_size = 0, 0, buffer_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.pointer] = obs
        self.act[self.pointer] = act
        self.rew[self.pointer] = rew
        self.next_obs[self.pointer] = next_obs
        self.done[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs[idxs],
                     next_obs=self.next_obs[idxs],
                     act=self.act[idxs],
                     rew=self.rew[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class CustomCritic(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.v_net = mlp(sizes=sizes, activation=nn.Softmax)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.

def train(policy_lr=1e-3,
          q_lr=1e-3,
          hidden_sizes=[32],
          epochs=100,
          replay_buff_size=1000000,
          batch_sample_size=100,
          discount=0.99,
          epoch_size=4000,
          update_after=1000,
          update_every=50,
          polyak=0.995,
          act_noise=0.1,
          start_steps=10000,
          env_type='CartPole-v1'):

    env = gym.make(env_type)
    obs_dim = env.observation_space.shape[0]
    acts_n = env.action_space.shape[0]
    act_limit = env.action_space.high[0]


    q_net = CustomCritic(sizes=[obs_dim + acts_n] + hidden_sizes + [1])
    q_opt = torch.optim.Adam(params=q_net.parameters(), lr=q_lr)

    # Real policies
    policy = simple_deterministc_nn(obs_dim, acts_n)
    policy_opt = torch.optim.Adam(params=policy.parameters(), lr=policy_lr)

    # Target policies
    target_q_net = deepcopy(q_net)

    target_policy = deepcopy(policy)

    def get_action(policy, obs):
        with torch.no_grad():
            act_no_noise = policy(obs).numpy()
        act = act_no_noise + (act_noise * np.random.randn(acts_n))
        return np.clip(act, -act_limit, act_limit)

    def compute_targets(next_obs, rew, done):
        with torch.no_grad():
            target_a = target_policy(next_obs)
            target_q = target_q_net(torch.cat([next_obs, target_a], dim=-1))
            return rew + discount * (1 - done) * target_q

    def get_q_loss(obs, act, targets):
        q_val = target_q_net(torch.cat([obs, act], dim=-1))
        return ((q_val - targets) ** 2).mean()

    def get_policy_loss(obs):
        act = policy(obs)
        q_val = q_net(torch.cat([obs, act], dim=-1))
        return -q_val.mean()

    def update():
        transitions = replay_buffer.sample_batch(batch_sample_size)
        obs = transitions['obs']
        act = transitions['act']
        next_obs = transitions['next_obs']
        rew = transitions['rew']
        done = transitions['done']

        targets = compute_targets(next_obs, rew, done)

        q_opt.zero_grad()
        q_loss = get_q_loss(obs, act, targets)
        q_loss.backward()
        q_opt.step()

        for q in q_net.parameters():
            q.requires_grad = False

        policy_opt.zero_grad()
        policy_loss = get_policy_loss(obs)
        policy_loss.backward()
        policy_opt.step()

        for q in q_net.parameters():
            q.requires_grad = True

        with torch.no_grad():
            for q, q_targ in zip(q_net.parameters(), target_q_net.parameters()):
                q_targ.data.mul_(polyak)
                q_targ.data.add_((1 - polyak) * q.data)

        with torch.no_grad():
            for p, p_targ in zip(policy.parameters(), target_policy.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


    # Keeps s, a, r, s', d
    replay_buffer = ReplayBuffer(obs_dim, acts_n, replay_buff_size)
    obs = env.reset()
    render_attempt = True
    acc_rew = 0
    for t in range(epochs * epoch_size):

        if render_attempt:
            env.render()

        if t > start_steps:
            act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
        else:
            act = env.action_space.sample()

        new_obs, reward, done, _ = env.step(act)
        done = 1 if done else 0

        replay_buffer.store(obs, act, reward, new_obs, done)

        obs = new_obs
        acc_rew += reward

        if done:
            obs = env.reset()
            render_attempt = False

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                update()
        if t % epoch_size == 0:
            print(f"step: {t}")
            print(f"acc reward: {acc_rew}")
            acc_rew = 0
            render_attempt = True





if __name__ == '__main__':
    # Possible env 'MountainCarContinuous-v0'
    # Possible env 'Pendulum-v0'
    train(epochs=50, env_type='Pendulum-v0', act_noise=0.1)