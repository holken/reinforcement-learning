import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class CustomCritic(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.v_net = mlp(sizes=sizes, activation=nn.Softmax)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


def train(policy_lr=1e-2,
          val_lr=1e-3,
          hidden_sizes=[32],
          batch_size=5000,
          epochs=50,
          env_type='CartPole-v0',
          epsilon=0.2):
    env = gym.make(env_type)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Policy & Optimizer
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    optimizer_policy = torch.optim.Adam(logits_net.parameters(), lr=policy_lr)

    # Value function
    val_net = CustomCritic(sizes=[obs_dim] + hidden_sizes + [n_acts])
    optimizer_val = torch.optim.Adam(val_net.parameters(), lr=val_lr)

    batch_weights = [0] * batch_size
    oldp = torch.randn(batch_size)

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def get_return_to_go(rews):
        n = len(rews)
        rtgs = [0] * n
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def get_policy_loss(obs, act, adv, oldp):
        newp = get_policy(obs).log_prob(act)
        policy_ratio = torch.exp(newp - oldp)
        clip_adv = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon) * adv

        policy_loss = -(torch.min(policy_ratio * adv, clip_adv)).mean()
        return policy_loss

    def get_advantage(vals, rews, discount=0.99, lam=0.97):
        n = len(rews)
        adv = [0] * n
        for i in range(n):
            td_error = rews[i] + ((vals[i + 1] if i + 1 < n else 0) * discount * lam)
            adv[i] = td_error - vals[i]
        return adv

    def update_value_function(obs, rewards):
        logits = val_net(obs)
        return ((logits - rewards.reshape(-1, 1))**2).mean()


    def update(obs, acts, vals, rews):
        #nonlocal batch_weights
        oldp = get_policy(obs=torch.as_tensor(obs, dtype=torch.float32)).log_prob(torch.as_tensor(acts, dtype=torch.float32)).detach()
        batch_weights = get_advantage(vals, rews)


        for i in range(5):
            optimizer_policy.zero_grad()
            batch_loss = get_policy_loss(obs=torch.as_tensor(obs, dtype=torch.float32),
                                  act=torch.as_tensor(acts, dtype=torch.float32),
                                  adv=torch.as_tensor(batch_weights, dtype=torch.float32),
                                  oldp=torch.as_tensor(oldp, dtype=torch.float32))
            batch_loss.backward()
            optimizer_policy.step()

        for i in range(5):
            optimizer_val.zero_grad()
            batch_value_loss = update_value_function(obs=torch.as_tensor(obs, dtype=torch.float32),
                                                     rewards=torch.as_tensor(rews, dtype=torch.float32))
            batch_value_loss.backward()
            optimizer_val.step()

        print(f"Batch_loss: {batch_loss}")
        print(f"batch_value_loss: {batch_value_loss}")
        print(f"avg_reward: {sum(rews) / len(rews)}")
        print(f"reward:: {sum(rews)}")

    def train_one_epoch(curr_epoch):

        # Lasting throughout the whole epoch
        batch_obs = []
        batch_act = []
        batch_rews = []
        batch_vals = []

        # Resets per episode
        obs = env.reset()
        ep_rews = []
        done = False

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        while True:

            if not finished_rendering_this_epoch:
                env.render()

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            batch_act.append(act)
            batch_obs.append(obs.copy())

            obs, rew, done, _ = env.step(act)

            ep_rews.append(rew)
            batch_vals.append(val_net(torch.as_tensor(obs, dtype=torch.float32)).mean())

            if done or len(batch_obs) == batch_size:
                batch_rews += get_return_to_go(ep_rews)

                obs, done, ep_rews = env.reset(), False, []

                finished_rendering_this_epoch = True

                if len(batch_obs) == batch_size:
                    break

        print(f"Epoch: {curr_epoch}")
        update(batch_obs, batch_act, batch_vals, batch_rews)

    for i in range(epochs):
        train_one_epoch(i)
    env.close()


if __name__ == '__main__':
    # Possible env 'MountainCar-v0'
    train(epochs=50, env_type='MountainCar-v0')