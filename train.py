import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torchvision.transforms as T
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from logger import MetricLogger
from sora_env import Env


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done or trunk:
                break

        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        return torch.tensor(observation.copy(), dtype=torch.float)

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        return T.Grayscale()(observation)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        return transforms(observation).squeeze(0)


env = Env(render_mode="human")
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(72, 128))
env = gym.wrappers.FrameStack(env, num_stack=4)

assert env.observation_space.shape == (4, 72, 128)


class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        assert state_dim == (4, 72, 128)
        assert action_dim == 16

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cpu"
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu"))
        )

        self.net = Net(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.burnin = 1e4  # min exps before training
        self.learn_every = 3  # no of exps between updates to Q_online
        self.sync_every = 1e4  # no of exps between Q_target & Q_online sync

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.batch_size = 32
        self.gamma = 0.9

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = (state[0] if isinstance(state, tuple) else state).__array__()
            assert state.shape == (4, 72, 128)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            assert state.shape == (1, 4, 72, 128)
            action_values = self.net(state, model="online")
            assert action_values.shape == (1, 16), print(
                action_values, action_values.shape
            )
            action_idx = torch.argmax(action_values, axis=1).item()

        assert 0 <= action_idx < 16

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        assert state.shape == next_state.shape == (4, 72, 128)
        # action = torch.tensor([action])
        # reward = torch.tensor([reward])
        # done = torch.tensor([done])
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        assert action.shape == reward.shape == done.shape == (1,)

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        assert state.shape == next_state.shape == (self.batch_size, 4, 72, 128)
        assert action.shape == reward.shape == done.shape == (self.batch_size, 1)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # assert state.shape == (self.batch_size, 4, 72, 128)
        # assert action.shape == (self.batch_size, 4)

        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]

        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # assert next_state.shape == (self.batch_size, 4, 72, 128)
        # assert reward.shape == done.shape == (self.batch_size,)

        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        assert h == 72 and w == 128

        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            # 4, 72, 128
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(
    state_dim=(4, 72, 128), action_dim=2**env.action_space.n, save_dir=save_dir
)
logger = MetricLogger(save_dir)

pygame.init()

episodes = 40_000
for e in range(episodes):
    state = env.reset()

    while True:
        if e % 20 == 0:
            env.render()

        action = agent.act(state)
        bin_action = np.array([int(b) for b in np.binary_repr(action).rjust(4, "0")])

        next_state, reward, done, trunc, info = env.step(bin_action)

        agent.cache(state, next_state, action, reward, done)

        q, loss = agent.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or trunc:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
