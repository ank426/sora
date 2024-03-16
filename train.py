import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from sora_env import Env

env = Env()


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
        transform = self.permute_orientation(observation)
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


env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = gym.wrappers.FrameStack(env, num_stack=4)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cpu"
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu"))
        )

        self.net = Net(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = env.action_space.sample()
        else:
            state = (
                state[0].__array__() if isinstance(state, tuple) else state.__array__()
            )
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action = torch.round(action_values)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

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
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        pass


class Net:
    pass
