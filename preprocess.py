import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T


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


def preprocess(env, skip=4, grayscale=True, shape=(72, 128), num_stack=4):
    env = SkipFrame(env, skip=skip)
    if grayscale:
        env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = gym.wrappers.FrameStack(env, num_stack=num_stack)

    # assert env.observation_space.shape == (num_stack, *shape)

    return env
