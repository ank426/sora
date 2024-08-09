import datetime
from pathlib import Path

import numpy as np
import pygame

from agent import Agent
from preprocess import preprocess
from sora_env import Env

# torch.cuda.empty_cache()


env = Env(render_mode="human")
env = preprocess(env, skip=4, grayscale=True, shape=(72, 128), num_stack=4)


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(
    state_dim=(4, 72, 128), action_dim=2**env.action_space.n, save_dir=save_dir
)

pygame.init()

state, info = env.reset()

while True:
    env.render()

    action = agent.act(state)
    bin_action = np.array([int(b) for b in np.binary_repr(action).rjust(4, "0")])

    next_state, reward, done, trunc, info = env.step(bin_action)

    state = next_state

    if done or trunc:
        break
