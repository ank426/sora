import datetime
from pathlib import Path

import numpy as np

from agent import Agent
from logger import MetricLogger
from preprocess import preprocess
from sora_env import Env

# import psutil
# import pygame


# torch.cuda.empty_cache()
# process = psutil.Process()

env = Env(render_mode="human")
env = preprocess(env, skip=4, grayscale=True, shape=(72, 128), num_stack=4)

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(
    state_dim=(4, 72, 128), action_dim=2**env.action_space.n, save_dir=save_dir
)
logger = MetricLogger(save_dir)

# pygame.init()

episodes = 40_000
for e in range(episodes):
    state, info = env.reset()

    i = 0
    while i < info * 2 + 50:
        # if e % 1000 == 0:
        #     env.render()

        action = agent.act(state)
        bin_action = np.array([int(b) for b in np.binary_repr(action).rjust(4, "0")])

        next_state, reward, done, trunc, info = env.step(bin_action)

        agent.cache(state, next_state, action, reward, done)

        q, loss = agent.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or trunc:
            break

        i += 1

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)

    # print(e, process.memory_info().rss / 1024 / 1024 / 1024)
