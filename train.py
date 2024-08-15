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

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

hyp_par = dict(
    skip=4,
    grayscale=True,
    shape=(72, 128),
    num_stack=4,
    burnin=1e4,
    learn_every=3,
    sync_every=1e4,
    lr=0.00025,
    batch_size=32,
    gamma=0.99,
    storage=20000,
    exploration_rate_decay=0.99999,  # 0.99999975
    exploration_rate_min=0.1,
)

# env = Env(render_mode="human")
# skip = 4
# grayscale = True
# shape = (72, 128)
# num_stack = 4

env = preprocess(
    Env(render_mode="human"),
    skip=hyp_par["skip"],
    grayscale=hyp_par["grayscale"],
    shape=hyp_par["shape"],
    num_stack=hyp_par["num_stack"],
)

agent = Agent(
    action_dim=2**env.action_space.n,
    hyp_par=hyp_par,
    exploration_rate=1,
    save_every=1e4,  # 5e5
    save_dir=save_dir,
)
logger = MetricLogger(save_dir)

# pygame.init()

episodes = 40_000
for e in range(episodes):
    state, score = env.reset()

    i = 0
    while i < score * 2 + 50:
        # if e % 1000 == 0:
        #     env.render()

        action = agent.act(state)
        bin_action = np.array([int(b) for b in np.binary_repr(action).rjust(4, "0")])

        next_state, reward, done, trunc, score = env.step(bin_action)

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
