import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from agent import Agent
from logger import MetricLogger
from preprocess import preprocess
from sora_env import Env

# import psutil
# import pygame

# torch.cuda.empty_cache()
# process = psutil.Process()


def new():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
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
        exploration_rate_min=0.01,
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

    return env, agent, logger


def load():
    def latest_run(checkpoint_path):
        runs = [
            d
            for d in os.listdir(checkpoint_path)
            if os.path.isdir(os.path.join(checkpoint_path, d))
        ]
        runs_with_times = [
            (run, datetime.strptime(run, "%Y-%m-%dT%H-%M-%S")) for run in runs
        ]
        latest_run = max(runs_with_times, key=lambda x: x[1])[0]

        return latest_run

    def latest_model(checkpoint_path, run_dir):
        models = [
            int(m[9:-6])
            for m in os.listdir(os.path.join(checkpoint_path, run_dir))
            if m.startswith("sora_net_") and m.endswith(".chkpt")
        ]
        latest_model = "sora_net_" + str(max(models)) + ".chkpt"
        return latest_model

    checkpoint_path = Path("checkpoints")
    run = latest_run(checkpoint_path)
    model = latest_model(checkpoint_path, run)
    save_dir = checkpoint_path / run
    load_path = checkpoint_path / run / model

    model_dict = torch.load(load_path, weights_only=False)

    hyp_par = model_dict["hyp_par"]
    skip = hyp_par["skip"]
    grayscale = hyp_par["grayscale"]
    shape = hyp_par["shape"]
    num_stack = hyp_par["num_stack"]

    env = preprocess(
        Env(render_mode="human"),
        skip=skip,
        grayscale=grayscale,
        shape=shape,
        num_stack=num_stack,
    )

    hyp_par["exploration_rate_min"] = 0.01

    agent = Agent(
        action_dim=2**env.action_space.n,
        hyp_par=hyp_par,
        exploration_rate=model_dict["exploration_rate"],
        save_every=1e4,
        save_dir=save_dir,
    )
    agent.net.load_state_dict(model_dict["model"])

    logger = MetricLogger(save_dir)

    return env, agent, logger


# env, agent, logger = new()
env, agent, logger = load()


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
