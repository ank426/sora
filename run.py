import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
import torch

from agent import Agent
from preprocess import preprocess
from sora_env import Env

# torch.cuda.empty_cache()


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

agent = Agent(action_dim=2**env.action_space.n, hyp_par=hyp_par, exploration_rate=0)
agent.net.load_state_dict(model_dict["model"])

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
