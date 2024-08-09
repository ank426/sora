import numpy as np
import pygame

from sora_env import Env

env = Env(render_mode="human")
observation, info = env.reset()

pygame.init()

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    env.render()

    action = np.zeros(4)

    pressed = pygame.key.get_pressed()
    action[0] = pressed[pygame.K_LEFT] or pressed[pygame.K_a] or pressed[pygame.K_h]
    action[1] = pressed[pygame.K_DOWN] or pressed[pygame.K_s] or pressed[pygame.K_j]
    action[2] = pressed[pygame.K_UP] or pressed[pygame.K_w] or pressed[pygame.K_k]
    action[3] = pressed[pygame.K_RIGHT] or pressed[pygame.K_d] or pressed[pygame.K_l]

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        done = True
