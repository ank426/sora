from copy import deepcopy

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image

window_size = (720, 1280)

colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "sky_blue": (0, 162, 232),
    "brown": (153, 76, 0),
    "grass_green": (0, 153, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (96, 96, 96),
    "purple": (102, 0, 204),
    "blue_green": (0, 204, 204),
}
for name, color in colors.items():
    colors[name] = np.array(color, np.uint8)

# arialblack = pygame.font.SysFont("arialblack", 50)
# comicsans = pygame.font.SysFont("comicsans", 40)

width = 80
jump_vel = -30


orig_plat_list = [
    [145, 420, 10, 100, "portal"],
    [0, 520, 1280, 30, "grass"],
    [500, 350, 240, 30, "grass"],
    [900, 200, 220, 30, "grass"],
    [480, 50, 200, 30, "grass"],
    [100, -100, 180, 30, "grass"],
    [450, -300, 160, 30, "grass"],
    [800, -450, 140, 30, "grass"],
    [1140, -600, 120, 30, "grass"],
    [1020, -800, 100, 30, "grass"],
    [800, -950, 80, 30, "grass"],
    [600, -1100, 60, 30, "grass"],
    [400, -1250, 40, 30, "grass"],
    [200, -1400, 20, 30, "grass"],
    [400, -1550, 880, 30, "grass"],
    [600, -1750, 20, 30, "grass"],
    [800, -1900, 20, 30, "grass"],
    [1000, -2050, 20, 30, "grass"],
    [1200, -2200, 20, 30, "grass"],
    [0, -2350, 1000, 30, "grass"],
    [500, -3150, 10, 100, "endportal"],
]

orig_mov_plat_list = [
    [400, -3010, 200, 30, 200, 600, -2710, -2510, 6, 6, "grass"],
    [700, -3000, 200, 30, 700, 1000, -2910, -2610, 6, 6, "grass"],
    [315, -3115, 50, 50, 315, 715, -3315, -2915, 8, 8, "fire"],
]


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=window_size + (3,), dtype=np.uint8
        )

        self.action_space = gym.spaces.MultiBinary(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = np.zeros((3,) + window_size, dtype=np.uint8)
        obs[:, :, :] = colors["sky_blue"].reshape(3, 1, 1)

        for plat in self.plat_list:
            plat_x, plat_y, plat_width, plat_height, plat_type = plat
            if 1280 > plat_x > -plat_width:
                if plat_type == "grass":
                    obs[
                        :,
                        max(0, plat_y + 5) : plat_y + plat_height,
                        max(0, plat_x) : plat_x + plat_width,
                    ] = colors["brown"].reshape(3, 1, 1)
                    obs[
                        :,
                        max(0, plat_y) : plat_y + 5,
                        max(0, plat_x) : plat_x + plat_width,
                    ] = colors["green"].reshape(3, 1, 1)
                elif plat_type == "fire":
                    obs[
                        :,
                        max(0, plat_y) : plat_y + plat_height,
                        max(0, plat_x) : plat_x + plat_width,
                    ] = colors["red"].reshape(3, 1, 1)
                elif plat_type == "ice":
                    obs[
                        :,
                        max(0, plat_y) : plat_y + plat_height,
                        max(0, plat_x) : plat_x + plat_width,
                    ] = colors["blue"].reshape(3, 1, 1)
                elif plat_type == "portal":
                    obs[
                        :,
                        max(0, plat_y - 45) : plat_y + plat_height + 45,
                        max(0, plat_x - 45) : plat_x + plat_width + 45,
                    ] = colors["black"].reshape(3, 1, 1)

        for mov_plat in self.mov_plat_list:
            plat_x, plat_y, plat_width, plat_height = mov_plat[:4]
            plat_type = mov_plat[-1]
            if 1280 > plat_x > -plat_width:
                if plat_type == "grass":
                    obs[
                        :,
                        plat_y + 5 : plat_y + plat_height,
                        plat_x : plat_x + plat_width,
                    ] = colors["brown"].reshape(3, 1, 1)
                    obs[
                        :,
                        plat_y : plat_y + 5,
                        plat_x : plat_x + plat_width,
                    ] = colors[
                        "grass_green"
                    ].reshape(3, 1, 1)
                elif plat_type == "fire":
                    obs[
                        :,
                        plat_y : plat_y + plat_height,
                        plat_x : plat_x + plat_width,
                    ] = colors["red"].reshape(3, 1, 1)

        obs[:, self.y : self.y + self.height, self.x : self.x + width] = colors[
            "green"
        ].reshape(3, 1, 1)

        # 3, 720, 1280
        obs = np.transpose(obs, (1, 2, 0))
        # 720, 1280, 3
        image = Image.fromarray(obs, mode="RGB")

        return image

    def reset(self, seed=None, options=None):
        super().reset()

        self.height = 120
        self.x = 110
        self.y = 400
        self.vel_x = 0
        self.vel_y = 0

        self.on_ground = True
        self.on_ice = False
        self.half_height = False
        self.on_mov_plat_vel_x = 0

        self.plat_list = deepcopy(orig_plat_list)
        self.mov_plat_list = deepcopy(orig_mov_plat_list)

        self.frame_counter = 0
        self.checkpoint_counter = 0
        self.death_counter = 0
        self.real_frame_counter = 0

        return self._get_obs(), 0

    def step(self, action):
        truncated = terminated = False

        old_half_height = self.half_height
        if self.half_height:
            self.y -= self.height
            self.height *= 2
            self.half_height = False

        if action[1]:
            self.height //= 2
            self.y += self.height
            self.half_height = True

        old_plat_y_list = []
        for plat in self.plat_list:
            old_plat_y_list.append(plat[1])

        old_mov_plat_y_list = []
        for mov_plat in self.mov_plat_list:
            old_mov_plat_y_list.append(mov_plat[1])

        old_x = self.x
        old_y = self.y

        if self.on_ground:
            if action[2] and not old_half_height:
                self.vel_y = jump_vel
            if self.on_ice:
                if self.vel_x > 0:
                    self.vel_x = 16
                elif self.vel_x < 0:
                    self.vel_x = -16
            else:
                if action[0]:
                    self.vel_x = -10 + self.on_mov_plat_vel_x
                elif action[3]:
                    self.vel_x = +10 + self.on_mov_plat_vel_x
                elif abs(self.vel_x) < 3:
                    self.vel_x = self.on_mov_plat_vel_x
                elif self.vel_x > self.on_mov_plat_vel_x:
                    self.vel_x -= 3
                elif self.vel_x < self.on_mov_plat_vel_x:
                    self.vel_x += 3

        self.x += self.vel_x

        for mov_plat in self.mov_plat_list:
            if mov_plat[0] <= mov_plat[4]:
                mov_plat[8] = abs(mov_plat[8])
            elif mov_plat[0] >= mov_plat[5]:
                mov_plat[8] = -abs(mov_plat[8])
            mov_plat[0] += mov_plat[8]

        for mov_plat in self.mov_plat_list:
            if mov_plat[1] <= mov_plat[6]:
                mov_plat[9] = abs(mov_plat[9])
            elif mov_plat[1] >= mov_plat[7]:
                mov_plat[9] = -abs(mov_plat[9])
            mov_plat[1] += mov_plat[9]

        if self.x <= 0:
            self.x = 0
            self.vel_x = -self.vel_x
        elif self.x >= 1280 - width:
            self.x = 1280 - width
            self.vel_x = -self.vel_x

        self.vel_y += 2
        self.y += self.vel_y

        self.on_ground = False
        self.on_ice = False
        self.on_mov_plat_vel_x = 0

        for i in range(len(self.plat_list)):
            plat_x, plat_y, plat_width, plat_height, plat_type = self.plat_list[i]
            if (
                plat_x - width < self.x < plat_x + plat_width
                and plat_y - self.height < self.y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    self.frame_counter = -1
                elif plat_type == "portal":
                    pass
                elif plat_type == "endportal":
                    terminated = True
                    break
                else:
                    if not plat_y - self.height < old_y < plat_y + plat_height:
                        if self.y < old_y:
                            self.y = plat_y + plat_height
                            self.vel_y = -self.vel_y // 2
                        elif self.y > old_y:
                            self.y = plat_y - self.height
                            self.vel_y = 0
                            self.on_ground = True
                    elif not plat_x - width < old_x < plat_x + plat_width:
                        self.vel_x = -self.vel_x
                        if self.x > old_x:
                            self.x = plat_x - width
                        elif self.x < old_x:
                            self.x = plat_x + plat_width
                    if (
                        old_half_height
                        and not self.half_height
                        and not self.y + self.height // 2 < plat_y + plat_height
                    ):
                        self.half_height = True
                        self.height = self.height // 2
                        self.y += self.height
                    if plat_type == "ice":
                        self.on_ice = True

        for i in range(len(self.mov_plat_list)):
            (
                plat_x,
                plat_y,
                plat_width,
                plat_height,
                _,
                _,
                _,
                _,
                plat_vel_x,
                plat_vel_y,
                plat_type,
            ) = self.mov_plat_list[i]
            if (
                plat_x - width < self.x < plat_x + plat_width
                and plat_y - self.height < self.y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    self.frame_counter = -1
                else:
                    old_plat_x = plat_x - plat_vel_x
                    old_plat_y = plat_y - plat_vel_y
                    if not old_plat_y - self.height < old_y < old_plat_y + plat_height:
                        if plat_y - self.y > old_plat_y - old_y:
                            self.y = plat_y + plat_height
                            self.vel_y = -self.vel_y // 2
                        elif plat_y - self.y < old_plat_y - old_y:
                            self.y = plat_y - self.height
                            self.vel_y = plat_vel_y
                            self.on_ground = True
                            self.on_mov_plat_vel_x = plat_vel_x
                    elif not old_plat_x - width < old_x < old_plat_x + plat_width:
                        self.vel_x = -self.vel_x
                        if plat_x - self.x < old_plat_x - old_x:
                            self.x = plat_x - width
                        elif plat_x - self.x > old_plat_x - old_x:
                            self.x = plat_x + plat_width
                    if (
                        old_half_height
                        and not self.half_height
                        and not self.y + self.height // 2 < plat_y + plat_height
                    ):
                        self.half_height = True
                        self.height = self.height // 2
                        self.y += self.height
                    if plat_type == "ice":
                        self.on_ice = True

        if not self.half_height:
            for plat in self.plat_list:
                plat[1] += 400 - self.y
            for mov_plat in self.mov_plat_list:
                mov_plat[1] += 400 - self.y
                mov_plat[6] += 400 - self.y
                mov_plat[7] += 400 - self.y
            self.y = 400
        else:
            for plat in self.plat_list:
                plat[1] += 460 - self.y
            for mov_plat in self.mov_plat_list:
                mov_plat[1] += 460 - self.y
                mov_plat[6] += 460 - self.y
                mov_plat[7] += 460 - self.y
            self.y = 460

        if self.frame_counter == -1:
            truncated = True

        score = self.plat_list[0][1] - 370
        reward = score + old_plat_y_list[0]

        self.frame_counter += 1
        self.real_frame_counter += 1

        return self._get_obs(), reward, terminated, truncated, score

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_size[1], window_size[0]))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size[1], window_size[0]))
        canvas.fill(colors["sky_blue"])

        for plat in self.plat_list:
            plat_x, plat_y, plat_width, plat_height, plat_type = plat
            if 720 > plat_y > -plat_height:
                if plat_type == "grass":
                    pygame.draw.rect(
                        canvas,
                        colors["brown"],
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        canvas,
                        colors["grass_green"],
                        pygame.Rect(plat_x, plat_y, plat_width, 5),
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        canvas,
                        colors["red"],
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        canvas,
                        colors["blue"],
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "portal" or plat_type == "endportal":
                    pygame.draw.rect(
                        canvas,
                        colors["black"],
                        pygame.Rect(
                            plat_x - 45, plat_y - 50, plat_width + 90, plat_height + 50
                        ),
                    )

        for mov_plat in self.mov_plat_list:
            (
                plat_x,
                plat_y,
                plat_width,
                plat_height,
                _,
                _,
                _,
                _,
                _,
                _,
                plat_type,
            ) = mov_plat
            if 720 > plat_y > -plat_height:
                if plat_type == "grass":
                    pygame.draw.rect(
                        canvas,
                        colors["brown"],
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        canvas,
                        colors["grass_green"],
                        pygame.Rect(plat_x, plat_y, plat_width, 5),
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        canvas,
                        colors["red"],
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        canvas,
                        colors["blue"],
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )

        pygame.draw.rect(
            canvas, colors["green"], pygame.Rect(self.x, self.y, width, self.height)
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])


if __name__ == "__main__":
    env = Env(render_mode="human")
    obs, info = env.reset()
    # obs = np.transpose(obs, (1, 2, 0))
    # image = Image.fromarray(obs, mode="RGB")
    # image.show()
    # for i in range(10):
    #     obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1]))
    obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 1]))
    while True:
        env.render()
    print(reward, terminated, truncated, info)

    # obs = np.transpose(obs, (1, 2, 0))
    # image = Image.fromarray(obs, mode="RGB")
    # image.show()
