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
    [0, 600, 800, 30, "grass"],
    [800, 420, 200, 30, "grass"],
    [1150, 300, 200, 30, "grass"],
    [1500, 600, 1600, 30, "grass"],
    [1550, 420, 200, 30, "grass"],
    [2300, 500, 500, 30, "grass"],
    [2300, 200, 30, 310, "grass"],
    [2500, 300, 1100, 30, "grass"],
    [3800, 600, 700, 30, "grass"],
    [4000, 400, 300, 30, "grass"],
    [4000, 200, 280, 30, "grass"],
    [4480, 150, 30, 300, "grass"],
    [4450, 150, 300, 30, "grass"],
    [6000, 600, 1000, 30, "ice"],
    [5000, 600, 1000, 30, "grass"],
    [5500, 500, 50, 100, "fire"],
    [7000, 600, 200, 30, "grass"],
    [6400, 400, 200, 30, "grass"],
    [6800, 200, 500, 30, "grass"],
    [8100, 400, 800, 30, "ice"],
    [7600, 400, 500, 30, "grass"],
    [8100, 290, 500, 30, "fire"],
    [8830, 290, 500, 30, "ice"],
    [9800, 650, 500, 30, "ice"],
    [10600, 450, 300, 30, "ice"],
    [11200, 250, 200, 30, "ice"],
    [11900, 400, 400, 30, "grass"],
    [12666, 600, 50, 50, "fire"],
    [13333, 600, 50, 50, "fire"],
    [14000, 500, 300, 30, "grass"],
    [15000, 600, 50, 50, "fire"],
    [15500, 600, 50, 50, "fire"],
    [16000, 600, 50, 50, "fire"],
    [16200, 500, 30, 30, "fire"],
    [16500, 500, 800, 30, "grass"],
    [17500, 600, 50, 50, "fire"],
    [17800, 600, 50, 50, "fire"],
    [18500, 100, 50, 50, "fire"],
    [18800, 100, 50, 50, "fire"],
    [19200, 500, 1000, 30, "grass"],
    [20400, 350, 300, 30, "ice"],
    [21500, 600, 2000, 30, "ice"],
    [21200, 600, 300, 30, "grass"],
    [23600, 500, 1500, 30, "grass"],
    [24400, 400, 10, 100, "portal"],
    [25150, 600, 400, 30, "tramp"],
    [26000, 500, 400, 30, "acc"],
    [25600, 500, 400, 30, "grass"],
    [26500, 600, 2500, 30, "grass"],
    [29200, 600, 1000, 30, "grass"],
    [29050, 300, 100, 100, "fall"],
]

orig_mov_plat_list = [
    [12300, 500, 300, 30, 12300, 13500, 500, 500, 16, 0, "grass"],
    [14300, 500, 200, 30, 14300, 15000, 500, 500, 14, 0, "grass"],
    [15900, 500, 300, 30, 15200, 15900, 500, 500, 14, 0, "grass"],
    [17400, 300, 300, 30, 17400, 18000, 300, 300, 16, 0, "grass"],
    [18200, 600, 300, 30, 18200, 18600, 600, 600, 16, 0, "grass"],
    [19500, 400, 100, 30, 19500, 20200, 400, 400, 16, 0, "fire"],
    [21700, 550, 100, 30, 21700, 21800, 550, 550, 18, 0, "fire"],
    [22300, 550, 100, 30, 22300, 22500, 550, 550, 20, 0, "fire"],
    [22800, 550, 100, 30, 22800, 22900, 550, 550, 18, 0, "fire"],
    [23050, 550, 100, 30, 23050, 23450, 550, 550, 16, 0, "fire"],
]


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3,) + window_size, dtype=np.uint8
        )

        self.action_space = gym.spaces.MultiBinary(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = np.zeros((3,) + window_size, dtype=np.uint8)
        obs[:, :, :] = colors["black" if self.night else "sky_blue"].reshape(3, 1, 1)

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
                    ] = colors["grass_green"].reshape(3, 1, 1)
                elif plat_type == "fire":
                    obs[
                        :,
                        plat_y : plat_y + plat_height,
                        plat_x : plat_x + plat_width,
                    ] = colors["red"].reshape(3, 1, 1)

        obs[:, self.y : self.y + self.height, self.x : self.x + width] = colors[
            "green"
        ].reshape(3, 1, 1)

        obs = np.transpose(obs, (1, 2, 0))
        image = Image.fromarray(obs, mode="RGB")

        return image

    def reset(self, seed=None, options=None):
        super().reset()

        self.height = 120
        self.x = 600
        self.y = 600 - self.height
        self.vel_x = 0
        self.vel_y = 0

        self.on_ground = True
        self.on_ice = False
        self.half_height = False
        self.night = False
        self.on_mov_plat_vel_x = 0

        self.plat_list = deepcopy(orig_plat_list)
        self.mov_plat_list = deepcopy(orig_mov_plat_list)

        self.frame_counter = 0
        self.checkpoint_counter = 0
        self.death_counter = 0
        self.real_frame_counter = 0

        return self._get_obs(), None

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

        old_plat_x_list = []
        for plat in self.plat_list:
            old_plat_x_list.append(plat[0])

        old_mov_plat_x_list = []
        for mov_plat in self.mov_plat_list:
            old_mov_plat_x_list.append(mov_plat[0])

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
                    self.vel_x = -10
                elif action[3]:
                    self.vel_x = +10
                elif self.vel_x > 0:
                    self.vel_x -= 2
                elif self.vel_x < 0:
                    self.vel_x += 2

        if (
            self.on_mov_plat_vel_x
            and (self.on_mov_plat_vel_x > 0) == (self.vel_x > 0)
            and (action[0] or action[3])
        ):
            self.vel_x += self.on_mov_plat_vel_x
        self.on_mov_plat_vel_x = False

        for plat in self.plat_list:
            plat[0] -= self.vel_x
        for mov_plat in self.mov_plat_list:
            mov_plat[0] -= self.vel_x
            mov_plat[4] -= self.vel_x
            mov_plat[5] -= self.vel_x

        for mov_plat in self.mov_plat_list:
            if mov_plat[0] <= mov_plat[4]:
                mov_plat[8] = abs(mov_plat[8])
            elif mov_plat[0] >= mov_plat[5]:
                mov_plat[8] = -abs(mov_plat[8])
            mov_plat[0] += mov_plat[8]

        self.y += self.vel_y

        if self.y >= 720:
            self.frame_counter = -1
        else:
            self.vel_y += 2
            self.on_ground = False

        chng_plat_x = 0
        self.on_ice = False
        for i in range(len(self.plat_list)):
            plat_x, plat_y, plat_width, plat_height, plat_type = self.plat_list[i]
            old_plat_x = old_plat_x_list[i]
            if (
                plat_x - width < self.x < plat_x + plat_width
                and plat_y - self.height < self.y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    self.frame_counter = -1
                elif plat_type == "portal":
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
                    elif not old_plat_x - width < self.x < old_plat_x + plat_width:
                        self.vel_x = -self.vel_x
                        if self.x > old_plat_x:
                            chng_plat_x += self.x - plat_width - plat_x
                        elif self.x < old_plat_x:
                            chng_plat_x += self.x + width - plat_x
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
                plat_x1,
                plat_x2,
                plat_y1,
                plat_y2,
                plat_vel_x,
                plat_vel_y,
                plat_type,
            ) = self.mov_plat_list[i]
            old_plat_x = old_mov_plat_x_list[i]
            if (
                plat_x - width < self.x < plat_x + plat_width
                and plat_y - self.height < self.y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    self.frame_counter = -1
                else:
                    if not plat_y - self.height < old_y < plat_y + plat_height:
                        if self.y < old_y:
                            self.y = plat_y + plat_height
                            self.vel_y = -self.vel_y // 2
                        elif self.y > old_y:
                            self.y = plat_y - self.height
                            self.vel_y = 0
                            self.vel_x = plat_vel_x
                            if plat_vel_x > 0:
                                self.vel_x += 2
                            elif plat_vel_x < 0:
                                self.vel_x -= 2
                            self.on_ground = True
                            self.on_mov_plat_vel_x = plat_vel_x
                    elif not old_plat_x - width < self.x < old_plat_x + plat_width:
                        self.vel_x = -self.vel_x
                        if self.x > old_plat_x:
                            chng_plat_x += self.x - plat_width - plat_x
                        elif self.x < old_plat_x:
                            chng_plat_x += self.x + width - plat_x
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

        if self.frame_counter == -1:
            truncated = True

        # if self.frame_counter == -1:
        #     self.death_counter += 1
        #     self.vel_x = 0
        #     self.vel_y = 0
        #     self.plat_list = deepcopy(orig_plat_list)
        #     self.mov_plat_list = deepcopy(orig_mov_plat_list)
        #     check = old_plat_x_list[0]
        #     if -1300 < check:
        #         self.y = 600 - self.height
        #     elif -3250 < check:
        #         chng_plat_x = -1300
        #         self.y = 600 - self.height
        #     elif -4450 < check:
        #         chng_plat_x = -3250
        #         self.y = 600 - self.height
        #     elif -7050 < check:
        #         chng_plat_x = -4450
        #         self.y = 600 - self.height
        #     elif -11400 < check:
        #         chng_plat_x = -7050
        #         self.y = 400 - self.height
        #     elif -13500 < check:
        #         chng_plat_x = -11400
        #         self.y = 400 - self.height
        #     elif -16000 < check:
        #         chng_plat_x = -13500
        #         self.y = 500 - self.height
        #     elif -18650 < check:
        #         chng_plat_x = -16000
        #         self.y = 500 - self.height
        #     elif -20700 < check:
        #         chng_plat_x = -18650
        #         self.y = 500 - self.height
        #     elif -23400 < check:
        #         chng_plat_x = -20700
        #         self.y = 600 - self.height
        #     elif -28000 < check:
        #         chng_plat_x = -23400
        #         self.y = 500 - self.height
        #     else:
        #         chng_plat_x = -28000
        #         self.y = 600 - self.height
        #     self.checkpoint_counter = self.fps

        for plat in self.plat_list:
            plat[0] += chng_plat_x
        for mov_plat in self.mov_plat_list:
            mov_plat[0] += chng_plat_x
            mov_plat[4] += chng_plat_x
            mov_plat[5] += chng_plat_x

        # self.score = -old_plat_x_list[0]
        score = -self.plat_list[0][0]
        reward = score + old_plat_x_list[0]

        if 15600 < score < 20400:
            self.night = True
        else:
            self.night = False

        if self.checkpoint_counter:
            self.checkpoint_counter -= 1

        self.frame_counter += 1
        self.real_frame_counter += 1

        return self._get_obs(), reward, terminated, truncated, None

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
        if not self.night:
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
                plat_x1,
                plat_x2,
                plat_y1,
                plat_y2,
                plat_vel_x,
                plat_vel_y,
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
