from copy import deepcopy

import pygame

pygame.init()
res720p = (1280, 720)
"""
import pygame.locals
flags = pygame.locals.FULLSCREEN | pygame.locals.DOUBLEBUF
screen = pygame.display.set_mode(res720p, flags, 16)
"""
screen = pygame.display.set_mode(res720p)
clock = pygame.time.Clock()
fps = 30
pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
sky_blue = (0, 162, 232)
brown = (153, 76, 0)
grass_green = (0, 153, 0)
black = (0, 0, 0)
white = (255, 255, 255)
grey = (96, 96, 96)
purple = (102, 0, 204)
blue_green = (0, 204, 204)

is_green = True
night = False

arialblack = pygame.font.SysFont("arialblack", 50)
comicsans = pygame.font.SysFont("comicsans", 40)

width = 80
height = 120  # even
half_height = False

level = 1

if level == 1:
    done = False

    x = 600
    y = 600 - height
    vel_x = 0
    vel_y = 0
    jump_vel = -30
    on_ground = True
    on_ice = False
    on_mov_plat_vel_x = 0
    fall_vel_y = 0

    # [plat_x, plat_y, plat_width, plat_height, plat_type]
    # Platforms sharing boundaries glitch when player hit that.
    # Ice must be before adj/nearby platform.
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
    plat_list = deepcopy(orig_plat_list)

    # [plat_x,plat_y,plat_width,plat_height,plat_x1,plat_x2,plat_y1,plat_y2,plat_vel_x,plat_vel_y,plat_type]
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
    mov_plat_list = deepcopy(orig_mov_plat_list)

    headstart = 0

    for plat in plat_list:
        plat[0] -= headstart
    for mov_plat in mov_plat_list:
        mov_plat[0] -= headstart
        mov_plat[4] -= headstart
        mov_plat[5] -= headstart
    if headstart:
        y = 0

    # image = pygame.image.load('C:\\Users\\Avinash\\Desktop\\image.jpg').convert()
    # screen.blit(image, (0,0))

    frame_counter = 0
    color_counter = 0
    checkpoint_counter = 0
    death_counter = 0
    real_frame_counter = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        pygame.display.set_caption("Sora: Level 1")

        old_half_height = half_height
        if half_height:
            y -= height
            height *= 2
            half_height = False

        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
            height = height // 2
            y += height
            half_height = True

        old_plat_x_list = []
        for plat in plat_list:
            old_plat_x_list.append(plat[0])

        old_mov_plat_x_list = []
        for mov_plat in mov_plat_list:
            old_mov_plat_x_list.append(mov_plat[0])

        old_y = y

        if on_ground:
            if (pressed[pygame.K_UP] or pressed[pygame.K_w]) and not old_half_height:
                vel_y = jump_vel
            if on_ice:
                if vel_x > 0:
                    vel_x = 16
                elif vel_x < 0:
                    vel_x = -16
            else:
                if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
                    vel_x = -10
                elif pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
                    vel_x = +10
                elif vel_x > 0:
                    vel_x -= 2
                elif vel_x < 0:
                    vel_x += 2

        if (
            on_mov_plat_vel_x
            and (on_mov_plat_vel_x > 0) == (vel_x > 0)
            and (
                pressed[pygame.K_LEFT]
                or pressed[pygame.K_a]
                or pressed[pygame.K_RIGHT]
                or pressed[pygame.K_d]
            )
        ):
            vel_x += on_mov_plat_vel_x
        on_mov_plat_vel_x = False

        for plat in plat_list:
            plat[0] -= vel_x
        for mov_plat in mov_plat_list:
            mov_plat[0] -= vel_x
            mov_plat[4] -= vel_x
            mov_plat[5] -= vel_x

        # plat_x, plat_y, plat_width, plat_height, plat_x1, plat_x2, plat_y1, plat_y2, plat_vel_x, plat_vel_y, plat_type = mov_plat

        for mov_plat in mov_plat_list:
            if mov_plat[0] <= mov_plat[4]:
                mov_plat[8] = abs(mov_plat[8])
            elif mov_plat[0] >= mov_plat[5]:
                mov_plat[8] = -abs(mov_plat[8])
            mov_plat[0] += mov_plat[8]
        """
        for mov_plat in mov_plat_list:
            if mov_plat[1] <= mov_plat[6]:
                mov_plat[9] = abs(mov_plat[9])
            elif mov_plat[1] >= mov_plat[7]:
                mov_plat[9] = -abs(mov_plat[9])
                y -= 100
            mov_plat[1] += mov_plat[9]
        """
        y += vel_y

        if y >= 720:
            frame_counter = -1
        else:
            vel_y += 2
            on_ground = False

        chng_plat_x = 0
        on_ice = False
        for i in range(len(plat_list)):
            plat_x, plat_y, plat_width, plat_height, plat_type = plat_list[i]
            old_plat_x = old_plat_x_list[i]
            if plat_type == "fall":
                fall_vel_y += 1
                if plat_y > 720:
                    plat_list[i][1] = 100
                    fall_vel_y = 0
                plat_list[i][1] += fall_vel_y
            if (
                plat_x - width < x < plat_x + plat_width
                and plat_y - height < y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    frame_counter = -1
                elif plat_type == "portal":
                    level += 1
                    done = True
                    break
                else:
                    if not plat_y - height < old_y < plat_y + plat_height:
                        if y < old_y:
                            y = plat_y + plat_height
                            vel_y = -vel_y // 2
                        elif y > old_y:
                            y = plat_y - height
                            if plat_type == "tramp":
                                vel_y = -60
                            else:
                                vel_y = 0
                            if plat_type == "acc":
                                vel_x += 10
                            on_ground = True
                    elif not old_plat_x - width < x < old_plat_x + plat_width:
                        vel_x = -vel_x
                        if x > old_plat_x:
                            chng_plat_x += x - plat_width - plat_x
                        elif x < old_plat_x:
                            chng_plat_x += x + width - plat_x
                    if (
                        old_half_height
                        and not half_height
                        and not y + height // 2 < plat_y + plat_height
                    ):
                        half_height = True
                        height = height // 2
                        y += height
                    if plat_type == "ice":
                        on_ice = True

        for i in range(len(mov_plat_list)):
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
            ) = mov_plat_list[i]
            old_plat_x = old_mov_plat_x_list[i]
            if (
                plat_x - width < x < plat_x + plat_width
                and plat_y - height < y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    frame_counter = -1
                else:
                    if not plat_y - height < old_y < plat_y + plat_height:
                        if y < old_y:
                            y = plat_y + plat_height
                            vel_y = -vel_y // 2
                        elif y > old_y:
                            y = plat_y - height
                            vel_y = 0
                            vel_x = plat_vel_x
                            if plat_vel_x > 0:
                                vel_x += 2
                            elif plat_vel_x < 0:
                                vel_x -= 2
                            on_ground = True
                            on_mov_plat_vel_x = plat_vel_x
                    elif not old_plat_x - width < x < old_plat_x + plat_width:
                        vel_x = -vel_x
                        if x > old_plat_x:
                            chng_plat_x += x - plat_width - plat_x
                        elif x < old_plat_x:
                            chng_plat_x += x + width - plat_x
                    if (
                        old_half_height
                        and not half_height
                        and not y + height // 2 < plat_y + plat_height
                    ):
                        half_height = True
                        height = height // 2
                        y += height
                    if plat_type == "ice":
                        on_ice = True

        if frame_counter == -1:
            death_counter += 1
            vel_x = 0
            vel_y = 0
            plat_list = deepcopy(orig_plat_list)
            mov_plat_list = deepcopy(orig_mov_plat_list)
            check = old_plat_x_list[0]
            if -1300 < check:
                y = 600 - height
            elif -3250 < check:
                chng_plat_x = -1300
                y = 600 - height
            elif -4450 < check:
                chng_plat_x = -3250
                y = 600 - height
            elif -7050 < check:
                chng_plat_x = -4450
                y = 600 - height
            elif -11400 < check:
                chng_plat_x = -7050
                y = 400 - height
            elif -13500 < check:
                chng_plat_x = -11400
                y = 400 - height
            elif -16000 < check:
                chng_plat_x = -13500
                y = 500 - height
            elif -18650 < check:
                chng_plat_x = -16000
                y = 500 - height
            elif -20700 < check:
                chng_plat_x = -18650
                y = 500 - height
            elif -23400 < check:
                chng_plat_x = -20700
                y = 600 - height
            elif -28000 < check:
                chng_plat_x = -23400
                y = 500 - height
            else:
                chng_plat_x = -28000
                y = 600 - height
            checkpoint_counter = fps

        for plat in plat_list:
            plat[0] += chng_plat_x
        for mov_plat in mov_plat_list:
            mov_plat[0] += chng_plat_x
            mov_plat[4] += chng_plat_x
            mov_plat[5] += chng_plat_x

        score = -old_plat_x_list[0]

        if 15600 < score < 20400:
            night = True
        else:
            night = False

        if night:
            screen.fill(black)
        else:
            screen.fill(sky_blue)

        if pressed[pygame.K_SPACE]:
            color_counter += 1
        else:
            color_counter = 0
        if color_counter == fps // 2:
            color_counter = 0
            is_green = not is_green

        if is_green:
            color = green
        else:
            color = red

        for plat in plat_list:
            plat_x, plat_y, plat_width, plat_height, plat_type = plat
            if 1280 > plat_x > -plat_width:
                if plat_type == "grass":
                    pygame.draw.rect(
                        screen,
                        brown,
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        screen, grass_green, pygame.Rect(plat_x, plat_y, plat_width, 5)
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        screen,
                        red,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        screen,
                        blue,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "portal":
                    pygame.draw.rect(
                        screen,
                        black,
                        pygame.Rect(
                            plat_x - 45, plat_y - 50, plat_width + 90, plat_height + 50
                        ),
                    )
                elif plat_type == "tramp":
                    pygame.draw.rect(
                        screen,
                        purple,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "acc":
                    pygame.draw.rect(
                        screen,
                        blue_green,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "fall":
                    pygame.draw.rect(
                        screen,
                        grey,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
        for mov_plat in mov_plat_list:
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
            if 1280 > plat_x > -plat_width:
                if plat_type == "grass":
                    pygame.draw.rect(
                        screen,
                        brown,
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        screen, grass_green, pygame.Rect(plat_x, plat_y, plat_width, 5)
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        screen,
                        red,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        screen,
                        blue,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )

        pygame.draw.rect(screen, color, pygame.Rect(x, y, width, height))

        if night:
            scoretext = arialblack.render("Score: " + str(score), True, white)
            deathtext = arialblack.render("Deaths: " + str(death_counter), True, white)
            timetext = arialblack.render(
                "Time: " + str(real_frame_counter // fps) + "s", True, white
            )
        else:
            scoretext = arialblack.render("Score: " + str(score), True, black)
            deathtext = arialblack.render("Deaths: " + str(death_counter), True, black)
            timetext = arialblack.render(
                "Time: " + str(real_frame_counter // fps) + "s", True, black
            )
        scoreRect = scoretext.get_rect()
        scoreRect.center = (640, 75)
        screen.blit(scoretext, scoreRect)
        deathRect = deathtext.get_rect()
        deathRect.center = (150, 75)
        screen.blit(deathtext, deathRect)
        timeRect = timetext.get_rect()
        timeRect.center = (1100, 75)
        screen.blit(timetext, timeRect)

        if score < 800:
            note = comicsans.render(
                "While holding left or right, press up to jump backward or forward.",
                True,
                black,
            )
        elif 1200 < score < 1600:
            note = comicsans.render("Hold down for half height.", True, black)
        elif 2500 < score < 2700:
            note = comicsans.render(
                "Hold space for half a second to turn red. No point though.",
                True,
                black,
            )
        elif 4300 < score < 4800:
            note = comicsans.render("FIRE", True, red)
        elif 5000 < score < 6000:
            note = comicsans.render("ICE", True, blue)
        elif 11300 < score < 13000:
            note = comicsans.render("Platforms moving back and forth.", True, black)
        elif 13400 < score < 14000:
            note = comicsans.render(
                "Multiple platforms moving back and forth", True, black
            )
        elif 15800 < score < 16600:
            note = comicsans.render("Valley of Darkness", True, white)
        elif 20500 < score < 22900:
            note = comicsans.render("Fire and Ice", True, black)
        elif 23200 < score:
            note = comicsans.render("LEVEL 1 COMPLETE", True, black)
        else:
            note = False

        if checkpoint_counter:
            checkpoint_counter -= 1
            if night:
                note = comicsans.render("Checkpoint Activated", True, white, black)
            else:
                note = comicsans.render("Checkpoint Activated", True, black, sky_blue)

        if note:
            noteRect = note.get_rect()
            noteRect.center = (640, 150)
            screen.blit(note, noteRect)

        frame_counter += 1
        real_frame_counter += 1

        clock.tick(fps)
        print(clock.get_time())
        pygame.display.flip()

if level == 2:
    done = False

    x = 110
    y = 400
    vel_x = 0
    vel_y = 0
    jump_vel = -30
    on_ground = True
    on_ice = False
    on_mov_plat_vel_x = 0
    fall_vel_y = 0

    # [plat_x,plat_y,plat_width,plat_height,plat_type] #Platforms sharing boundaries glitch when player hit that. #Ice must be before adj/nearby platform.
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
    plat_list = deepcopy(orig_plat_list)

    # [plat_x,plat_y,plat_width,plat_height,plat_x1,plat_x2,plat_y1,plat_y2,plat_vel_x,plat_vel_y,plat_type]
    orig_mov_plat_list = [
        [400, -3010, 200, 30, 200, 600, -2710, -2510, 6, 6, "grass"],
        [700, -3000, 200, 30, 700, 1000, -2910, -2610, 6, 6, "grass"],
        [315, -3115, 50, 50, 315, 715, -3315, -2915, 8, 8, "fire"],
    ]
    mov_plat_list = deepcopy(orig_mov_plat_list)

    headstart = 0
    y -= headstart

    frame_counter = 0
    color_counter = 0
    checkpoint_counter = 0
    death_counter = 0
    real_frame_counter = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        pygame.display.set_caption("Sora: Level 2")

        old_half_height = half_height
        if half_height:
            y -= height
            height *= 2
            half_height = False

        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
            height = height // 2
            y += height
            half_height = True

        old_x = x
        old_y = y

        if on_ground:
            if (pressed[pygame.K_UP] or pressed[pygame.K_w]) and not old_half_height:
                vel_y = jump_vel
            if on_ice:
                if vel_x > 0:
                    vel_x = 16
                elif vel_x < 0:
                    vel_x = -16
            else:
                if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
                    vel_x = -10 + on_mov_plat_vel_x
                elif pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
                    vel_x = +10 + on_mov_plat_vel_x
                elif abs(vel_x) < 3:
                    vel_x = on_mov_plat_vel_x
                elif vel_x > on_mov_plat_vel_x:
                    vel_x -= 3
                elif vel_x < on_mov_plat_vel_x:
                    vel_x += 3

        x += vel_x

        # plat_x, plat_y, plat_width, plat_height, plat_x1, plat_x2, plat_y1, plat_y2, plat_vel_x, plat_vel_y, plat_type = mov_plat
        for mov_plat in mov_plat_list:
            if mov_plat[0] <= mov_plat[4]:
                mov_plat[8] = abs(mov_plat[8])
            elif mov_plat[0] >= mov_plat[5]:
                mov_plat[8] = -abs(mov_plat[8])
            mov_plat[0] += mov_plat[8]

        for mov_plat in mov_plat_list:
            if mov_plat[1] <= mov_plat[6]:
                mov_plat[9] = abs(mov_plat[9])
            elif mov_plat[1] >= mov_plat[7]:
                mov_plat[9] = -abs(mov_plat[9])
            mov_plat[1] += mov_plat[9]

        if x <= 0:
            x = 0
            vel_x = -vel_x
        elif x >= 1280 - width:
            x = 1280 - width
            vel_x = -vel_x

        vel_y += 2
        y += vel_y

        on_ground = False
        on_ice = False
        on_mov_plat_vel_x = 0

        for i in range(len(plat_list)):
            plat_x, plat_y, plat_width, plat_height, plat_type = plat_list[i]
            if (
                plat_x - width < x < plat_x + plat_width
                and plat_y - height < y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    frame_counter = -1
                elif plat_type == "portal":
                    pass
                elif plat_type == "endportal":
                    done = True
                    break
                else:
                    if not plat_y - height < old_y < plat_y + plat_height:
                        if y < old_y:
                            y = plat_y + plat_height
                            vel_y = -vel_y // 2
                        elif y > old_y:
                            y = plat_y - height
                            vel_y = 0
                            on_ground = True
                    elif not plat_x - width < old_x < plat_x + plat_width:
                        vel_x = -vel_x
                        if x > old_x:
                            x = plat_x - width
                        elif x < old_x:
                            x = plat_x + plat_width
                    if (
                        old_half_height
                        and not half_height
                        and not y + height // 2 < plat_y + plat_height
                    ):
                        half_height = True
                        height = height // 2
                        y += height
                    if plat_type == "ice":
                        on_ice = True

        for i in range(len(mov_plat_list)):
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
            ) = mov_plat_list[i]
            # old_plat_x = old_mov_plat_x_list[i]
            if (
                plat_x - width < x < plat_x + plat_width
                and plat_y - height < y < plat_y + plat_height
            ):
                if plat_type == "fire":
                    frame_counter = -1
                else:
                    old_plat_x = plat_x - plat_vel_x
                    old_plat_y = plat_y - plat_vel_y
                    if not old_plat_y - height < old_y < old_plat_y + plat_height:
                        if plat_y - y > old_plat_y - old_y:
                            y = plat_y + plat_height
                            vel_y = -vel_y // 2
                        elif plat_y - y < old_plat_y - old_y:
                            y = plat_y - height
                            vel_y = plat_vel_y
                            on_ground = True
                            on_mov_plat_vel_x = plat_vel_x
                    elif not old_plat_x - width < old_x < old_plat_x + plat_width:
                        vel_x = -vel_x
                        if plat_x - x < old_plat_x - old_x:
                            x = plat_x - width
                        elif plat_x - x > old_plat_x - old_x:
                            x = plat_x + plat_width
                    if (
                        old_half_height
                        and not half_height
                        and not y + height // 2 < plat_y + plat_height
                    ):
                        half_height = True
                        height = height // 2
                        y += height
                    if plat_type == "ice":
                        on_ice = True

        if not half_height:
            for plat in plat_list:
                plat[1] += 400 - y
            for mov_plat in mov_plat_list:
                mov_plat[1] += 400 - y
                mov_plat[6] += 400 - y
                mov_plat[7] += 400 - y
            y = 400
        else:
            for plat in plat_list:
                plat[1] += 460 - y
            for mov_plat in mov_plat_list:
                mov_plat[1] += 460 - y
                mov_plat[6] += 460 - y
                mov_plat[7] += 460 - y
            y = 460

        if frame_counter == -1:
            death_counter += 1
            vel_x = 0
            vel_y = 0
            # check = plat_list[0][1] - 370
            plat_list = deepcopy(orig_plat_list)
            mov_plat_list = deepcopy(orig_mov_plat_list)
            # if 1000 < check:
            y = -2470
            x = 100
            # else:
            # y = 0
            checkpoint_counter = fps

        score = plat_list[0][1] - 370

        space_line = 20000
        if score >= space_line:
            screen.fill((0, 0, 0))
        else:
            screen.fill(
                (0, 162 * (1 - score / space_line), 232 * (1 - score / space_line))
            )

        if pressed[pygame.K_SPACE]:
            color_counter += 1
        else:
            color_counter = 0
        if color_counter == fps // 2:
            color_counter = 0
            is_green = not is_green

        if is_green:
            color = green
        else:
            color = red

        for plat in plat_list:
            plat_x, plat_y, plat_width, plat_height, plat_type = plat
            if 720 > plat_y > -plat_height:
                if plat_type == "grass":
                    pygame.draw.rect(
                        screen,
                        brown,
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        screen, grass_green, pygame.Rect(plat_x, plat_y, plat_width, 5)
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        screen,
                        red,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        screen,
                        blue,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "portal" or plat_type == "endportal":
                    pygame.draw.rect(
                        screen,
                        black,
                        pygame.Rect(
                            plat_x - 45, plat_y - 50, plat_width + 90, plat_height + 50
                        ),
                    )

        for mov_plat in mov_plat_list:
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
                        screen,
                        brown,
                        pygame.Rect(plat_x, plat_y + 5, plat_width, plat_height - 5),
                    )
                    pygame.draw.rect(
                        screen, grass_green, pygame.Rect(plat_x, plat_y, plat_width, 5)
                    )
                elif plat_type == "fire":
                    pygame.draw.rect(
                        screen,
                        red,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )
                elif plat_type == "ice":
                    pygame.draw.rect(
                        screen,
                        blue,
                        pygame.Rect(plat_x, plat_y, plat_width, plat_height),
                    )

        pygame.draw.rect(screen, color, pygame.Rect(x, y, width, height))

        scoretext = arialblack.render("Score: " + str(score), True, black)
        scoreRect = scoretext.get_rect()
        scoreRect.center = (640, 75)
        screen.blit(scoretext, scoreRect)

        deathtext = arialblack.render("Deaths: " + str(death_counter), True, black)
        deathRect = deathtext.get_rect()
        deathRect.center = (150, 75)
        screen.blit(deathtext, deathRect)

        timetext = arialblack.render(
            "Time: " + str(real_frame_counter // fps) + "s", True, black
        )
        timeRect = timetext.get_rect()
        timeRect.center = (1100, 75)
        screen.blit(timetext, timeRect)

        if score < 250:
            note = comicsans.render("LEVEL 2", True, black)
        elif 250 < score < 600:
            note = comicsans.render("Go Above", True, black)
        else:
            note = False

        if checkpoint_counter:
            checkpoint_counter -= 1
            if night:
                note = comicsans.render("Checkpoint Activated", True, white)
            else:
                note = comicsans.render("Checkpoint Activated", True, black)

        if note:
            noteRect = note.get_rect()
            noteRect.center = (640, 150)
            screen.blit(note, noteRect)

        frame_counter += 1
        real_frame_counter += 1

        clock.tick(fps)
        print(clock.get_time())
        pygame.display.flip()
