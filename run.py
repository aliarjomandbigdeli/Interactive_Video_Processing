import numpy as np
import pygame
import random
import cv2

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 1/22/2019
"""


def run():
    # Initialize the game engine
    pygame.init()

    # Variables
    width = 640
    height = 480
    WHITE = [255, 255, 255]
    snow_list = []
    cam = cv2.VideoCapture(0)
    history = 600
    learning_rate = 1.0 / history

    # Set the height and width of the screen
    SIZE = [width, height]
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Interactive Rain Animation")

    # Loop 50 times and add a snow flake in a random x,y position
    for i in range(250):
        x = random.randrange(0, width)
        y = random.randrange(0, height)
        snow_list.append([x, y, 1])

    clock = pygame.time.Clock()
    water_drop = pygame.image.load("drop.png").convert_alpha()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    # Loop until the user clicks the close button.
    done = False
    while not done:

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        ret_val, img = cam.read()
        frame = get_frame(cam, 0.5)
        frame = cv2.flip(frame, 1)
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        background_sub = mask & frame
        gray = cv2.cvtColor(background_sub, cv2.COLOR_RGB2GRAY)
        ret, binary_thresh = cv2.threshold(gray, 63, 255, cv2.THRESH_BINARY)
        cv2.imshow('mask', mask)
        screen.fill([0, 0, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rot90(img)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))

        # Process each snow flake in the list
        for i in range(len(snow_list)):

            # Draw the snow flake
            if (draw_allowable(binary_thresh, snow_list[i][0], snow_list[i][1], 16, 16) and snow_list[i][2] == 1):
                screen.blit(water_drop, [snow_list[i][0], snow_list[i][1]])
            else:
                snow_list[i][2] = 0

            # Move the snow flake down one pixel
            snow_list[i][1] += 1

            # If the snow flake has moved off the bottom of the screen
            if snow_list[i][1] > height - 80:
                # Reset it just above the top
                y = random.randrange(-50, -10)
                snow_list[i][1] = y
                # Give it a new x position
                x = random.randrange(0, width)
                snow_list[i][0] = x
                snow_list[i][2] = 1

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        clock.tick(20)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(10)
        if c == 27:
            break

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor):
    # Read the current frame from the video capture object
    _, frame = cap.read()

    return frame


def draw_allowable(mask, x, y, w, h):
    if (y < 100 or y > 550):
        return True
    x = int(x + w / 2 - 2)
    y = int(y + h / 2 - 2)
    white_counter = 0
    for i in range(x, x + 4):
        for j in range(y, y + 4):
            if (i < mask.shape[1] and j < mask.shape[0]):
                if (int(mask[j, i]) != 0):
                    white_counter += 1
    if (white_counter >= 5):
        return False
    else:
        return True
    return True


if __name__ == '__main__':
    run()
