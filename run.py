import numpy as np
import pygame
import random
import cv2

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 1/22/2019
"""


def main():
    # Initialize the game engine
    pygame.init()

    # Variables
    width = 640
    height = 480
    drop_list = []
    cam = cv2.VideoCapture(0)
    history = 600
    learning_rate = 1.0 / history

    # Set the height and width of the screen
    SIZE = [width, height]
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Animated Rain")

    # Loop 50 times and add a rain flake in a random x,y position
    for i in range(250):
        x = random.randrange(0, width)
        y = random.randrange(0, height)
        drop_list.append([x, y, 1])

    clock = pygame.time.Clock()
    water_drop = pygame.image.load("drop.png").convert_alpha()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    # Loop until the user press 'Esc' key
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
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, binary_thresh = cv2.threshold(gray, 63, 255, cv2.THRESH_BINARY)
        cv2.imshow('binary', mask)
        screen.fill([0, 0, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rot90(img)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))

        # Process each drop flake in the list
        for i in range(len(drop_list)):

            # Draw the drops
            if (draw_allowable(binary_thresh, drop_list[i][0], drop_list[i][1], 16, 16) and drop_list[i][2] == 1):
                screen.blit(water_drop, [drop_list[i][0], drop_list[i][1]])
            else:
                drop_list[i][2] = 0

            # Move the drop flake down one pixel
            drop_list[i][1] += 1

            # If the drop flake has moved off the bottom of the screen
            if drop_list[i][1] > height - 80:
                # Reset it just above the top
                y = random.randrange(-50, -10)
                drop_list[i][1] = y
                # Give it a new x position
                x = random.randrange(0, width)
                drop_list[i][0] = x
                drop_list[i][2] = 1

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


def draw_allowable(mask, x, y, width, height):
    if (y < 100 or y > 550):
        return True
    x = int(x + width / 2 - 2)
    y = int(y + height / 2 - 2)
    white_pixels_counter = 0
    for i in range(x, x + 4):
        for j in range(y, y + 4):
            if (i < mask.shape[1] and j < mask.shape[0]):
                if (int(mask[j, i]) != 0):
                    white_pixels_counter += 1
    if (white_pixels_counter >= 5):
        return False
    else:
        return True
    return True


if __name__ == '__main__':
    main()
