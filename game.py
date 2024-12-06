import numpy as np
import pygame

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
DT = 1 / FPS
SCALE = 200

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("PENDULUM BALANCER")

clock = pygame.time.Clock()


class Pendulum:
    def __init__(self, screen_width, screen_height, scale, g=9.81, l=1):
        self.theta = 0
        self.theta_dot = 0
        self.g = g
        self.l = l

        self.x = 0
        self.x_dot = 0

        self.scale = scale
        self.screen_width = screen_width
        self.screen_height = screen_height

    def control(self):
        """Compute control input for base acceleration."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            return 10
        elif keys[pygame.K_LEFT]:
            return -10
        else:
            return 0

    def update(self, dt):
        """Update the pendulum's state."""
        x_dot_dot = self.control()
        self.theta_dot += (-3 / self.l * np.cos(self.theta) * x_dot_dot -
                           3 * self.g / 2 / self.l * np.sin(self.theta)) * dt
        self.theta += self.theta_dot * dt
        self.x_dot += x_dot_dot * dt
        self.x += self.x_dot * dt

    def draw(self, screen):
        """Draw the pendulum."""
        screen.fill((255, 255, 255))
        x1 = self.screen_width // 2 + self.scale * self.x
        x2 = x1 + self.scale * self.l * np.sin(self.theta)
        y1 = self.screen_height // 2
        y2 = self.screen_height // 2 + self.scale * self.l * np.cos(self.theta)
        pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 5)
        pygame.display.flip()


# Initialize the pendulum
pendulum = Pendulum(SCREEN_WIDTH, SCREEN_HEIGHT, SCALE)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    pendulum.update(DT)
    pendulum.draw(screen)
    clock.tick(FPS)
