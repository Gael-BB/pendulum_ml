import numpy as np
import pygame

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
ITERATION_PER_FRAME = 1
DT = 1 / FPS / ITERATION_PER_FRAME
SCALE = 300

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("PENDULUM BALANCER")

clock = pygame.time.Clock()


class Pendulum:
    def __init__(self, screen_width, screen_height, scale, g=9.81, l=1):
        self.theta = 0.1
        self.theta_dot = 0
        self.g = g
        self.l = l

        self.x = screen_width // 2
        self.x_dot = 0.1

        self.scale = scale
        self.y = 2 * screen_height // 3

    def control(self):
        """Compute control input for base acceleration."""
        k_p_theta = 10
        k_d_theta = 1
        k_p_x = 0
        k_d_x = 0

        return -k_p_theta * self.theta - k_d_theta * self.theta_dot + k_p_x * (self.x - SCREEN_WIDTH // 2) + k_d_x * self.x_dot

    def update(self, dt, iterations_per_frame):
        """Update the pendulum's state."""
        for _ in range(iterations_per_frame):
            x_dot_dot = self.control()

            self.theta_dot += (3 / self.l * np.cos(self.theta) * x_dot_dot +
                               3 * self.g / 2 / self.l * np.sin(self.theta)) * dt
            self.theta += self.theta_dot * dt

            self.x_dot -= x_dot_dot * dt
            self.x += self.x_dot * dt * self.scale

    def draw(self, screen):
        """Draw the pendulum."""
        screen.fill((255, 255, 255))
        x2 = self.x + self.scale * np.sin(self.theta)
        y2 = self.y - self.scale * np.cos(self.theta)
        pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), (x2, y2), 5)
        pygame.display.flip()


# Initialize the pendulum
pendulum = Pendulum(SCREEN_WIDTH, SCREEN_HEIGHT, SCALE)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    pendulum.update(DT, ITERATION_PER_FRAME)
    pendulum.draw(screen)
    clock.tick(FPS)
