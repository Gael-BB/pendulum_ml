import pygame
import numpy as np

class PendulumEnv:
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 30
    DT = 1 / FPS
    SCALE = 200
    G = 9.81
    L = 1
    THETA_DOT_DECAY = 0.99

    CART_WIDTH = 80
    CART_HEIGHT = 40
    WHEEL_RADIUS = 12

    def __init__(self, do_render=True, manual_control=False):
        self.do_render = do_render
        self.manual_control = manual_control

        # theta, theta_dot, x, x_dot
        self.state = np.zeros(4)
        self.done = False

        # Define action space and observation space
        self.action_space = 3
        self.observation_space = (4,)  # Full state: theta, theta_dot, x, x_dot

        # Pygame setup
        if do_render:
            self.render_setup()

    def render_setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Pendulum Balancer")
        self.clock = pygame.time.Clock()

    def reset(self):
        # Initialize state and reset flags
        self.state = np.zeros(4)
        self.done = False
        return self.state, 0

    def step(self, action):
        if self.manual_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                action = 2
            elif keys[pygame.K_LEFT]:
                action = 0
            else:
                action = 1

        # Action logic
        x_dot_dot = (action - 1) * 10
        self.state[1] += (-3 / self.L * np.cos(self.state[0]) * x_dot_dot -
                          3 * self.G / 2 / self.L * np.sin(self.state[0])) * self.DT
        self.state[0] += self.state[1] * self.DT
        self.state[3] += x_dot_dot * self.DT
        self.state[2] += self.state[3] * self.DT

        self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
        self.state[1] *= self.THETA_DOT_DECAY

        self.done = np.abs(
            self.state[2]) > self.SCREEN_WIDTH // self.SCALE // 2

        reward = 1 - np.cos(self.state[0])

        if self.do_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            self.render()

        return self.state, reward, self.done, {}, 0

    def render(self):
        self.screen.fill((255, 255, 255))
        x1 = self.SCREEN_WIDTH // 2 + self.SCALE * self.state[2]
        x2 = x1 + self.SCALE * self.L * np.sin(self.state[0])
        y1 = self.SCREEN_HEIGHT // 2
        y2 = self.SCREEN_HEIGHT // 2 + self.SCALE * self.L * np.cos(self.state[0])

        cart_x = x1 - self.CART_WIDTH // 2
        cart_y = y1 - self.CART_HEIGHT // 2

        pygame.draw.rect(self.screen, (255, 0, 0), (cart_x, cart_y, self.CART_WIDTH, self.CART_HEIGHT))
        pygame.draw.circle(self.screen, (100, 0, 0), (cart_x + 20, cart_y + self.CART_HEIGHT), self.WHEEL_RADIUS)
        pygame.draw.circle(self.screen, (100, 0, 0), (cart_x + self.CART_WIDTH - 20, cart_y + self.CART_HEIGHT), self.WHEEL_RADIUS)
        pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x2, y2), 5)
        
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        if self.do_render:
            pygame.quit()