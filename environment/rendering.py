import pygame
import numpy as np

GRID = 10
CELL_SIZE = 60  # Each grid cell is 60x60 pixels
WINDOW_SIZE = GRID * CELL_SIZE

# Colors
COLOR_BG = (230, 230, 230)
COLOR_RANGER = (0, 128, 255)
COLOR_ANIMAL = (34, 139, 34)
COLOR_POACHER = (255, 0, 0)
COLOR_COVERAGE = (200, 200, 200)
COLOR_TEXT = (0, 0, 0)

pygame.init()
FONT = pygame.font.SysFont("Arial", 16)

class Renderer:
    def __init__(self, env):
        self.env = env
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("WildlifeLoop Simulation")
        self.clock = pygame.time.Clock()
        self.coverage_map = np.zeros((GRID, GRID))

    def update_coverage(self, visited):
        self.coverage_map = np.zeros((GRID, GRID))
        for zid in visited:
            y = zid // GRID
            x = zid % GRID
            self.coverage_map[y, x] = 1

    def render(self):
        self.screen.fill(COLOR_BG)

        # Draw coverage
        for y in range(GRID):
            for x in range(GRID):
                if self.coverage_map[y, x] > 0:
                    rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, COLOR_COVERAGE, rect)

        # Draw animals
        for a in self.env._animals:
            x, y = a * CELL_SIZE
            pygame.draw.circle(self.screen, COLOR_ANIMAL, (int(x+CELL_SIZE/2), int(y+CELL_SIZE/2)), 12)

        # Draw poachers
        for i, p_on in enumerate(self.env._poacher_on):
            if p_on:
                px, py = self.env.POACHER_SPOTS[i] * CELL_SIZE
                pygame.draw.rect(self.screen, COLOR_POACHER,
                                 pygame.Rect(int(px+CELL_SIZE/4), int(py+CELL_SIZE/4), CELL_SIZE//2, CELL_SIZE//2))

        # Draw ranger
        rx, ry = self.env._ranger * CELL_SIZE
        pygame.draw.circle(self.screen, COLOR_RANGER, (int(rx+CELL_SIZE/2), int(ry+CELL_SIZE/2)), 15)

        # Draw grid lines
        for i in range(GRID+1):
            pygame.draw.line(self.screen, (150, 150, 150), (i*CELL_SIZE,0), (i*CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(self.screen, (150, 150, 150), (0,i*CELL_SIZE), (WINDOW_SIZE, i*CELL_SIZE))

        # Draw info text
        info_lines = [
            f"Step: {self.env._step}/{self.env.MAX_STEPS}",
            f"Battery: {self.env._battery:.2f}",
            f"Coverage: {len(self.env._visited)}/{GRID*GRID}",
            f"Caught Poachers: {self.env._caught}",
            f"Missed: {self.env._missed}",
            f"False Alerts: {self.env._n_false}",
            f"Cumulative Reward: {self.env._cum_reward:.2f}"
        ]
        for i, line in enumerate(info_lines):
            text_surf = FONT.render(line, True, COLOR_TEXT)
            self.screen.blit(text_surf, (5, 5 + i*18))

        pygame.display.flip()
        self.clock.tick(self.env.metadata.get("render_fps", 10))

    def close(self):
        pygame.quit()