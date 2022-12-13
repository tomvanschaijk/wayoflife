"""
My implementation of Conway's Game of Life
"""
import pygame as pg
import numpy as np
from conwaygolgrid import ConwayGoLGrid


BACKGROUND_COLOR = (5, 5, 5)
GRID_COLOR = (20, 20, 20)
NEW_COLOR = (50, 250, 5)
SURVIVOR_COLOR = (40, 100, 40)
DEAD_COLOR = (60, 0, 0)
STATS_COLOR, STATS_ALPHA = (100, 100, 100), 180
FPS = 60
WIDTH, HEIGHT = 1920, 1200
CELL_SIZE = 10


def initialize(width: int, height: int, cell_size: int
               ) -> tuple[ConwayGoLGrid, pg.Surface, pg.time.Clock]:
    """Initialize all we need to start running the game"""
    pg.init()
    pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN])
    pg.display.set_caption("Game of Life")
    screen = pg.display.set_mode((width, height))
    grid = ConwayGoLGrid.new(cell_size, width, height, BACKGROUND_COLOR, GRID_COLOR,
                             NEW_COLOR, SURVIVOR_COLOR, DEAD_COLOR)
    return grid, screen, pg.time.Clock()


def load_random_cell_layout(grid: ConwayGoLGrid) -> None:
    """Load up the grid with a random set of live cells"""
    cells = np.random.choice([0, 1], grid.shape, p=[0.93, 0.07])
    grid.create_cell_layout(cells)


def handle_events(grid: ConwayGoLGrid, running: bool) -> tuple[bool, bool]:
    """Handling the PyGame events in the main loop"""
    for event in pg.event.get():
        match event.type:
            case pg.QUIT:
                return running, True
            case pg.KEYDOWN:
                match event.key:
                    case pg.K_SPACE:
                        running = not running
                    case pg.K_c: load_random_cell_layout(grid)

        mouse_button = pg.mouse.get_pressed()
        if mouse_button[0] or mouse_button[2]:
            mouse_position = pg.mouse.get_pos()
            coordinates = (mouse_position[1] // grid.cell_size, mouse_position[0] // grid.cell_size)
            if mouse_button[0]:
                grid.resurrect_cell(coordinates)
            else:
                grid.clear_cell(coordinates)

    return running, False


def update_stats_display(text: str) -> None:
    """Updates the caption"""
    stats_display = pg.Surface((220, 20))
    stats_display.fill(STATS_COLOR)
    stats_display.set_alpha(STATS_ALPHA)
    font_name = pg.font.match_font("calibri")
    font = pg.font.Font(font_name, 12, bold=True)
    text_surface = font.render(text, True, (0, 0, 0))
    stats_display.blit(text_surface, (5, 5))

    return stats_display


def draw_surfaces(screen: pg.Surface, grid: pg.Surface, stats_text: str) -> None:
    """Draws all surfaces to the screen"""
    screen_rect = screen.get_rect()
    screen.set_clip(screen_rect)
    screen.blit(grid, screen_rect)

    stats = update_stats_display(stats_text)
    stats_rect = stats.get_rect()
    stats_rect.topleft = (5, screen_rect.height - stats_rect.height - 5)
    screen.set_clip(stats_rect)
    screen.blit(stats, stats_rect)


def main():
    """Main function"""
    grid, screen, clock = initialize(WIDTH, HEIGHT, CELL_SIZE)
    running = False
    while True:
        running, exit_program = handle_events(grid, running)

        if exit_program:
            pg.quit()
            break

        if running:
            grid.update()
            clock.tick(FPS)

        stats_text = (f"actual/target fps: {int(clock.get_fps())}/{FPS}" +
                      f" - cell size: {grid.cell_size}x{grid.cell_size}")
        draw_surfaces(screen, grid.surface, stats_text)
        pg.display.flip()


if __name__ == "__main__":
    main()
