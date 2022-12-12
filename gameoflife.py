"""
My implementation of Conway's Game of Life
"""
import pygame as pg
import numpy as np


BACKGROUND_COLOR = (5, 5, 5)
GRID_COLOR = (20, 20, 20)
NEW_COLOR = (50, 250, 5)
SURVIVOR_COLOR = (40, 100, 40)
DEAD_COLOR = (60, 0, 0)
FPS = 60
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10


def initialize(width: int, height: int, cell_size: int
               ) -> tuple[np.ndarray, pg.Surface, pg.time.Clock]:
    """Initialize all we need to start running the game"""
    pg.init()
    pg.display.set_caption("Game of Life")
    screen = pg.display.set_mode((width, height))
    screen.fill(GRID_COLOR)

    columns, rows = width // cell_size, height // cell_size
    cells = np.zeros((rows, columns), dtype=int)

    return cells, screen, pg.time.Clock()


def update(cells, cell_size, screen, running):
    """Update the screen"""
    updated_cells = np.zeros((cells.shape[0], cells.shape[1]), dtype=int)

    for row, col in np.ndindex(cells.shape):
        alive_neighbours = np.sum(cells[row-1:row+2, col-1:col+2]) - cells[row, col]
        color = BACKGROUND_COLOR if cells[row, col] == 0 else NEW_COLOR

        if cells[row, col] == 1:
            if alive_neighbours == 2 or alive_neighbours == 3:
                updated_cells[row, col] = 1
                if running:
                    color = SURVIVOR_COLOR
            else:
                if running:
                    color = DEAD_COLOR
        else:
            if alive_neighbours == 3:
                updated_cells[row, col] = 1
                if running:
                    color = NEW_COLOR

        pg.draw.rect(screen, color, (col * cell_size, row * cell_size,
                                     cell_size - 1, cell_size - 1))

    return updated_cells


def handle_events(cells: np.ndarray, cell_size: int, screen: pg.Surface,
                  running: bool) -> tuple[bool, bool]:
    """Handling the PyGame events in the main loop"""
    for event in pg.event.get():
        if event.type == pg.QUIT:
            return running, True

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                running = not running
                update(cells, cell_size, screen, running)

        mouse_pressed = pg.mouse.get_pressed()
        if mouse_pressed[0] or mouse_pressed[2]:
            mouse_position = pg.mouse.get_pos()
            coordinates = (mouse_position[1] // cell_size, mouse_position[0] // cell_size)
            cells[coordinates] = 1 if mouse_pressed[0] else 0
            update(cells, cell_size, screen, running)

    return running, False


def main():
    """Main function"""
    cells, screen, clock = initialize(WIDTH, HEIGHT, CELL_SIZE)
    running = False
    cells = update(cells, CELL_SIZE, screen, running)
    while True:
        running, exit_program = handle_events(cells, CELL_SIZE, screen, running)

        if exit_program:
            pg.quit()
            break

        if running:
            cells = update(cells, CELL_SIZE, screen, running)
            clock.tick(FPS)

        pg.display.flip()


if __name__ == "__main__":
    main()
