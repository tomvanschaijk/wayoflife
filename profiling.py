"""Purely here for benchmarking"""
import numpy as np
import pygame as pg
from line_profiler import LineProfiler
from conwaygolgrid import ConwayGoLGrid, CellSize

background_color = (5, 5, 5)
new_color = (50, 250, 5)
survivor_color = (40, 100, 40)
dead_color = (60, 0, 0)


def create_grid() -> tuple[ConwayGoLGrid, pg.Surface]:
    """Create the grid to profile"""
    width, height = 800, 600
    grid_color = (20, 20, 20)

    pg.init()
    screen = pg.display.set_mode((width, height))
    grid = ConwayGoLGrid.new(CellSize.XS, width, height, screen, background_color, grid_color,
                             new_color, survivor_color, dead_color, 1)
    return grid, screen


def update_screen(grid: ConwayGoLGrid, screen: pg.Surface) -> None:
    """Update the screen"""
    screen_rect = screen.get_rect()
    screen.set_clip(screen_rect)
    screen.blit(grid.surface, screen_rect)
    pg.display.flip()


def profile_update(grid: ConwayGoLGrid, screen: pg.Surface, file) -> None:
    """Profile the perform_update function"""
    profiler = LineProfiler()

    wrapped = profiler(grid._ConwayGoLGrid__perform_update)
    wrapped(grid._ConwayGoLGrid__cells, grid._ConwayGoLGrid__neighbour_count,
            grid._ConwayGoLGrid__new_cells, grid._ConwayGoLGrid__survivor_cells,
            grid._ConwayGoLGrid__dead_cells, new_color, survivor_color,
            dead_color, background_color)

    update_screen(grid, screen)

    profiler.print_stats(file)


def main() -> None:
    """Run the profiling"""
    file_name = "results.txt"

    try:
        grid, screen = create_grid()
        cells = np.random.choice([False, True], grid.shape, p=[0.93, 0.07])
        grid.create_cell_layout(cells)
        update_screen(grid, screen)

        with open(file_name, mode="w", encoding="utf-8") as output_file:
            profile_update(grid, screen, output_file)
            pg.quit()
            print(f"Output written to {file_name}")
    except IOError:
        print(f"Output could not be written to {file_name}")

    try:
        with open(file_name, mode="r", encoding="utf-8") as output_file:
            print(output_file.read())
    except IOError:
        print(f"File '{file_name}' not found")


if __name__ == "__main__":
    main()
