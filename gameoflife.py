"""
My implementation of Conway's Game of Life. The twist in this implementation is that it
doesn't only consider cells to be alive or dead. A cell can have several states:
New: a cell that just became alive because of the game rules
Survivor: an alive cell that hasn't changed state in the current iteration
Dead: a celll that is marked as dead, and will be removed from the game in the next iteration

This small change in how cells are visualised allows you to follow the evolution of each
cell and the environment as a whole. Coupled with that, there are a number of inputs you
can do to interact with the environment:
* the game can be paused at any time
* a random cell layout can be generated
* new cells can be injected into the grid
* alive and dead cells can switch places
* all surviving cells can be removed
* surviving cells that haven't change states in x iterations can be removed
* while the game is paused, you can move through the states one step at a time
* it's possible to revert to x amount of previous steps
* colors of all cells can be changed or reset to default values
* cell size can be changed to a predefined set of sizes
* target framerate can be changed
"""
from random import randint
import pygame as pg
import numpy as np
from conwaygolgrid import ConwayGoLGrid


DEFAULT_BACKGROUND_COLOR = (5, 5, 5)
DEFAULT_GRID_COLOR = (20, 20, 20)
DEFAULT_NEW_COLOR = (50, 250, 5)
DEFAULT_SURVIVOR_COLOR = (40, 100, 40)
DEFAULT_DEAD_COLOR = (60, 0, 0)
MIN_FPS, MAX_FPS, DEFAULT_FPS = 5, 120, 60
WIDTH, HEIGHT = 1900, 1200
CELL_SIZES, DEFAULT_CELL_SIZE = [3, 5, 10, 20, 50], 5
HELP_MENU_DIMENSIONS = (550, 660)
HELP_MENU_TOPLEFT = (20, 20)
HELP_MENU_COLOR, HELP_MENU_ALPHA = (150, 150, 150), 200
STATS_COLOR = (100, 100, 100)
PURGE_TRIGGER = 100
MAX_PREVIOUS_GRID_STATES = 10
HELP_MENU_TEXT = [
    "F1 - toggle this menu",
    "F2 - go fullscreen",
    "F3 - make the window resizable",
    "F4 - toggle stats display",
    "Q - quit the game",
    "SPACEBAR - start/pause",
    "ENTER - reset the game (watch out, there's no coming back from this)",
    "LCLICK - create a new cell at the mouse position",
    "RCLICK - clear the cell at the mouse position",
    "C - create a new random grid layout",
    "I - inverts new and dead cells: new cells die, dead cells spring to life!",
    "L - burst of life: sends in a random number of new cells to the grid",
    "M - 'mercy' killing..? wipe out all survivors",
    f"P - purge out any survivors that have not changed status in {PURGE_TRIGGER} iterations",
    "F - move the game forward 1 step (only when paused)",
    f"W - wait what..? back up a step (max {MAX_PREVIOUS_GRID_STATES} steps, only when paused)",
    "N - change the color of new cells",
    "S - change the color of survivor cells",
    "D - change the color of dead cells",
    "A - change the color of all cells",
    "R - reset the colors of all cells to their default values",
    f"↑ - reset the game and increase the cell size (max {CELL_SIZES[len(CELL_SIZES)-1]})",
    f"↓ - reset the game and decrease the cell size (min {CELL_SIZES[0]})",
    f"MWHEELUP - increase the target framerate (max {MAX_FPS})",
    f"MWHEELDOWN - decrease the target framerate (min {MIN_FPS})",
    ]


def initialize(width: int, height: int, cell_size: int
               ) -> tuple[ConwayGoLGrid, pg.Surface, pg.Surface, pg.time.Clock]:
    """Initialize all we need to start running the game"""
    pg.init()
    pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN])
    pg.display.set_caption("Game of Life")
    screen = pg.display.set_mode((width, height), pg.RESIZABLE)
    help_menu = create_help_menu()
    grid = ConwayGoLGrid.new(cell_size, width, height, pg.Surface((width, height)),
                             DEFAULT_BACKGROUND_COLOR, DEFAULT_GRID_COLOR, DEFAULT_NEW_COLOR,
                             DEFAULT_SURVIVOR_COLOR, DEFAULT_DEAD_COLOR, MAX_PREVIOUS_GRID_STATES)
    return grid, screen, help_menu, pg.time.Clock()


def create_help_menu() -> pg.Surface:
    """Create the help menu"""
    help_menu = pg.Surface(HELP_MENU_DIMENSIONS)
    help_menu.fill(HELP_MENU_COLOR)
    help_menu.set_alpha(HELP_MENU_ALPHA)
    font_name = pg.font.match_font("calibri")
    font = pg.font.Font(font_name, 24, bold=True)
    text = "How to control the game"
    text_surface = font.render(text, True, (0, 0, 0))
    help_menu.blit(text_surface, (10, 10))

    for i, text in enumerate(HELP_MENU_TEXT):
        font = pg.font.Font(font_name, 16, bold=True)
        text_surface = font.render(text, True, (0, 0, 0))
        help_menu.blit(text_surface, (50, 40 + ((i+1) * 24)))
    return help_menu


def update_stats_display(text: str) -> None:
    """Updates the caption"""
    stats_display = pg.Surface((500, 30))
    stats_display.fill(STATS_COLOR)
    font_name = pg.font.match_font("calibri")
    font = pg.font.Font(font_name, 14, bold=True)
    text_surface = font.render(text, True, (0, 0, 0))
    stats_display.blit(text_surface, (5, 8))

    return stats_display


def load_random_cell_layout(grid: ConwayGoLGrid) -> None:
    """Load up the grid with a random set of live cells"""
    cells = np.random.choice([False, True], grid.shape, p=[0.93, 0.07])
    grid.create_cell_layout(cells)


def lifewave(grid: ConwayGoLGrid, running: bool) -> None:
    """Send a wave of random live cells to the grid"""
    cells = np.random.choice([False, True], grid.shape, p=[0.97, 0.03])
    grid.overlay_new_cells(cells, not running)


def change_grid_size(grid: ConwayGoLGrid, running: bool, direction: int) -> None:
    """Changes the size of the grid and resets the game.
    If the game is running, a new set of live cells will start the game"""
    index = CELL_SIZES.index(grid.cell_size)
    if index + direction < 0 or index + direction >= len(CELL_SIZES) :
        return

    grid.change_cell_size(CELL_SIZES[index + direction])
    if running:
        load_random_cell_layout(grid)


def reset_colors(grid: ConwayGoLGrid) -> None:
    """Resets the grid colors to their default values"""
    grid.change_new_color(DEFAULT_NEW_COLOR)
    grid.change_survivor_color(DEFAULT_SURVIVOR_COLOR)
    grid.change_dead_color(DEFAULT_DEAD_COLOR)


def handle_events(grid: ConwayGoLGrid, running: bool, draw_menu: bool,
                  draw_stats: bool, fps: int) -> tuple[bool, bool, bool, int]:
    """Handling the PyGame events in the main loop"""
    for event in pg.event.get():
        match event.type:
            case pg.QUIT:
                return False, False, False, True, fps
            case pg.MOUSEBUTTONDOWN:
                change = 5 if fps < 30 else 10
                if event.button == 4:
                    fps = min(fps + change, MAX_FPS)
                if event.button == 5:
                    fps = max(MIN_FPS, fps - change)
            case pg.KEYDOWN:
                match event.key:
                    case pg.K_F1: draw_menu = not draw_menu
                    case pg.K_F2: pg.display.set_mode((WIDTH, HEIGHT), pg.FULLSCREEN)
                    case pg.K_F3: pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE)
                    case pg.K_F4: draw_stats = not draw_stats
                    case pg.K_q: return False, False, False, True, fps
                    case pg.K_SPACE: running = not running
                    case pg.K_RETURN:
                        running = False
                        grid.reset()
                    case pg.K_c: load_random_cell_layout(grid)
                    case pg.K_i: grid.invert()
                    case pg.K_l: lifewave(grid, running)
                    case pg.K_m: grid.wipe_survivors()
                    case pg.K_p: grid.purge_survivors(PURGE_TRIGGER)
                    case pg.K_f:
                        if not running:
                            grid.update()
                    case pg.K_w:
                        if not running:
                            grid.reverse()
                    case pg.K_n:
                        grid.change_new_color((randint(0,255), randint(0,255), randint(0,255)))
                    case pg.K_s:
                        grid.change_survivor_color((randint(0,255), randint(0,255), randint(0,255)))
                    case pg.K_d:
                        grid.change_dead_color((randint(0,255), randint(0,255), randint(0,255)))
                    case pg.K_a:
                        grid.change_all_colors(((randint(0,255), randint(0,255), randint(0,255)),
                                                (randint(0,255), randint(0,255), randint(0,255)),
                                                (randint(0,255), randint(0,255), randint(0,255))))
                    case pg.K_r: reset_colors(grid)
                    case pg.K_UP: change_grid_size(grid, running, 1)
                    case pg.K_DOWN: change_grid_size(grid, running, -1)

        mouse_button = pg.mouse.get_pressed()
        if mouse_button[0] or mouse_button[2]:
            mouse_position = pg.mouse.get_pos()
            coordinates = (mouse_position[1] // grid.cell_size, mouse_position[0] // grid.cell_size)
            if mouse_button[0]:
                grid.resurrect_cell(coordinates)
            else:
                grid.clear_cell(coordinates)

    return running, draw_menu, draw_stats, False, fps


def draw_surfaces(screen: pg.Surface, grid: pg.Surface, help_menu: pg.Surface,
                  draw_menu: bool, draw_stats: bool, stats_text: str) -> None:
    """Draws all surfaces to the screen"""
    screen_rect = screen.get_rect()
    screen.set_clip(screen_rect)
    screen.blit(grid, screen_rect)

    if draw_menu:
        menu_rect = help_menu.get_rect()
        menu_rect.topleft = HELP_MENU_TOPLEFT
        screen.set_clip(menu_rect)
        screen.blit(help_menu, menu_rect)

    if draw_stats:
        stats = update_stats_display(stats_text)
        stats_rect = stats.get_rect()
        stats_rect.topleft = (5, screen_rect.height - stats_rect.height - 5)
        screen.set_clip(stats_rect)
        screen.blit(stats, stats_rect)


def main():
    """Main function"""
    grid, screen, help_menu, clock = initialize(WIDTH, HEIGHT, DEFAULT_CELL_SIZE)
    fps = DEFAULT_FPS
    running = False
    draw_menu = False
    draw_stats = True
    while True:
        (running, draw_menu, draw_stats, exit_program,
         fps) = handle_events(grid, running, draw_menu, draw_stats, fps)

        if exit_program:
            pg.quit()
            break

        if running:
            grid.update()
            clock.tick(fps)

        stats_text = (f"actual/target fps: {int(clock.get_fps())}/{fps}" +
                      f" - cell size: {grid.cell_size}x{grid.cell_size}" +
                      f" - iteration: {grid.iteration}" +
                      f" - alive cells: {grid.alive_count} ({grid.alive_percentage:.1f}%)")
        draw_surfaces(screen, grid.surface, help_menu, draw_menu, draw_stats, stats_text)
        pg.display.flip()


if __name__ == "__main__":
    main()
