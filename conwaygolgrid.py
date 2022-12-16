"""Grid object"""
from copy import copy
from collections import namedtuple, deque
import pygame as pg
import numpy as np
from numba import njit


NUMBA_EMPTY_PLACEHOLDER = (-99, -99)

Settings = namedtuple("Settings", ["background_color", "grid_color", "new_color",
                                   "survivor_color", "dead_color", "cells", "neighbour_count",
                                   "new_cells", "survivor_cells", "survivor_duration", "dead_cells",
                                   "iteration"])

class ConwayGoLGrid():
    """Class representing the grid for Conway's Game of Life

    To construct, use the static method 'new' instead of the default public constructor
    for built-in validation of the values
    """

    def __init__(self, cell_size: int, width: int, height: int, surface: pg.Surface,
                 background_color: tuple[int, int, int], grid_color: tuple[int, int, int],
                 new_color: tuple[int, int, int], survivor_color: tuple[int, int, int],
                 dead_color: tuple[int, int, int], max_backups: int):
        self.__cell_size = cell_size
        self.__width = width
        self.__height = height
        self.__surface = surface
        self.__background_color = background_color
        self.__grid_color = grid_color
        self.__new_color = new_color
        self.__survivor_color = survivor_color
        self.__dead_color = dead_color

        self.__cells: np.ndarray = np.ndarray([])
        self.__neighbour_count: np.ndarray = np.ndarray([])
        self.__new_cells: set[tuple[(int, int)]] = set()
        self.__survivor_cells: set[tuple[(int, int)]] = set()
        self.__survivor_duration: dict[tuple[int, int], int] = {}
        self.__dead_cells: set[tuple[(int, int)]] = set()
        self.__iteration: int = 0
        self.__backups = deque(maxlen=max_backups)
        self.reset()

    @property
    def cell_size(self) -> int:
        """The size of each square cell in the grid"""
        return self.__cell_size

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the grid"""
        return self.__cells.shape

    @property
    def iteration(self) -> int:
        """The amount of iterations passed"""
        return self.__iteration

    @property
    def alive_count(self) -> int:
        """The amount of alive cells in the grid"""
        return np.count_nonzero(self.__cells)

    @property
    def alive_percentage(self) -> int:
        """The amount of alive cells in the grid"""
        return np.count_nonzero(self.__cells) / self.__cells.size * 100

    @property
    def surface(self) -> pg.Surface:
        """The surface the game is drawn onto"""
        return self.__surface

    @staticmethod
    def new(cell_size: int, width: int, height: int, screen: pg.Surface,
            background_color: tuple[int, int, int], grid_color: tuple[int, int, int],
            new_color: tuple[int, int, int], survivor_color: tuple[int, int, int],
            dead_color: tuple[int, int, int], max_backups: int):
        """Creates a new grid after successful validation of values"""
        # possible todo: validate the values sent in, wrap the value in a "Result" object
        # holding not only the Grid if succesfully created, but also other values such
        # as a bool value signifying successful creation, a list of errors/warnings, ...
        return ConwayGoLGrid(cell_size, width, height, screen, background_color, grid_color,
                             new_color, survivor_color, dead_color, max_backups)

    def update(self) -> None:
        """Updates the grid to the next step in the iteration, following Conway's Game
        of Life rules. Evaluates the grid, and redraws all changed cells"""
        self.__store_state()
        (updated_cells,
         cells_to_redraw) = ConwayGoLGrid.__perform_update(self.__cells, self.__neighbour_count,
                                                           self.__new_cells, self.__survivor_cells,
                                                           self.__dead_cells, self.__new_color,
                                                           self.__survivor_color, self.__dead_color,
                                                           self.__background_color)
        self.__cells = updated_cells
        self.__update_survivor_duration()
        self.__iteration += 1
        self.__draw_cells(cells_to_redraw)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __perform_update(cells: np.ndarray, neighbour_count: np.ndarray,
                         new_cells: set[tuple[int, int]], survivor_cells: set[tuple[int, int]],
                         dead_cells: set[tuple[int, int]], new_color: tuple[int, int, int],
                         survivor_color: tuple[int, int, int], dead_color: tuple[int, int, int],
                         background_color: tuple[int, int, int]
                         ) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int, int]]]]:
        """Updates the grid to the next step in the iteration, following Conway's Game of Life
        rules. Evaluates each cell, and returns the list of cells to be redrawn and their colors
        """
        rows, columns = cells.shape
        # Grab the coordinates of the non-background cells
        active_cells = sorted(new_cells.union(survivor_cells).union(dead_cells))

        # Per active cell, grab the coordinates from surrounding cells, add them to a set
        # to be able to evaluate each cell once.
        cells_to_evaluate = set()
        for row, col in active_cells:
            for i in range(max(0, row-1), min(row+2, rows)):
                for j in range(max(0, col-1), min(col+2, columns)):
                    cells_to_evaluate.add((i, j))

        updated_cells = np.array([[False] * cells.shape[1]] * cells.shape[0])
        neighbour_count_to_update: list[tuple[tuple[int, int], int]] = []
        cells_to_redraw: list[tuple[int, int, tuple[int, int, int]]] = []
        for row, col in cells_to_evaluate:
            cell = (row, col)
            # Count the alive cells around the current cell
            alive_neighbours = neighbour_count[row, col]
            if cells[cell]:
                if alive_neighbours in (2, 3):
                    updated_cells[cell] = True
                    cells_to_redraw.append((row, col, survivor_color))
                    survivor_cells.add(cell)
                    new_cells.discard(cell)
                    dead_cells.discard(cell)
                else:
                    cells_to_redraw.append((row, col, dead_color))
                    neighbour_count_to_update.append((cell, -1))
                    dead_cells.add(cell)
                    new_cells.discard(cell)
                    survivor_cells.discard(cell)
            else:
                if alive_neighbours == 3:
                    updated_cells[cell] = True
                    cells_to_redraw.append((row, col, new_color))
                    neighbour_count_to_update.append((cell, 1))
                    new_cells.add(cell)
                    survivor_cells.discard(cell)
                    dead_cells.discard(cell)
                else:
                    cells_to_redraw.append((row, col, background_color))
                    dead_cells.discard(cell)

        for coordinates, delta in neighbour_count_to_update:
            row, col = coordinates
            for i in range(max(0, row-1), min(row+2, rows)):
                for j in range(max(0, col-1), min(col+2, columns)):
                    if (i, j) != coordinates:
                        neighbour_count[(i, j)] += delta

        return updated_cells, cells_to_redraw

    def reset(self) -> None:
        """Resets the entire grid"""
        self.__cells = np.full((self.__height // self.__cell_size,
                                self.__width // self.__cell_size),
                               False, dtype=bool)
        self.__neighbour_count = np.zeros(self.__cells.shape, dtype=int)
        # The initial entry NUMBA_EMPTY_PLACEHOLDER is added since Numba can not handle empty sets
        self.__new_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__survivor_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__survivor_duration = {}
        self.__dead_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__iteration = 0
        self.__backups = deque(maxlen=self.__backups.maxlen)
        self.__draw_grid()

    def resurrect_cell(self, coordinates: tuple[int, int]) -> None:
        """Brings a cell in the grid to life"""
        if (coordinates in self.__new_cells or coordinates in self.__survivor_cells
            or coordinates == NUMBA_EMPTY_PLACEHOLDER):
            return

        self.__store_state()
        self.__cells[coordinates] = True
        self.__update_neighbour_count(coordinates)
        self.__new_cells.add(coordinates)
        self.__dead_cells.discard(coordinates)
        self.__draw_cells([(coordinates[0], coordinates[1], self.__new_color)])

    def clear_cell(self, coordinates: tuple[int, int]) -> None:
        """Clears a cell from the grid"""
        if coordinates == NUMBA_EMPTY_PLACEHOLDER:
            return

        self.__store_state()
        if self.__cells[coordinates]:
            self.__cells[coordinates] = False
            self.__update_neighbour_count(coordinates)

        self.__new_cells.discard(coordinates)
        self.__survivor_cells.discard(coordinates)
        self.__survivor_duration.pop(coordinates, None)
        self.__dead_cells.discard(coordinates)

        self.__draw_cells([(coordinates[0], coordinates[1], self.__background_color)])

    def create_cell_layout(self, cells: np.ndarray) ->  None:
        """Creates a certain grid layout to continue the game from"""
        if cells.shape != self.__cells.shape:
            return

        self.__store_state()
        self.__cells = cells
        self.__neighbour_count = ConwayGoLGrid.__create_neighbour_count(self.__cells)
        self.__new_cells = set(map(tuple, np.transpose((self.__cells).nonzero())))
        self.__new_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        self.__survivor_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__survivor_duration = {}
        self.__dead_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__iteration = 0
        self.__draw_grid()

    def overlay_new_cells(self, new_cells: np.ndarray, redraw: bool) ->  None:
        """Overlay the grid with another grid, only taking over the alive cells"""
        if new_cells.shape != self.__cells.shape:
            return

        self.__store_state()
        self.__cells = np.logical_or(self.__cells, new_cells)
        self.__new_cells = set(map(tuple, np.transpose((self.__cells).nonzero())))
        self.__new_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        self.__survivor_cells.difference_update(self.__new_cells)
        self.__survivor_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        self.__dead_cells.difference_update(self.__new_cells)
        self.__dead_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        _ = [self.__survivor_duration.pop((row, col), None)
                for row, col in np.ndindex(new_cells.shape)
                    if new_cells[row, col]]

        self.__neighbour_count = ConwayGoLGrid.__create_neighbour_count(self.__cells)

        if redraw:
            self.__draw_grid()

    def wipe_survivors(self) -> None:
        """Wipe all survivor cells off the grid"""
        self.__store_state()
        for cell in self.__survivor_cells:
            if cell != NUMBA_EMPTY_PLACEHOLDER:
                self.__cells[cell] = False
                self.__update_neighbour_count(cell)
        self.__survivor_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__survivor_duration = {}
        self.__draw_grid()

    def purge_survivors(self, purge_trigger: int) -> None:
        """Purge all survivor cells that have not updated their status for the given
        number of iterations. Basically, kill off the stragglers"""
        self.__store_state()
        for cell in list(self.__survivor_duration.keys()):
            if self.__survivor_duration[cell] >= purge_trigger:
                self.__survivor_duration.pop(cell, None)
                if cell != NUMBA_EMPTY_PLACEHOLDER:
                    self.__cells[cell] = False
                self.__update_neighbour_count(cell)
                # We don't want to remove the NUMBA_EMPTY_PLACEHOLDER entry, or Numba will cry ;-)
                if cell != NUMBA_EMPTY_PLACEHOLDER:
                    self.__survivor_cells.discard(cell)
        self.__draw_grid()

    def invert(self) ->  None:
        """Turns all new cells into dead cells and vice versa"""
        self.__store_state()
        self.__new_cells, self.__dead_cells = self.__dead_cells, self.__new_cells
        for cell in self.__new_cells:
            if cell != NUMBA_EMPTY_PLACEHOLDER:
                self.__cells[cell] = True
                self.__update_neighbour_count(cell)
        for cell in self.__dead_cells:
            if cell != NUMBA_EMPTY_PLACEHOLDER:
                self.__cells[cell] = False
                self.__update_neighbour_count(cell)
        self.__draw_grid()

    def change_cell_size(self, value: int):
        """Change the size of each square cell in the grid"""
        if self.__cell_size != value:
            self.__cell_size = value
            self.reset()

    def reverse(self) -> None:
        """Reverse to the previous state of the grid if possible"""
        if len(self.__backups) == 0:
            return

        backup = self.__backups.pop()
        self.__background_color = backup.background_color
        self.__grid_color = backup.grid_color
        self.__new_color = backup.new_color
        self.__survivor_color = backup.survivor_color
        self.__dead_color = backup.dead_color
        self.__cells = copy(backup.cells)
        self.__neighbour_count = copy(backup.neighbour_count)
        self.__new_cells = copy(backup.new_cells)
        self.__survivor_cells = copy(backup.survivor_cells)
        self.__survivor_duration = copy(backup.survivor_duration)
        self.__dead_cells = copy(backup.dead_cells)
        self.__iteration = backup.iteration
        self.__draw_grid()

    def change_new_color(self, color: tuple[int, int, int]):
        """Changes the color of a new cell"""
        self.__store_state()
        self.__new_color = color
        self.__draw_new_cells()

    def change_survivor_color(self, color: tuple[int, int, int]):
        """Changes the color of a survivor cell"""
        self.__store_state()
        self.__survivor_color = color
        self.__draw_survivor_cells()

    def change_dead_color(self, color: tuple[int, int, int]):
        """Changes the color of a dead cell"""
        self.__store_state()
        self.__dead_color = color
        self.__draw_dead_cells()

    def change_all_colors(self, colors: tuple[tuple[int, int, int],
                                              tuple[int, int, int],
                                              tuple[int, int, int]]):
        """Changes the color of all cells"""
        self.__store_state()
        self.__new_color = colors[0]
        self.__survivor_color = colors[1]
        self.__dead_color = colors[2]

        self.__draw_all_cells()

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __create_neighbour_count(cells: np.ndarray) -> None:
        """Create the neighbour count for all cells in the grid"""
        (rows, columns) = cells.shape
        neighbour_count = np.array([[0] * columns] * rows)
        for row, col in np.ndindex((rows, columns)):
            coordinates = (row, col)
            if cells[coordinates]:
                delta = 1 if cells[coordinates] else -1
                for i in range(max(0, row-1), min(row+2, rows)):
                    for j in range(max(0, col-1), min(col+2, columns)):
                        if (i, j) != coordinates:
                            neighbour_count[(i, j)] += delta
        return neighbour_count

    def __update_neighbour_count(self, updated_cell_coordinates: tuple[int, int]) -> None:
        """Update the neighbours of the cell with the given coordinates, and update each
        of their neighbour count values, depending on the value of the given cell"""
        if updated_cell_coordinates == NUMBA_EMPTY_PLACEHOLDER:
            return

        delta = 1 if self.__cells[updated_cell_coordinates] else -1
        ConwayGoLGrid.__perform_update_neighbour_count(self.__neighbour_count,
                                                       updated_cell_coordinates, delta)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __perform_update_neighbour_count(neighbour_count: np.ndarray,
                                         updated_cell_coordinates: tuple[int, int],
                                         delta: int) -> None:
        rows, columns = neighbour_count.shape
        row, col = updated_cell_coordinates
        for i in range(max(0, row-1), min(row+2, rows)):
            for j in range(max(0, col-1), min(col+2, columns)):
                if (i, j) != updated_cell_coordinates:
                    neighbour_count[(i, j)] += delta

    def __store_state(self) -> None:
        """Store the current  state of the grid to be able to reverse"""
        if len(self.__backups) > 0 and len(self.__backups) >= self.__backups.maxlen:
            self.__backups.popleft()

        backup = Settings(self.__background_color, self.__grid_color, self.__new_color,
                          self.__survivor_color, self.__dead_color, copy(self.__cells),
                          copy(self.__neighbour_count), copy(self.__new_cells),
                          copy(self.__survivor_cells), copy(self.__survivor_duration),
                          copy(self.__dead_cells), self.__iteration)
        self.__backups.append(backup)

    def __update_survivor_duration(self) -> None:
        """Update the dictionary holding the lifetime of survivor cells"""
        # Whatever sits in survivor_duration but not in surivor_cells can be removed
        # These are cells that might have been blanked out or changed status during update
        # survivor_cells is the source of truth here
        _ = [self.__survivor_duration.pop(cell, None)
                for cell in list(self.__survivor_duration.keys())
                    if cell not in self.__survivor_cells]
        # Now add/increment cells from surivor_cells in survivor_duration
        for survivor in self.__survivor_cells:
            if survivor in self.__survivor_duration:
                self.__survivor_duration[survivor] += 1
            else:
                self.__survivor_duration[survivor] = 0

    def __draw_grid(self) -> None:
        """Draws the grid"""
        self.__surface.fill(self.__background_color)
        self.__draw_gridlines()
        self.__draw_all_cells()

    def __draw_gridlines(self) -> None:
        """Draws the gridlines"""
        _ = [pg.draw.line(self.__surface, self.__grid_color, (x, 0), (x, self.__height))
             for x in range(0, self.__width, self.__cell_size)]
        _ = [pg.draw.line(self.__surface, self.__grid_color, (0, y), (self.__width, y))
             for y in range(0, self.__height, self.__cell_size)]

    def __draw_cells(self, cells_to_redraw: list[tuple[int, int, tuple[int, int, int]]]) -> None:
        """Draws the given cells in the given color"""
        for row, column, color in cells_to_redraw:
            dimensions = (column * self.__cell_size + 1, row * self.__cell_size + 1,
                          self.__cell_size - 1, self.__cell_size - 1)
            pg.draw.rect(self.__surface, color, dimensions)

    def __draw_all_cells(self) -> None:
        """Draws all cells in the grid"""
        self.__draw_new_cells()
        self.__draw_survivor_cells()
        self.__draw_dead_cells()

    def __draw_new_cells(self) -> None:
        """Draws the new cells in the grid"""
        cells_to_redraw = [(i, j, self.__new_color) for i, j in self.__new_cells]
        self.__draw_cells(cells_to_redraw)

    def __draw_survivor_cells(self) -> None:
        """Draws the survivor cells in the grid"""
        cells_to_redraw = [(i, j, self.__survivor_color) for i, j in self.__survivor_cells]
        self.__draw_cells(cells_to_redraw)

    def __draw_dead_cells(self) -> None:
        """Draws the dead cells in the grid"""
        cells_to_redraw = [(i, j, self.__dead_color) for i, j in self.__dead_cells]
        self.__draw_cells(cells_to_redraw)
