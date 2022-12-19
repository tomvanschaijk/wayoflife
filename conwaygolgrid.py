"""Grid object"""
from copy import copy
from collections import namedtuple, deque
import pygame as pg
import numpy as np
from numba import njit


NUMBA_EMPTY_PLACEHOLDER = (-99, -99)

Settings = namedtuple("Settings", ["background_color", "grid_color", "cell_color",
                                   "cells", "alive_cells", "neighbour_count",
                                   "alive_duration", "iteration"])

class ConwayGoLGrid():
    """Class representing the grid for Conway's Game of Life

    To construct, use the static method 'new' instead of the default public constructor
    for built-in validation of the values
    """

    def __init__(self, cell_size: int, width: int, height: int, surface: pg.Surface,
                 background_color: tuple[int, int, int], grid_color: tuple[int, int, int],
                 cell_color: tuple[int, int, int], max_backups: int):
        self.__cell_size = cell_size
        self.__width = width
        self.__height = height
        self.__surface = surface
        self.__background_color = background_color
        self.__grid_color = grid_color
        self.__cell_color = cell_color

        self.__cells: np.ndarray = np.ndarray([])
        self.__neighbour_count: np.ndarray = np.ndarray([])
        self.__alive_cells: set[tuple[(int, int)]] = set()
        self.__alive_duration: dict[tuple[int, int], int] = {}
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
    def alive_percentage(self) -> float:
        """The amount of alive cells in the grid"""
        return np.count_nonzero(self.__cells) / self.__cells.size * 100

    @property
    def surface(self) -> pg.Surface:
        """The surface the game is drawn onto"""
        return self.__surface

    @staticmethod
    def new(cell_size: int, width: int, height: int, screen: pg.Surface,
            background_color: tuple[int, int, int], grid_color: tuple[int, int, int],
            cell_color: tuple[int, int, int], max_backups: int):
        """Creates a new grid after successful validation of values"""
        # possible todo: validate the values sent in, wrap the value in a "Result" object
        # holding not only the Grid if succesfully created, but also other values such
        # as a bool value signifying successful creation, a list of errors/warnings, ...
        return ConwayGoLGrid(cell_size, width, height, screen, background_color, grid_color,
                             cell_color, max_backups)

    def update(self) -> None:
        """Updates the grid to the next step in the iteration, following Conway's Game
        of Life rules. Evaluates the grid, and redraws all changed cells"""
        self.__store_state()
        (updated_cells,
         cells_to_redraw) = ConwayGoLGrid.__perform_update(self.__cells, self.__neighbour_count,
                                                           self.__alive_cells, self.__cell_color,
                                                           self.__background_color)
        self.__cells = updated_cells
        self.__update_alive_duration()
        self.__iteration += 1
        self.__draw_cells(cells_to_redraw)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __perform_update(cells: np.ndarray, neighbour_count: np.ndarray,
                         alive_cells: set[tuple[int, int]], cell_color: tuple[int, int, int],
                         background_color: tuple[int, int, int]
                         ) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int, int]]]]:
        """Updates the grid to the next step in the iteration, following Conway's Game of Life
        rules. Evaluates each cell, and returns the list of cells to be redrawn and their colors
        """
        rows, columns = cells.shape
        # Per alive cell, grab the coordinates from surrounding cells, add them to a set
        # to be able to evaluate each cell once.
        cells_to_evaluate = set([(i, j) for row, col in alive_cells
                                for i in range(max(0, row-1), min(row+2, rows))
                                    for j in range(max(0, col-1), min(col+2, columns))])

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
                    cells_to_redraw.append((row, col, cell_color))
                else:
                    cells_to_redraw.append((row, col, background_color))
                    neighbour_count_to_update.append((cell, -1))
                    alive_cells.discard(cell)
            else:
                if alive_neighbours == 3:
                    updated_cells[cell] = True
                    cells_to_redraw.append((row, col, cell_color))
                    neighbour_count_to_update.append((cell, 1))
                    alive_cells.add(cell)

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
        self.__alive_cells = set([NUMBA_EMPTY_PLACEHOLDER])
        self.__alive_duration = {}
        self.__iteration = 0
        self.__backups = deque(maxlen=self.__backups.maxlen)
        self.__draw_grid()

    def resurrect_cell(self, coordinates: tuple[int, int]) -> None:
        """Brings a cell in the grid to life"""
        if (coordinates in self.__alive_cells or coordinates == NUMBA_EMPTY_PLACEHOLDER):
            return

        self.__store_state()
        self.__cells[coordinates] = True
        self.__update_neighbour_count(coordinates)
        self.__alive_cells.add(coordinates)
        self.__draw_cells([(coordinates[0], coordinates[1], self.__cell_color)])

    def clear_cell(self, coordinates: tuple[int, int]) -> None:
        """Clears a cell from the grid"""
        if coordinates == NUMBA_EMPTY_PLACEHOLDER:
            return

        self.__store_state()
        if self.__cells[coordinates]:
            self.__cells[coordinates] = False
            self.__update_neighbour_count(coordinates)

        self.__alive_cells.discard(coordinates)
        self.__alive_duration.pop(coordinates, None)

        self.__draw_cells([(coordinates[0], coordinates[1], self.__background_color)])

    def create_cell_layout(self, cells: np.ndarray) ->  None:
        """Creates a certain grid layout to continue the game from"""
        if cells.shape != self.__cells.shape:
            return

        self.__store_state()
        self.__cells = cells
        self.__neighbour_count = ConwayGoLGrid.__create_neighbour_count(self.__cells)
        self.__alive_cells = set(map(tuple, np.transpose((self.__cells).nonzero())))
        self.__alive_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        self.__alive_duration = {}
        self.__iteration = 0
        self.__draw_grid()

    def overlay_new_cells(self, new_cells: np.ndarray, redraw: bool) ->  None:
        """Overlay the grid with another grid, only taking over the alive cells"""
        if new_cells.shape != self.__cells.shape:
            return

        self.__store_state()
        self.__cells = np.logical_or(self.__cells, new_cells)
        self.__alive_cells = set(map(tuple, np.transpose((self.__cells).nonzero())))
        self.__alive_cells.add(NUMBA_EMPTY_PLACEHOLDER)
        _ = [self.__alive_duration.pop((row, col), None)
                for row, col in np.ndindex(new_cells.shape)
                    if (row, col) in self.__alive_cells]

        self.__neighbour_count = ConwayGoLGrid.__create_neighbour_count(self.__cells)

        if redraw:
            self.__draw_grid()

    def purge(self, purge_trigger: int) -> None:
        """Purge all alive cells that have not updated their status for the given
        number of iterations. Basically, kill off the stragglers"""
        self.__store_state()
        for cell in list(self.__alive_duration.keys()):
            if self.__alive_duration[cell] >= purge_trigger:
                self.__alive_duration.pop(cell, None)
                if cell != NUMBA_EMPTY_PLACEHOLDER:
                    self.__cells[cell] = False
                    self.__alive_cells.discard(cell)

        self.__neighbour_count = ConwayGoLGrid.__create_neighbour_count(self.__cells)
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

        backup: Settings = self.__backups.pop()
        self.__background_color = backup.background_color
        self.__grid_color = backup.grid_color
        self.__cell_color = backup.cell_color
        self.__cells = copy(backup.cells)
        self.__neighbour_count = copy(backup.neighbour_count)
        self.__alive_cells = copy(backup.alive_cells)
        self.__alive_duration = copy(backup.alive_duration)
        self.__iteration = backup.iteration
        self.__draw_grid()

    def change_cell_color(self, color: tuple[int, int, int]):
        """Changes the color of a new cell"""
        self.__store_state()
        self.__cell_color = color
        self.__draw_all_cells()

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __create_neighbour_count(cells: np.ndarray) -> np.ndarray:
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
        backup = Settings(self.__background_color, self.__grid_color, self.__cell_color,
                          copy(self.__cells),copy(self.__alive_cells),
                          copy(self.__neighbour_count), copy(self.__alive_duration),
                          self.__iteration)
        self.__backups.append(backup)

    def __update_alive_duration(self) -> None:
        """Update the dictionary holding the lifetime of alive cells"""
        for cell in self.__alive_cells:
            if cell in self.__alive_duration:
                self.__alive_duration[cell] += 1
            else:
                self.__alive_duration[cell] = 0

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
        """Draws the new cells in the grid"""
        cells_to_redraw = [(i, j, self.__cell_color) for i, j in self.__alive_cells]
        self.__draw_cells(cells_to_redraw)
