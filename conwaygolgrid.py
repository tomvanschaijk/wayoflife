"""Grid object"""
import numpy as np
import pygame as pg
from numba import njit


class ConwayGoLGrid():
    """Class representing the grid for Conway's Game of Life

    To construct, use the static method 'new' instead of the default public constructor
    for built-in validation of the values
    """

    def __init__(self, cell_size: int, width: int, height: int,
                 background_color: tuple[int, int, int], grid_color: tuple[int, int, int],
                 new_color: tuple[int, int, int], survivor_color: tuple[int, int, int],
                 dead_color: tuple[int, int, int]):
        """Initialize all we need to start running the game"""
        self.__cell_size = cell_size
        self.__width = width
        self.__height = height
        self.__surface = pg.Surface((width, height))
        self.__background_color = background_color
        self.__grid_color = grid_color
        self.__new_color = new_color
        self.__survivor_color = survivor_color
        self.__dead_color = dead_color
        self.__cells: np.ndarray
        self.__new_cells: set[tuple[(int, int)]]
        self.__survivor_cells: set[tuple[(int, int)]]
        self.__dead_cells: set[tuple[(int, int)]]
        self.__rows: int
        self.__columns: int
        self.reset()

    @property
    def cell_size(self) -> int:
        """The size of each square cell in the grid"""
        return self.__cell_size

    @property
    def rows(self) -> int:
        """The amount of rows in the grid"""
        return self.__rows

    @property
    def columns(self) -> int:
        """The amount of columns in the grid"""
        return self.__columns

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the grid"""
        return (self.__rows, self.__columns)

    @property
    def surface(self) -> pg.Surface:
        """The surface the game is drawn onto"""
        return self.__surface

    @staticmethod
    def new(cell_size: int, width: int, height: int, background_color: tuple[int, int, int],
            grid_color: tuple[int, int, int], new_color: tuple[int, int, int],
            survivor_color: tuple[int, int, int], dead_color: tuple[int, int, int]):
        """Creates a new grid after successful validation of values"""
        # possible todo:  validate the values sent in, wrap the value in a "Result" object
        # holding not only the Grid if succesfully created, but also other values such
        # as a bool value signifying successful creation, a list of errors/warnings, ...
        return ConwayGoLGrid(cell_size, width, height, background_color, grid_color,
                             new_color, survivor_color, dead_color)

    def update(self) -> None:
        """Updates the grid to the next step in the iteration, following Conway's Game
        of Life rules. Evaluates the grid, and redraws all changed cells"""
        (updated_cells,
         cells_to_redraw) = ConwayGoLGrid.__perform_update(self.__cells, self.__new_cells,
                                                         self.__survivor_cells, self.__dead_cells,
                                                         self.__rows, self.__columns,
                                                         self.__new_color, self.__survivor_color,
                                                         self.__dead_color, self.__background_color)
        self.__cells = updated_cells
        self.__draw_cells(cells_to_redraw)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __perform_update(cells: np.ndarray, new_cells: set[tuple[int, int]],
                         survivor_cells: set[tuple[int, int]], dead_cells: set[tuple[int, int]],
                         rows: int, columns: int, new_color: tuple[int, int, int],
                         survivor_color: tuple[int, int, int], dead_color: tuple[int, int, int],
                         background_color: tuple[int, int, int]
                         ) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int, int]]]]:
        """Updates the grid to the next step in the iteration, following Conway's Game of Life
        rules. Evaluates each cell, and returns the list of cells to be redrawn and their colors
        """
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
        cells_to_redraw: list[tuple[int, int, tuple[int, int, int]]] = []
        for row, col in cells_to_evaluate:
            cell = (row, col)
            # Count the alive cells around the current cell
            alive_neighbours = (np.sum(cells[max(0, row-1):row+2, max(0, col-1):col+2])
                                - cells[cell])
            if cells[cell]:
                if alive_neighbours in (2, 3):
                    updated_cells[cell] = True
                    cells_to_redraw.append((row, col, survivor_color))
                    survivor_cells.add(cell)
                    new_cells.discard(cell)
                    dead_cells.discard(cell)
                else:
                    cells_to_redraw.append((row, col, dead_color))
                    dead_cells.add(cell)
                    new_cells.discard(cell)
                    survivor_cells.discard(cell)
            else:
                if alive_neighbours == 3:
                    updated_cells[cell] = True
                    cells_to_redraw.append((row, col, new_color))
                    new_cells.add(cell)
                    survivor_cells.discard(cell)
                    dead_cells.discard(cell)
                else:
                    cells_to_redraw.append((row, col, background_color))
                    dead_cells.discard(cell)
        return updated_cells, cells_to_redraw

    def reset(self) -> None:
        """Resets the entire grid"""
        self.__rows = self.__height // self.__cell_size
        self.__columns = self.__width // self.__cell_size
        self.__cells = np.full((self.__height // self.__cell_size,
                                self.__width // self.__cell_size),
                               False, dtype=bool)
        # The initial entry (-99, -99) is added since Numba can not handle empty sets
        self.__new_cells = set([(-99, -99)])
        self.__survivor_cells = set([(-99, -99)])
        self.__dead_cells = set([(-99, -99)])
        self.__draw_grid()

    def resurrect_cell(self, coordinates: tuple[int, int]) -> None:
        """Brings a cell in the grid to life"""
        if coordinates in self.__new_cells:
            return

        self.__cells[coordinates] = True
        self.__new_cells.add(coordinates)
        self.__survivor_cells.discard(coordinates)
        self.__dead_cells.discard(coordinates)
        self.__draw_cells([(coordinates[0], coordinates[1], self.__new_color)])

    def clear_cell(self, coordinates: tuple[int, int]) -> None:
        """Clears a cell from the grid"""
        self.__cells[coordinates] = False
        self.__new_cells.discard(coordinates)
        self.__survivor_cells.discard(coordinates)
        self.__dead_cells.discard(coordinates)
        self.__draw_cells([(coordinates[0], coordinates[1], self.__background_color)])

    def create_cell_layout(self, cells: np.ndarray) ->  None:
        """Creates a certain grid layout to continue the game from"""
        if cells.shape != (self.__rows, self.__columns):
            return

        self.__cells = cells.copy()
        self.__new_cells = set([(-99, -99)])
        for coordinate in np.ndindex(cells.shape):
            if cells[coordinate]:
                self.__new_cells.add(coordinate)
        self.__survivor_cells = set([(-99, -99)])
        self.__dead_cells = set([(-99, -99)])
        self.__draw_grid()

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
        cells_to_redraw: list[tuple[int, int, tuple[int, int, int]]] = []
        cells_to_redraw = [(row, col, self.__new_color)
                            for row, col in np.ndindex(self.__cells.shape)
                                if self.__cells[row, col]]
        self.__draw_cells(cells_to_redraw)
