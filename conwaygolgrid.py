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
        self.__cells = np.zeros((height // cell_size, width // cell_size), dtype=int)
        self.__cell_size = cell_size
        self.__width = width
        self.__height = height
        self.__rows = height // cell_size
        self.__columns = width // cell_size
        self.__surface = pg.Surface((width, height))
        self.__background_color = background_color
        self.__grid_color = grid_color
        self.__new_color = new_color
        self.__survivor_color = survivor_color
        self.__dead_color = dead_color
        self.__draw_grid()

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
         cells_to_redraw) = ConwayGoLGrid.__perform_update(self.__cells, self.__background_color,
                                                           self.__new_color, self.__survivor_color,
                                                           self.__dead_color)

        self.__cells = updated_cells
        self.__draw_cells(cells_to_redraw)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def __perform_update(cells: np.ndarray, background_color: tuple[int, int, int],
                         new_color: tuple[int, int, int], survivor_color: tuple[int, int, int],
                         dead_color: tuple[int, int, int]
                         ) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int, int]]]]:
        """Updates the grid to the next step in the iteration, following Conway's Game of Life
        rules. Evaluates each cell, and returns the list of cells to be redrawn and their colors
        """
        cells_to_redraw: list[tuple[int, int, tuple[int, int, int]]] = []
        updated_cells = np.array([[0] * cells.shape[1]] * cells.shape[0])
        for row, col in np.ndindex(cells.shape):
            alive_neighbours = (np.sum(cells[max(0, row-1):row+2, max(0, col-1):col+2])
                                - cells[row, col])
            if cells[row, col] == 1:
                if alive_neighbours == 2 or alive_neighbours == 3:
                    updated_cells[row, col] = 1
                    cells_to_redraw.append((row, col, survivor_color))
                else:
                    cells_to_redraw.append((row, col, dead_color))
            else:
                if alive_neighbours == 3:
                    updated_cells[row, col] = 1
                    cells_to_redraw.append((row, col, new_color))
                else:
                    cells_to_redraw.append((row, col, background_color))

        return updated_cells, cells_to_redraw

    def resurrect_cell(self, coordinates: tuple[int, int]) -> None:
        """Brings a cell in the grid to life"""
        row, col = coordinates
        self.__cells[coordinates] = 1
        self.__draw_cells([(row, col, self.__new_color)])

    def clear_cell(self, coordinates: tuple[int, int]) -> None:
        """Clears a cell from the grid"""
        row, col = coordinates
        self.__cells[row, col] = 0
        self.__draw_cells([(row, col, self.__background_color)])

    def create_cell_layout(self, cells: np.ndarray) ->  None:
        """Creates a certain grid layout to continue the game from"""
        if cells.shape != (self.__rows, self.__columns):
            return

        self.__cells = cells
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
        for row, col in np.ndindex(self.__cells.shape):
            if(self.__cells[row, col]):
                cells_to_redraw.append((row, col, self.__new_color))
        self.__draw_cells(cells_to_redraw)
