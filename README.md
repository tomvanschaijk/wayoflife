My implementation of Conway's Game of Life. The twist in this implementation is that it doesn't only consider cells to be alive or dead. A cell can have several states:

* New: a cell that just became alive because of the game rules
* Survivor: an alive cell that hasn't changed state in the current iteration
* Dead: a celll that is marked as dead, and will be removed from the game in the next iteration

This small change in how cells are visualised allows you to follow the evolution of each cell and the environment as a whole. Coupled with that, there are a number of inputs you can perform to interact with the grid:
* the game can be paused at any time
* a random cell layout can be generated
* several pre-built patterns can be added to the grid
* new cells can be injected into the grid
* alive and dead cells can switch places
* all surviving cells can be removed
* surviving cells that haven't change states in x iterations can be removed
* while the game is paused, you can move through the states one step at a time
* it's possible to revert x amount of previous steps
* colors of all cells can be changed or reset to default values
* cell size can be changed to a predefined set of sizes
* target framerate can be changed

The develop branch contains an implementation as explained above. There is also a just_2_states branch, which (unsurprisingly) reduces the implementation to just having 2 cell states: alive or dead. This is basically the implementation that is usually found in implementations online. The develop-branch does give you more opportunities to inspect the steps the algorithm performs, by stepping through each step of the algorithm, rewinding, increasing the cell size and so forth. The just_2_states branch dials down on a bit of these options, but yields a better framerate when the cell size is 3x3 or 5x5, when a lot of cells are active.

Throughout every commit, the implementation moves from a basic implementation to being more optimized. The obvious addition of Numba to speed up the functions that do the heavy lifting has been done. Additionally, some efforts were made to reduce the search space, such that not every cell is considered during the update step, where the next situation of the game is calculated. Lastly, counting the active neighbour cells during the update step is quite a costly operation. This cost has been moved outside of the update operation for a part. The speed gains after each step were quite obvious, and were measured using line_profiler. The results can be replicated by running the included profiling script. A few of these results can be found in the included txt files.

Obviously, there's a tipping point where some added performance just pushes the complexity over the boundaries of what's "acceptable" for a simple project such as this. In Python, that tipping point is reached quite early because of the nature of the language. Since I used this project as a learning ground for getting back into Python, it was a fun challenge to get it to run at a decent framerate for the smaller cell sizes, without revering to measures like parallelization and extreme caching. Those would obviously result in big gains, but would probably push the complexity of the code (especially given my limited expertise in Python) over a certain edge. Although this is my small playground, and I would probably be the only one reading the code, pushing the extremes is not worth my time at this point, especially not since I have to realize that, inherently, I'm fighting against a language that's by nature too slow for these kinds of problems. Still, I've had fun with this ;-)
