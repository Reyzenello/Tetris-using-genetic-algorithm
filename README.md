# Tetris-using-genetic-algorithm

This code implements a Tetris game with an AI player using the Bellman equation for decision-making. 

**1. Initialization and Setup:**

- Imports Pygame, `random`, and `numpy`.
- Initializes Pygame.
- Defines colors and game dimensions.
- Defines the `SHAPES` of the Tetrominoes as lists of lists, representing their block arrangements.
- Defines corresponding `SHAPE_COLORS`.

**2. `Tetromino` Class:**

```python
class Tetromino:
    # ...
```

Represents a single Tetromino piece.  Holds its position (`x`, `y`), shape, color, and rotation.

**3. `TetrisAI` Class:**

```python
class TetrisAI:
    # ...
```

This is where the AI logic resides.

- `__init__`:  Initializes the AI with a reference to the `TetrisGame` and a discount factor (`gamma`) for future rewards in the Bellman equation.
- `get_state_value`:  Evaluates a given game state using a simplified heuristic.
    - Calculates the number of holes, bumpiness (height differences between adjacent columns), and height of the highest block.
    - Calculates an immediate reward based on lines cleared and penalties for holes, bumpiness, and height.
    - Estimates a simplified future value (assuming the current number of lines cleared is maintained).
    - Applies the Bellman equation to combine the immediate reward and discounted future value.  This is a simplified implementation; a true Bellman equation would consider all possible future states and their probabilities.
- `count_holes`, `calculate_bumpiness`, `calculate_height`:  Helper functions to calculate the state evaluation metrics.
- `choose_action`: This is the core decision-making function.
    - Iterates through all possible rotations and x-positions for the current piece.
    - Simulates dropping the piece in each position.
    - Creates a copy of the game grid and applies the simulated move.
    - Evaluates the resulting state using `get_state_value`.
    - Keeps track of the action that leads to the highest state value.
    - Returns the best action (rotation, x-position).

**4. `TetrisGame` Class:**

```python
class TetrisGame:
    # ...
```

Handles game logic, rendering, and user input.

- `__init__`: Initializes the game screen, clock, grid, current piece, game over status, score, lines cleared, and the AI player.
- `new_piece`: Creates a new random `Tetromino`.
- `draw_grid`, `draw_piece`: Functions for drawing the grid and the current piece using Pygame.
- `is_valid_position`: Checks if a given piece position is valid (within bounds and not overlapping existing blocks).
- `merge_piece`: Merges the current piece into the game grid.
- `clear_lines`: Clears completed lines and updates the score.
- `rotate_piece`: Rotates the current piece.
- `run`: The main game loop.
    - Creates a new piece.
    - Handles events (like quitting).
    - Calls the AI's `choose_action` to get the best move.
    - Applies the AI's chosen rotation and x-position.
    - Moves the piece down at regular intervals.
    - Handles game over conditions.
    - Updates and renders the game state.

**5. Main Execution:**

- Creates a `TetrisGame` instance and runs the game.
