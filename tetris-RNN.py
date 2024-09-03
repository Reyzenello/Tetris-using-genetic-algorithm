import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Game dimensions
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]

# Colors for each shape
SHAPE_COLORS = [CYAN, YELLOW, MAGENTA, RED, GREEN, BLUE, ORANGE]

class Tetromino:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = SHAPE_COLORS[SHAPES.index(shape)]
        self.rotation = 0

class TetrisAI:
    def __init__(self, game):
        self.game = game
        self.gamma = 0.9  # Discount factor for future rewards

    def get_state_value(self, state):
        # Simplified state evaluation
        holes = self.count_holes(state)
        bumpiness = self.calculate_bumpiness(state)
        height = self.calculate_height(state)
        lines_cleared = self.game.lines_cleared

        # Bellman equation: V(s) = R(s) + γ * max(V(s'))
        # Where R(s) is the immediate reward and V(s') is the estimated future value
        immediate_reward = lines_cleared * 100 - holes * 10 - bumpiness * 5 - height * 2
        future_value = self.gamma * (lines_cleared * 100)  # Simplified future value estimation

        return immediate_reward + future_value

    def count_holes(self, state):
        holes = 0
        for col in range(GRID_WIDTH):
            block_found = False
            for row in range(GRID_HEIGHT):
                if state[row][col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def calculate_bumpiness(self, state):
        heights = [0] * GRID_WIDTH
        for col in range(GRID_WIDTH):
            for row in range(GRID_HEIGHT):
                if state[row][col] != 0:
                    heights[col] = GRID_HEIGHT - row
                    break
        return sum(abs(heights[i] - heights[i+1]) for i in range(GRID_WIDTH-1))

    def calculate_height(self, state):
        return max(GRID_HEIGHT - row for col in range(GRID_WIDTH) for row in range(GRID_HEIGHT) if state[row][col] != 0)

    def choose_action(self):
        best_action = None
        best_value = float('-inf')

        for rotation in range(4):
            for x in range(GRID_WIDTH):
                test_piece = Tetromino(x, 0, self.game.current_piece.shape)
                test_piece.rotation = rotation
                if self.game.is_valid_position(test_piece):
                    # Simulate dropping the piece
                    while self.game.is_valid_position(test_piece):
                        test_piece.y += 1
                    test_piece.y -= 1

                    # Create a copy of the game state and apply the move
                    test_state = [row[:] for row in self.game.grid]
                    self.game.merge_piece(test_state, test_piece)
                    
                    # Evaluate the resulting state
                    state_value = self.get_state_value(test_state)

                    if state_value > best_value:
                        best_value = state_value
                        best_action = (rotation, x)

        return best_action

class TetrisGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris with Bellman Equation")
        self.clock = pygame.time.Clock()
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = None
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.ai = TetrisAI(self)

    def new_piece(self):
        shape = random.choice(SHAPES)
        self.current_piece = Tetromino(GRID_WIDTH // 2 - len(shape[0]) // 2, 0, shape)

    def draw_grid(self):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, cell, (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.screen, WHITE, (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_piece(self, piece):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, piece.color, 
                                     ((piece.x + x) * BLOCK_SIZE, (piece.y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.screen, WHITE, 
                                     ((piece.x + x) * BLOCK_SIZE, (piece.y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    def is_valid_position(self, piece):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    if (piece.x + x < 0 or piece.x + x >= GRID_WIDTH or
                        piece.y + y >= GRID_HEIGHT or
                        (piece.y + y >= 0 and self.grid[piece.y + y][piece.x + x])):
                        return False
        return True

    def merge_piece(self, grid, piece):
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid[piece.y + y][piece.x + x] = piece.color

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if all(row)]
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        return len(lines_to_clear)

    def rotate_piece(self, piece):
        piece.shape = list(zip(*piece.shape[::-1]))
        if not self.is_valid_position(piece):
            piece.shape = list(zip(*piece.shape))[::-1]

    def run(self):
        self.new_piece()
        drop_time = 0
        while not self.game_over:
            self.clock.tick(18000000)
            drop_time += self.clock.get_rawtime()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

            # AI decision making
            action = self.ai.choose_action()
            if action:
                rotation, target_x = action
                while self.current_piece.rotation != rotation:
                    self.rotate_piece(self.current_piece)
                self.current_piece.x = target_x

            if drop_time > 500:  # Move piece down every 500ms
                self.current_piece.y += 1
                if not self.is_valid_position(self.current_piece):
                    self.current_piece.y -= 1
                    self.merge_piece(self.grid, self.current_piece)
                    lines_cleared = self.clear_lines()
                    self.score += lines_cleared * 100
                    self.lines_cleared += lines_cleared
                    self.new_piece()
                    if not self.is_valid_position(self.current_piece):
                        self.game_over = True
                drop_time = 0

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_piece(self.current_piece)
            pygame.display.flip()

        print(f"Game Over! Score: {self.score}")
        pygame.quit()

if __name__ == "__main__":
    game = TetrisGame()
    game.run()
