import pygame
import random
import numpy as np

# Tetris game settings
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128)  # Purple
]

# Tetromino shapes
TETROMINOES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # J
    [[1, 1, 1], [0, 0, 1]],  # L
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]  # Z
]

# Genetic Algorithm parameters
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
NUM_GENERATIONS = 100


class Tetromino:
    def __init__(self, x, y, shape, color):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = color
        self.rotation = 0

    def draw(self, screen):
        for row in range(len(self.shape)):
            for col in range(len(self.shape[row])):
                if self.shape[row][col]:
                    pygame.draw.rect(screen, self.color,
                                     (self.x + col * BLOCK_SIZE,
                                      self.y + row * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)
        self.shape = list(zip(*self.shape[::-1]))

    def move_down(self):
        self.y += BLOCK_SIZE

    def move_left(self):
        self.x -= BLOCK_SIZE

    def move_right(self):
        self.x += BLOCK_SIZE


class Tetris:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris with Genetic Algorithm")
        self.clock = pygame.time.Clock()
        self.grid = [[0 for _ in range(GRID_WIDTH)]
                     for _ in range(GRID_HEIGHT)]
        self.current_piece = self.get_new_piece()
        self.next_piece = self.get_new_piece()
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0

    def get_new_piece(self):
        shape = random.choice(TETROMINOES)
        color = random.choice(COLORS)
        return Tetromino(GRID_WIDTH // 2 - len(shape[0]) // 2,
                         0, shape, color)

    def draw_grid(self):
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                   BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

    def draw_next_piece(self):
        x_offset = SCREEN_WIDTH - 150
        y_offset = 100
        for row in range(len(self.next_piece.shape)):
            for col in range(len(self.next_piece.shape[row])):
                if self.next_piece.shape[row][col]:
                    pygame.draw.rect(self.screen, self.next_piece.color,
                                     (x_offset + col * BLOCK_SIZE,
                                      y_offset + row * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))

    def check_collision(self, piece, x_offset=0, y_offset=0):
        for row in range(len(piece.shape)):
            for col in range(len(piece.shape[row])):
                if piece.shape[row][col]:
                    new_x = piece.x + col + x_offset
                    new_y = piece.y + row + y_offset
                    if (new_x < 0 or new_x >= GRID_WIDTH or
                            new_y >= GRID_HEIGHT or
                            (new_y >= 0 and self.grid[new_y][new_x])):
                        return True
        return False

    def lock_piece(self):
        for row in range(len(self.current_piece.shape)):
            for col in range(len(self.current_piece.shape[row])):
                if self.current_piece.shape[row][col]:
                    x = self.current_piece.x + col
                    y = self.current_piece.y + row
                    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:  # Bounds check
                        self.grid[y][x] = self.current_piece.color

        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = self.get_new_piece()
        if self.check_collision(self.current_piece):
            self.game_over = True

    def clear_lines(self):
        lines_to_clear = []
        for row in range(GRID_HEIGHT):
            if all(self.grid[row]):
                lines_to_clear.append(row)

        for row in lines_to_clear:
            del self.grid[row]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
            self.lines_cleared += 1
            self.score += 100

    def get_state(self):
        # Feature engineering for the state representation
        # You can add more features here (e.g., number of holes, etc.)
        heights = [0] * GRID_WIDTH
        for col in range(GRID_WIDTH):
            for row in range(GRID_HEIGHT):
                if self.grid[row][col]:
                    heights[col] = GRID_HEIGHT - row
                    break
        return np.array(heights)

    def run(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

            # Handle user input (for testing purposes)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and not self.check_collision(
                    self.current_piece, x_offset=-1):
                self.current_piece.move_left()
            if keys[pygame.K_RIGHT] and not self.check_collision(
                    self.current_piece, x_offset=1):
                self.current_piece.move_right()
            if keys[pygame.K_DOWN] and not self.check_collision(
                    self.current_piece, y_offset=1):
                self.current_piece.move_down()
            if keys[pygame.K_UP]:
                self.current_piece.rotate()

            # Game logic
            if not self.check_collision(self.current_piece, y_offset=1):
                self.current_piece.move_down()
            else:
                self.lock_piece()

            # Drawing
            self.screen.fill(BLACK)
            self.draw_grid()
            self.current_piece.draw(self.screen)
            self.draw_next_piece()

            pygame.display.flip()
            self.clock.tick(5)  # Adjust speed here

        pygame.quit()


# --- Genetic Algorithm ---

class Agent:
    def __init__(self, chromosome=None):
        if chromosome is None:
            # Initialize with random weights
            self.chromosome = np.random.rand(GRID_WIDTH)
        else:
            self.chromosome = chromosome

    def get_move(self, game):
        # Simple example: choose the move that minimizes the aggregate height
        best_move = None
        min_height = float('inf')
        for _ in range(4):  # Try all 4 rotations
            for x in range(GRID_WIDTH - len(game.current_piece.shape[0]) + 1):
                temp_piece = Tetromino(x, game.current_piece.y,
                                       game.current_piece.shape,
                                       game.current_piece.color)
                while not game.check_collision(temp_piece, y_offset=1):
                    temp_piece.move_down()

                state = game.get_state()
                height = np.sum(state * self.chromosome)
                if height < min_height:
                    min_height = height
                    best_move = (game.current_piece.rotation, x)

            game.current_piece.rotate()

        return best_move


def create_initial_population():
    return [Agent() for _ in range(POPULATION_SIZE)]


def selection(population):
    # Simple tournament selection
    tournament_size = 3
    selected = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda agent: agent.fitness)
        selected.append(winner)
    return selected


def crossover(parent1, parent2):
    # Single-point crossover
    crossover_point = random.randint(0, len(parent1.chromosome) - 1)
    child_chromosome = np.concatenate((parent1.chromosome[:crossover_point],
                                       parent2.chromosome[crossover_point:]))
    return Agent(chromosome=child_chromosome)


def mutation(agent):
    # Randomly flip bits with a certain probability
    for i in range(len(agent.chromosome)):
        if random.random() < MUTATION_RATE:
            agent.chromosome[i] = 1 - agent.chromosome[i]
    return agent


def calculate_fitness(agent):
    game = Tetris()
    while not game.game_over:
        move = agent.get_move(game)
        if move is not None:
            rotation, x = move

            # Perform piece rotations
            current_rotation = game.current_piece.rotation
            while current_rotation != rotation:
                game.current_piece.rotate()
                current_rotation = (current_rotation + 1) % len(
                    game.current_piece.shape)

            # Correct horizontal positioning (should be in grid units, not pixels)
            game.current_piece.x = x * BLOCK_SIZE

            while not game.check_collision(game.current_piece, y_offset=1):
                game.current_piece.move_down()
            game.lock_piece()
        else:
            # if no valid moves are available, end the game
            game.game_over = True
    return game.score


def run_genetic_algorithm():
    population = create_initial_population()
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        for agent in population:
            agent.fitness = calculate_fitness(agent)

        # Selection
        selected = selection(population)

        # Crossover and mutation
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)
            offspring.append(mutation(child1))
            offspring.append(mutation(child2))

        # Replace the old population
        population = offspring

        # Print best agent's fitness
        best_agent = max(population, key=lambda agent: agent.fitness)
        print(f"Generation {generation}: Best fitness = {best_agent.fitness}")

    return best_agent


if __name__ == "__main__":
    # Uncomment to run the Tetris game manually
    # game = Tetris()
    # game.run()

    # Uncomment to run the genetic algorithm
    best_agent = run_genetic_algorithm()
    print("Best agent's chromosome:", best_agent.chromosome)
