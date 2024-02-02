import pygame
import sys
import pandas as pd
import numpy as np

# Define constants for the maze
GRID_SIZE = 20
MAZE_WIDTH, MAZE_HEIGHT = 400, 400
AGENT_SIZE = 20

# Define rewards and penalties
REWARD = 1
PENALTIES = 1
REWARD_COUNT = 0
PENALTY_COUNT = 0

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 192, 203)

# Define start and goal positions
start_position = (AGENT_SIZE, AGENT_SIZE)
goal_position = (MAZE_WIDTH - (AGENT_SIZE * 2), MAZE_HEIGHT - (AGENT_SIZE * 2))

traversed_path = []
traversed_path.append(start_position)

# Defining coordinates for inner walls
walls_df = pd.read_csv("walls.csv")
rewards_df = pd.read_csv("rewards.csv")
penalties_df = pd.read_csv("penalties.csv")

walls = []
not_walls = []
rewards = []
penalties = []

for i in range(len(walls_df)):
    x, y = walls_df['x_position'][i], walls_df['y_position'][i]
    walls.append((start_position[0] + (AGENT_SIZE * int(x)), start_position[1] + (AGENT_SIZE * int(y))))

for x in range(0, MAZE_WIDTH):
    walls.append((x, 0))
    walls.append((x, MAZE_HEIGHT - AGENT_SIZE))

for y in range(0, MAZE_HEIGHT):
    walls.append((0, y))
    walls.append((MAZE_WIDTH - AGENT_SIZE, y))

for cols in range(0, MAZE_WIDTH, AGENT_SIZE):
    for rows in range(0, MAZE_HEIGHT, AGENT_SIZE):
        if (cols, rows) not in walls:
            not_walls.append((cols, rows))
            
for i in range(len(rewards_df)):
    x, y = rewards_df['reward_x'][i], rewards_df['reward_y'][i]
    rewards.append((start_position[0] + (AGENT_SIZE * int(x)), start_position[1] + (AGENT_SIZE * int(y))))

for i in range(len(penalties_df)):
    x, y = penalties_df['penalty_x'][i], penalties_df['penalty_y'][i]
    penalties.append((start_position[0] + (AGENT_SIZE * int(x)), start_position[1] + (AGENT_SIZE * int(y))))

# Load the trained Q-table
Q_table = np.load("model.npy")

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((MAZE_WIDTH, MAZE_HEIGHT))
pygame.display.set_caption("Maze Navigation")

class Maze:
    def __init__(self, screen, position: tuple):
        self.screen = screen
        self.agent_position = position
    
    def draw_maze(self):
        global rewards, penalties
        global traversed_path
        
        screen.fill(WHITE)

        draw_grids()
        draw_path_forming_walls(walls)
        draw_outer_walls()

        for (x, y) in traversed_path:
            pygame.draw.rect(screen, BLUE, (x, y, AGENT_SIZE, AGENT_SIZE))
        
        for (x, y) in rewards:
            pygame.draw.rect(screen, YELLOW, (x, y, AGENT_SIZE, AGENT_SIZE))
            
        for (x, y) in penalties:
            pygame.draw.rect(screen, PINK, (x, y, AGENT_SIZE, AGENT_SIZE))
        
        # Draw start and goal positions
        pygame.draw.rect(screen, GREEN, (self.agent_position[0], self.agent_position[1], AGENT_SIZE, AGENT_SIZE))
        pygame.draw.rect(screen, RED, (goal_position[0], goal_position[1], AGENT_SIZE, AGENT_SIZE))

        pygame.display.flip()

    def move_agent(self, action):
        x, y = self.agent_position

        if action == 0:  # left
            x -= AGENT_SIZE
        elif action == 1:  # right
            x += AGENT_SIZE
        elif action == 2:  # up
            y -= AGENT_SIZE
        elif action == 3:  # down
            y += AGENT_SIZE

        new_position = (x, y)
        return new_position

    def get_actions(self):
        options = []

        if (self.agent_position[0] - AGENT_SIZE, self.agent_position[1]) in not_walls:
            options.append(0)  # left
        if (self.agent_position[0] + AGENT_SIZE, self.agent_position[1]) in not_walls:
            options.append(1)  # right
        if (self.agent_position[0], self.agent_position[1] - AGENT_SIZE) in not_walls:
            options.append(2)  # up
        if (self.agent_position[0], self.agent_position[1] + AGENT_SIZE) in not_walls:
            options.append(3)  # down
        
        return options

    def take_action(self, action):
        # Take the chosen action and observe the next state
        next_position = self.move_agent(action)
        self.done_traversing()
        return next_position

    def done_traversing(self):
        global traversed_path
        global REWARD_COUNT, PENALTY_COUNT
        
        if self.agent_position in rewards:
            rewards.remove(self.agent_position)
            REWARD_COUNT += REWARD
        elif self.agent_position in penalties:
            penalties.remove(self.agent_position)
            PENALTY_COUNT += PENALTIES

        traversed_path.append(self.agent_position)

def draw_outer_walls():
    for x in range(0, MAZE_WIDTH):
        pygame.draw.rect(screen, BLACK, (x, 0, AGENT_SIZE, AGENT_SIZE))
        pygame.draw.rect(screen, BLACK, (x, MAZE_HEIGHT - AGENT_SIZE, AGENT_SIZE, AGENT_SIZE))
        
    for y in range(0, MAZE_HEIGHT):
        pygame.draw.rect(screen, BLACK, (0, y, AGENT_SIZE, AGENT_SIZE))
        pygame.draw.rect(screen, BLACK, (MAZE_WIDTH - AGENT_SIZE, y, AGENT_SIZE, AGENT_SIZE))

def draw_path_forming_walls(coordinates):
    for coordinate in coordinates:
        pygame.draw.rect(screen, BLACK,
            (
                coordinate[0],
                coordinate[1],
                AGENT_SIZE,
                AGENT_SIZE
            )
        )

def draw_grids():
    # Draw grid lines
    for x in range(0, MAZE_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, MAZE_HEIGHT))
    for y in range(0, MAZE_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (MAZE_WIDTH, y))

# Main game loop
def game_loop():
    maze = Maze(screen, start_position)
    state = not_walls.index(start_position)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Choose action using the learned Q-table
        available_actions = maze.get_actions()
        action = available_actions[np.argmax(Q_table[state, available_actions])]

        # Take the chosen action and observe the next state
        next_position = maze.take_action(action)

        # Update the agent's position and draw the maze
        maze.agent_position = next_position
        maze.draw_maze()

        # Check if the agent has reached the goal
        if next_position == goal_position:
            print("Agent reached the goal!")
            print(f"REWARD COUNT = {REWARD_COUNT}\nPENALTY COUNT = {PENALTY_COUNT}")
            pygame.quit()
            sys.exit()

        pygame.time.delay(200)  # Add a delay to visualize the movement

        state = not_walls.index(next_position)

# Run the main game loop
game_loop()
