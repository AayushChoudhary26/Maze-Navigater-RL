import pygame
import sys
import pandas as pd
import numpy as np

# Define constants for the maze
GRID_SIZE = 20
MAZE_WIDTH, MAZE_HEIGHT = 400, 400
AGENT_SIZE = 20
STEPS_TAKEN = 0

# Delay in visual information
visual_delay = 20

# Define parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.7  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off

# Define rewards
REWARD = 5
REWARD_FOR_NEW_PATH_EXPLORATION = 20
REWARD_FOR_REACHING_GOAL = 100

# Define penalties
PENALTIES = 5
PENALTY_COUNT = 0
PENALTY_MAX_THRESHOLD = 2000
PENALTY_FOR_RETRACING_STEPS = 10

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

# Define Q-table
num_states = len(not_walls)
num_actions = 4  # left, right, up, down
Q_table = np.zeros((num_states, num_actions))

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((MAZE_WIDTH, MAZE_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Maze Navigation")

class Maze:
    def __init__(self, screen, position: tuple, step: int):
        self.screen = screen
        self.agent_position = position
        self.steps_taken = step
    
    def draw_maze(self, state):
        global rewards, penalties
        global traversed_path
        
        screen.fill(WHITE)
        self.agent_position = not_walls[state]

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
        pygame.time.delay(visual_delay)

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
        self.done_traversing()
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

    def reset(self):
        global PENALTY_COUNT, REWARDS_GOT, PENALTIES_GOT
        global traversed_path
        global rewards, penalties
        
        self.agent_position = start_position
        PENALTY_COUNT = 0
        REWARDS_GOT = 0
        PENALTIES_GOT = 0
        traversed_path = []
        
        for i in range(len(rewards_df)):
            x, y = rewards_df['reward_x'][i], rewards_df['reward_y'][i]
            rewards.append((start_position[0] + (AGENT_SIZE * int(x)), start_position[1] + (AGENT_SIZE * int(y))))

        for i in range(len(penalties_df)):
            x, y = penalties_df['penalty_x'][i], penalties_df['penalty_y'][i]
            penalties.append((start_position[0] + (AGENT_SIZE * int(x)), start_position[1] + (AGENT_SIZE * int(y))))

    def get_state(self):
        # Convert agent's position to a unique state index
        return not_walls.index(self.agent_position)

    def take_action(self, action):
        # Take the chosen action and observe the next state and reward
        next_position = self.move_agent(action)
        next_state = not_walls.index(next_position)
        reward = self.calculate_reward(next_position)
        return next_state, reward

    def calculate_reward(self, next_position):
        global PENALTY_COUNT
        global rewards, penalties
        
        # Calculate reward based on the next position
        if next_position == goal_position:
            # Calculate the reward multiplier based on the number of steps taken
            reward_multiplier = (1 / (1 + self.steps_taken))
            return REWARD_FOR_REACHING_GOAL * reward_multiplier
        
        elif next_position in rewards:
            rewards.pop(rewards.index(next_position))
            return REWARD
        
        elif next_position in penalties:
            PENALTY_COUNT += PENALTIES
            penalties.pop(penalties.index(next_position))
            return -PENALTIES
        
        elif next_position in traversed_path:
            PENALTY_COUNT += PENALTY_FOR_RETRACING_STEPS
            return -PENALTY_FOR_RETRACING_STEPS
        
        elif next_position not in traversed_path:
            return REWARD_FOR_NEW_PATH_EXPLORATION
        
        else:
            if PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD - 1) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 1/9)):
                return (REWARD * 2)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 1/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 2/9)):
                return (REWARD * 3)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 2/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 3/9)):
                return (REWARD * 4)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 3/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 4/9)):
                return (REWARD * 5)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 4/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 5/9)):
                return (REWARD * 6)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 5/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 6/9)):
                return (REWARD * 7)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 6/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 7/9)):
                return (REWARD * 8)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 7/9)) or PENALTY_COUNT > (PENALTY_MAX_THRESHOLD * (1 - 8/9)):
                return (REWARD * 9)
            elif PENALTY_COUNT <= (PENALTY_MAX_THRESHOLD * (1 - 8/9)) or PENALTY_COUNT > 10:
                return (REWARD * 10)
            elif PENALTY_COUNT <= 10 or PENALTY_COUNT > 9:
                return (REWARD * 11)
            elif PENALTY_COUNT <= 9 or PENALTY_COUNT > 8:
                return (REWARD * 12)
            elif PENALTY_COUNT <= 8 or PENALTY_COUNT > 7:
                return (REWARD * 13)
            elif PENALTY_COUNT <= 7 or PENALTY_COUNT > 6:
                return (REWARD * 14)
            elif PENALTY_COUNT <= 6 or PENALTY_COUNT > 5:
                return (REWARD * 15)
            elif PENALTY_COUNT <= 5 or PENALTY_COUNT > 4:
                return (REWARD * 16)
            elif PENALTY_COUNT <= 4 or PENALTY_COUNT > 3:
                return (REWARD * 17)
            elif PENALTY_COUNT <= 3 or PENALTY_COUNT > 2:
                return (REWARD * 18)
            elif PENALTY_COUNT <= 2 or PENALTY_COUNT > 1:
                return (REWARD * 19)
            elif PENALTY_COUNT <= 1 or PENALTY_COUNT > 0:
                return (REWARD * 20)
            elif PENALTY_COUNT == 0:
                return (REWARD * 30)
            else:
                return REWARD

    def done_traversing(self):
        global traversed_path
        
        traversed_path.append(self.agent_position)
        self.steps_taken += 1
        return self.steps_taken
    
    def is_done(self):
        if self.agent_position == goal_position:
            return True
        elif PENALTY_COUNT >= PENALTY_MAX_THRESHOLD:
            return True
        else:
            return False
    
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

# Training loop
def game_loop(max_episodes=100):
    for episode in range(max_episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        print(f"Episode: {episode + 1}")
        
        maze = Maze(screen, start_position, STEPS_TAKEN)
        maze.reset()  # Reset the environment for a new episode
        state = maze.get_state()

        while not maze.is_done():
            actions = maze.get_actions()
            
            # Choose action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                available_actions = maze.get_actions()
                action = available_actions[np.argmax(Q_table[state, available_actions])]

            # Take the chosen action and observe the next state and reward
            next_state, reward = maze.take_action(action)
            print(f"Agent Position: {next_state}\nActions Available: {actions}\nPenalties: {PENALTY_COUNT}\nRewards: {reward}\n")

            # Update Q-value using the Q-learning update rule
            Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
                        reward + gamma * np.max(Q_table[next_state, :]))

            # Move to the next state
            state = next_state
            maze.draw_maze(state)

    np.save("model.npy", Q_table)

# Main game loop
game_loop(100)
