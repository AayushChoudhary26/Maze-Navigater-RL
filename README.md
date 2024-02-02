# Maze Navigation with Reinforcement Learning

Welcome to the Maze Navigation project, where an intelligent agent learns to navigate a maze using Reinforcement Learning (RL).

## Overview

The project focuses on training an agent to move from a starting position to a goal while avoiding obstacles and seeking rewards within a maze environment. The agent's decision-making process is powered by Q-learning, a popular RL technique.

## Features

- **Customizable Maze:** Define the maze using CSV files for walls, rewards, and penalties. Easily modify the environment to suit your requirements.

- **Adaptive Decision-Making:** The agent dynamically chooses actions based on available options, showcasing adaptive decision-making capabilities.

- **Reinforcement Learning:** The agent learns from experiences, receiving rewards for reaching the goal and penalties for undesired actions. The Q-learning approach drives continuous improvement.

## Getting Started

1. Clone the repository:

   git clone https://github.com/AayushChoudhary26/Maze-Navigater-RL

2. Change directory to the cloned one:
   cd Maze-Navigater-RL
   
(Ignore step 3 and 4 if you want to run the trained model on existing csv files)
3. Customize the maze:
  Edit the CSV files (walls.csv, rewards.csv, penalties.csv) to define the maze's structure and rewards/penalties distribution.

4. Run the training loop:
  python train_model.py

5. Run the testing program:
   python test_model.py

Testing Output
After training, you will see testing output indicating the agent's performance, such as the number of steps taken, rewards obtained, and penalties incurred.

Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback on the project.


