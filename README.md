# Deep Q-Network Agent for Banana Navigation

## Project Overview

This project implements a Deep Q-Network (DQN) agent that learns to navigate a Unity ML-Agents environment and collect bananas. The agent must learn to collect yellow bananas (reward: +1) while avoiding blue bananas (reward: -1) in a large square world.

The implementation demonstrates core deep reinforcement learning concepts including:
- Q-learning with neural network function approximation
- Experience replay for improved learning stability
- Target networks for decoupled learning
- Epsilon-greedy exploration strategy

## Environment Details

### State Space
- **Dimension**: 37 continuous values
- **Description**: The state space contains the agent's velocity and ray-based perception of objects around its forward direction. This ray-based information allows the agent to perceive obstacles, walls, and banana positions.

### Action Space
The agent can take one of 4 discrete actions at each time step:
- **Action 0**: Move forward
- **Action 1**: Move backward
- **Action 2**: Turn left
- **Action 3**: Turn right

### Reward Structure
- **+1**: Collecting a yellow banana
- **-1**: Collecting a blue banana
- **0**: All other actions (movement, turning)

### Goal / Solving the Environment
The environment is considered **solved** when the agent achieves an **average score of at least +13 over 100 consecutive episodes**.

The implemented DQN agent solves this environment in approximately 368 episodes.

## Getting Started

### Prerequisites

1. Install dependencies by following these instructions: [Dependencies](https://github.com/udacity/Value-based-methods#dependencies) 

1. Download the unity banana app following these instructions: [Getting Started](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation#getting-started)

## Instructions

### Training the Agent

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Navigation.ipynb` in your browser

3. Follow the instructions in the cell.

> **_NOTE:_** If you just want to test the pretrained model, you can just run the last cell in the jupytor notebook.



## Algorithm Overview


For detailed algorithmic explanation, see `Report.md`.


