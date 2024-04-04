# Omok (Gomoku) RL agent

>[!WARNING]
> This repository is still under active development.

## Description
Training Omok AI with OpenAI Gym and Stable-Baselines3
This project is built on OpenAI Gym and Stable-Baselines3, providing a framework for two agents to compete and learn from each other, enhancing their strategies over time.

## Overview
Omok, also known as Gomoku, is a strategy board game traditionally played with Go pieces (black and white stones) on a Go board. The objective is to be the first to place five of one's own pieces in a row, either horizontally, vertically, or diagonally.

In this project, I have developed a training environment where two AI agents compete against each other in the game of Omok. The agents are trained using reinforcement learning techniques provided by Stable-Baselines3, integrated within an OpenAI Gym environment. To accelerate learning, one agent is periodically cloned from the other, ensuring that both agents evolve competitive strategies against increasingly skilled opponents.

## Features
### Custom Omok Environment
An OpenAI Gym-based environment for the Omok game.

### Reinforcement Learning Agents
Utilization of Stable-Baselines3 for training agents with advanced reinforcement learning algorithms.

### Competition-Driven Learning
Agents learn by competing against a clone of themselves.

*Periodic Cloning*: One agent is periodically cloned from the other to maintain a challenging learning environment and prevent stagnation.

## Demo
https://github.com/hi-jin/omok-RL/assets/51053567/5344ed03-a841-454a-ad88-c5897a8e065f

## Milestones
- [x] Implement the Omok Game
- [x] Wrap the Omok Game in a Gym Environment
- [x] Implement Learning Agents that compete
- [ ] Enhance performance with algorithm adjustments
- [ ] Implement various Omok rules
- [ ] Develop an API for the game
- [ ] Create a website for playing against the Omok AI

## Known Problems
- [ ] As the number of training episodes increases, both agents tend to place their pieces in the same positions repeatedly, leading to repetitive game outcomes in subsequent episodes.
- [ ] Agent sometimes tries to place the pieces on a wrong place (e.g. on the non-empty places).
