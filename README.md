[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"

# Continuous control

## Introduction

Continuous control is the second project of a Deep Reinforcement Learning Nanodegree Program. 
For this project, an agent will work with the [Reacher](https://github.com/ChalamPVS/Unity-Reacher) environment. 
In this environment, a double-jointed arm can move to target locations and its goal is to maintain its position
at the target location for as many time steps as possible.

![Trained Agent][image1]

## Project Details

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action 
vector should be a number between -1 and 1. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.


There are two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) 
copies of the same agent to distribute the task of gathering experience. 


When it comes to solving the environment, there are two options based on chosen version of the environment:
- **Option 1**: Solve the First Version 
  - The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.
- **Option 2:** Solve the Second Version
  - The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. 
  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
    - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
    - This yields an average score for each episode (where the average is over all 20 agents).
    
## Getting Started

- **Step 1** - Set up your Python environment
  - Python 3.6 was used in the project
  - Type `pip install -r requirements.txt` in command line to install requirements
- **Step 2** - Build the Unity Environment 
  - Download the environment:
    - [One Agent](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - [20 Agents](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)

  - Unzip the environment and copy it to the main folder of the project. Note that provided environment will work only on Linux based machines!
- **Step 3** - Check `main.py` and familiarize yourself with arguments and hyper-parameters.
- **Step 4** - Run `main.py`
